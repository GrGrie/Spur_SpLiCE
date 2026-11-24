"""Concept attribution at inference for one finished run (docs/PROJECT_CSA.md, Section 6).

Loads the final encoder of a run, fits the run's linear probe on the probe training split exactly as training
did, then occludes every audited concept unit in the evaluation images and records how the probe's average,
worst-group and per-group accuracy change, together with a size-matched control
(``cospro.curves.attribution``). The concept units are those of the run's own audit (``concept_curves.json``),
so unit ids match the training-time arms ``curves_unit_<k>``. The evaluation concept maps come from
``cospro.cli.build_concept_maps --like <training maps> --split val``; the first run that misses them builds them.

The run must have kept its final checkpoint (``--keep_checkpoints``), as the arm ``csa_audit`` does. Labels
enter only the group metrics, as for every probe of this project.

    python -m cospro.cli.attribute_concepts --study factors_metashift --arm csa_audit --seed 1

Output: ``concept_attribution_<split>.json`` next to the run's ``run.json``. Summarized by
``cospro.cli.summarize_concept_curves``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from cospro.cli.build_concept_maps import main as build_concept_maps
from cospro.cli.build_concept_maps import split_maps_path
from cospro.curves.attribution import cosine_shift, occlusion_features
from cospro.curves.core import hard_regions
from cospro.data.base import dataset_transforms
from cospro.data.registry import canonical_dataset_name, dataset_class
from cospro.evaluation.logistic_probe import fit_logistic_probe
from cospro.evaluation.probe import ProbeOptions, evaluate_probe, load_probe_encoder, seed_probe
from cospro.evaluation.protocol import resolve_evaluation_split
from cospro.metrics import compute_group_metrics
from cospro.pipeline.concept_factors import rows_for_subset
from cospro.tracking.artifacts import PROJECT_ROOT, atomic_write_json, resolve_output_root, scratch_root

SCHEMA = "concept-attribution-v1"


def resolve_uri(value: str) -> Path:
    """A path from a run record: ``artifact://`` lies under scratch, ``project://`` in the repository."""

    value = str(value)
    if value.startswith("artifact://"):
        return scratch_root() / value.removeprefix("artifact://")
    if value.startswith("project://"):
        return PROJECT_ROOT / value.removeprefix("project://")
    if value.startswith("external://"):
        return Path("/" + value.removeprefix("external://"))
    return Path(value)


def newest_complete_run(study_root: Path, arm: str, seed: int) -> Path:
    """The ``run.json`` of the newest complete attempt of ``arm`` and ``seed``."""

    candidates = sorted((study_root / f"seed_{seed:02d}" / arm).glob("*/run.json"))
    complete = [path for path in candidates
                if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"]
    if not complete:
        raise SystemExit(f"No complete run of {arm} seed {seed} under {study_root}.")
    return complete[-1]


def final_checkpoint(record: dict) -> Path:
    for artifact in record.get("artifacts", []):
        if artifact.get("kind") == "ssl_checkpoint" and artifact.get("retention_state") == "final":
            path = resolve_uri(artifact.get("storage_path") or artifact["uri"])
            if path.is_file():
                return path
            raise SystemExit(f"The final checkpoint {path} of this run is missing.")
    raise SystemExit("This run kept no final checkpoint; train it with --keep_checkpoints (arm csa_audit).")


class EvaluationRows(Dataset):
    """The evaluation images with their position: (image, row)."""

    def __init__(self, subset) -> None:
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, row: int):
        return self.subset[row][0], row


def group_summary(predictions: torch.Tensor, labels: torch.Tensor, metadata: torch.Tensor, groups: int) -> dict:
    metrics = compute_group_metrics(predictions, labels, metadata, n_groups=groups)
    return {"wga": metrics.worst_group * 100, "acc": metrics.average * 100,
            "group_acc": [float(value) * 100 for value in metrics.group_accuracy]}


def change(after: dict, before: dict) -> dict:
    return {"wga": after["wga"] - before["wga"], "acc": after["acc"] - before["acc"],
            "group_acc": [a - b for a, b in zip(after["group_acc"], before["group_acc"])]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-record", type=Path, default=None, help="Default: the newest complete attempt.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--eval_split", choices=("val", "test"), default=None)
    parser.add_argument("--final_test", action="store_true", help="Needed for --eval_split test.")
    parser.add_argument("--eval-maps", type=Path, default=None,
                        help="Concept maps of the evaluation images; default: beside the run's training maps.")
    parser.add_argument("--units", type=Path, default=None,
                        help="concept_curves.json whose units to occlude; default: the run's own audit.")
    parser.add_argument("--data_folder", default=os.environ.get("DATA_FOLDER", ""))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        args.eval_split = resolve_evaluation_split(args.eval_split, args.final_test)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    study_root = resolve_output_root(args.output_root) / "seeds" / args.study
    run_path = args.run_record or newest_complete_run(study_root, args.arm, args.seed)
    record = json.loads(run_path.read_text(encoding="utf-8"))
    config = record["config"]
    dataset_name = canonical_dataset_name(config["dataset"])
    adapter_class = dataset_class(dataset_name)
    units_path = args.units or next(run_path.parent.glob("**/concept_curves.json"), None)
    if units_path is None:
        raise SystemExit("No concept_curves.json in this run; pass --units from a concept_curves run of the study.")
    audit = json.loads(Path(units_path).read_text(encoding="utf-8"))
    checkpoint = final_checkpoint(record)
    data_folder = args.data_folder or str(resolve_uri(config["data_folder"]))
    training_maps = resolve_uri(audit["concept_maps"])
    eval_maps = args.eval_maps or split_maps_path(training_maps, args.eval_split)
    if not eval_maps.is_file():
        if args.eval_maps is not None:
            raise SystemExit(f"Missing evaluation concept maps {eval_maps}.")
        # One CLIP pass over the evaluation images, with every setting of the training maps.
        print(f"[INFO] Building the {args.eval_split} concept maps like {training_maps}", flush=True)
        build_concept_maps(["--dataset", dataset_name, "--like", str(training_maps), "--split", args.eval_split,
                            "--data-folder", data_folder, "--device", args.device,
                            "--num-workers", str(args.num_workers)])
    device = torch.device(args.device)
    start = time.time()

    # The probe of the run: same split, solver, seed and RNG order as training (evaluate_probe).
    if config.get("linear_probe_solver", "logistic") != "logistic":
        raise SystemExit("Concept attribution needs the logistic probe.")
    seed = int(config["seed"])
    options = ProbeOptions(
        data_folder=data_folder, train_split=config.get("train_set_linear_layer", "ds_train"),
        eval_split=args.eval_split, batch_size=args.batch_size, num_workers=args.num_workers, seed=seed,
        device=str(device), head=config.get("head", "mlp"), solver="logistic",
        l2=float(config.get("linear_probe_l2", 1e-3)), tolerance=float(config.get("linear_probe_tolerance", 1e-6)),
        max_epochs=int(config.get("linear_probe_max_epochs", 200)),
    )
    seed_probe(seed)
    encoder, feature_dim = load_probe_encoder(config["model"], str(checkpoint), device)
    reference = evaluate_probe(encoder, adapter_class, options, feature_dim=feature_dim)
    train_x, train_y, _ = reference.train_features.tensors
    _, labels, metadata = reference.eval_features.tensors

    # The evaluation images in the probe's order, their concept maps and the hard regions of the audit's units.
    dataset_config = adapter_class.Config(root_dir=data_folder, train_split=options.train_split,
                                          eval_split=args.eval_split)
    eval_transform = dataset_transforms(dataset_config.image_size, adapter_class.mean, adapter_class.std)[2]
    subset = adapter_class.from_config(dataset_config).get_subset(args.eval_split, transform=eval_transform)
    stored = torch.load(eval_maps, map_location="cpu", weights_only=False)
    unit_of_factor = torch.tensor(audit["unit_of_factor"], dtype=torch.long)
    if stored["maps"].shape[1] != len(unit_of_factor):
        raise SystemExit(f"{eval_maps} has {stored['maps'].shape[1]} factors; the audit has {len(unit_of_factor)}.")
    rows = rows_for_subset([str(sample_id) for sample_id in stored["sample_ids"]], dataset_name, subset.indices)
    regions = hard_regions(stored["maps"].index_select(0, rows), unit_of_factor)
    del stored
    audited = [unit["unit"] for unit in audit["units"] if unit["audited"]]
    loader = DataLoader(EvaluationRows(subset), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    features = occlusion_features(encoder, loader, regions, audited, device)
    clean = features["clean"]

    # One fit, every variant classified by the same probe: the clean images, each occlusion and each control.
    variants = [clean, *(features["occluded"][unit] for unit in audited), *(features["control"][unit] for unit in audited)]
    records, convergence = fit_logistic_probe(
        train_x, train_y, torch.cat(variants), labels.repeat(len(variants)), num_classes=adapter_class.num_classes,
        l2=options.l2, tolerance=options.tolerance, max_epochs=options.max_epochs,
    )
    predictions = records[-1].eval_predictions.view(len(variants), -1)
    groups = int(compute_group_metrics(predictions[0], labels, metadata).group_counts.numel())
    baseline = group_summary(predictions[0], labels, metadata, groups)
    typing = {entry["unit"]: entry["type"] for entry in audit.get("evaluation_only", {}).get("typing", [])}
    present = features["present"]
    results = []
    for position, unit in enumerate(audited):
        occluded = group_summary(predictions[1 + position], labels, metadata, groups)
        control = group_summary(predictions[1 + len(audited) + position], labels, metadata, groups)
        shown = present[:, unit]
        results.append({
            "unit": unit, "name": audit["units"][unit]["name"], "type": typing.get(unit, "n/a"),
            "eval_presence": float(shown.float().mean()), "images": int(shown.sum()),
            "embedding_shift": cosine_shift(clean, features["occluded"][unit], shown),
            "control_shift": cosine_shift(clean, features["control"][unit], shown),
            "occluded": occluded, "control": control,
            "change": change(occluded, baseline), "control_change": change(control, baseline),
            "net_change": change(occluded, control),
        })
    group_names = [f"(target,context)=({group // 2},{group % 2})" for group in range(groups)] if groups == 4 else None
    output = args.output or run_path.parent / f"concept_attribution_{args.eval_split}.json"
    atomic_write_json(output, {
        "schema": SCHEMA, "study": args.study, "arm": args.arm, "seed": seed, "attempt": run_path.parent.name,
        "split": args.eval_split, "checkpoint": str(checkpoint), "eval_maps": str(eval_maps), "units_source": str(units_path),
        "probe": {
            # The probe names its evaluation metrics "val" for either split.
            "reference_wga": reference.metrics.get("Last linear val worst-group acc"),
            "run_last_wga": record.get("final_metrics", {}).get("Last linear val worst-group acc"),
            "clean_feature_max_difference": float((clean - reference.eval_features.tensors[0]).abs().max()),
            "convergence": convergence,
        },
        "group_counts": [int(value) for value in compute_group_metrics(predictions[0], labels, metadata, groups).group_counts],
        "group_names": group_names, "clean": baseline, "units": results, "seconds": round(time.time() - start, 1),
    })
    print(f"[results] Concept attribution: {output}")
    return output


if __name__ == "__main__":
    main()
