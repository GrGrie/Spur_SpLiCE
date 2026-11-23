"""Patch-concept readout of a frozen I-JEPA encoder on Waterbirds.

Which CLIP/SpLiCE concepts do I-JEPA's patch tokens carry, where, and which of them decide a
group-balanced linear probe? See ``cospro.evaluation.patch_concept_readout`` for the three measurements.

    python -m cospro.cli.ijepa_concept_readout --model ijepa_vith14_in1k --data_folder ~/Datasets \\
        --concept_maps <scratch>/features/Spur_SpLiCE/waterbirds/concept_maps/<name>/concept_maps.pt

Every image is read with the evaluation transform (224-pixel resize), so the probes here have no crop
noise and their numbers can differ from ``cospro.cli.linear_probe``, which crops the probe's training
images at random. Outputs: ``outputs/seeds/<study>/seed_<NN>/<model>/<attempt>/readout.json`` and the
fitted ridge under ``<scratch>/features/Spur_SpLiCE/<study>/``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cospro.data.base import dataset_transforms
from cospro.data.registry import dataset_class
from cospro.evaluation.logistic_probe import fit_logistic_probe
from cospro.evaluation.patch_concept_readout import (
    concept_decision_shares, cross_fit_readout, image_folds, margin_direction, mean_pool, pooled_control,
    predict_concepts, resize_maps, shuffled_control,
)
from cospro.evaluation.protocol import resolve_evaluation_split
from cospro.metrics import compute_group_metrics
from cospro.models.ijepa import IJEPA_MODEL_NAMES, build_ijepa_encoder
from cospro.tracking.artifacts import atomic_write_json, run_directory, scratch_root

DATASET = "waterbirds"
GROUP_NAMES = ("landbird on land", "landbird on water", "waterbird on land", "waterbird on water")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ijepa_vith14_in1k", choices=IJEPA_MODEL_NAMES)
    parser.add_argument("--ckpt", default="", help="Official I-JEPA checkpoint; empty downloads it to scratch.")
    parser.add_argument("--data_folder", default=os.environ.get("DATA_FOLDER", ""))
    parser.add_argument("--concept_maps", required=True, type=Path,
                        help="concept_maps.pt of the training images (cospro.cli.build_concept_maps).")
    parser.add_argument("--final_test", action="store_true", help="Evaluate on test instead of val.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridges", default="1,10,100,1000,10000", help="Comma-separated ridge strengths to compare.")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 of the logistic probes.")
    parser.add_argument("--top", type=int, default=15, help="Concepts listed per ranking.")
    parser.add_argument("--skip_controls", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", default="ijepa_concept_readout")
    parser.add_argument("--attempt_id", default="local")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


@torch.no_grad()
def extract_tokens(encoder, subset, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Float16 patch tokens [images, patches, dim], labels and metadata of a subset, in subset order."""

    loader = DataLoader(subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    tokens, labels, metadata = [], [], []
    start = time.time()
    for images, batch_labels, batch_metadata in loader:
        images = images.to(args.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=images.is_cuda):
            tokens.append(encoder.patch_tokens(images).half().cpu())
        labels.append(batch_labels)
        metadata.append(batch_metadata)
        if len(tokens) % 20 == 0:
            print(f"[INFO] {sum(len(t) for t in tokens)}/{len(subset)} images, {time.time() - start:.0f} s", flush=True)
    return torch.cat(tokens), torch.cat(labels), torch.cat(metadata)


def aligned_maps(payload: dict, indices) -> torch.Tensor:
    """The maps of ``payload`` reordered to the dataset ``indices``."""

    position = {sample_id: row for row, sample_id in enumerate(payload["sample_ids"])}
    missing = [index for index in indices if f"{DATASET}:{int(index)}" not in position]
    if missing:
        raise ValueError(f"{len(missing)} training images have no concept map, e.g. {DATASET}:{int(missing[0])}.")
    return payload["maps"][[position[f"{DATASET}:{int(index)}"] for index in indices]]


def probe(train_x, train_y, eval_x, eval_y, eval_metadata, l2: float) -> tuple[dict, dict]:
    """Group metrics of a converged logistic probe and its fitted parameters."""

    parameters: dict = {}
    records, convergence = fit_logistic_probe(train_x, train_y, eval_x, eval_y, num_classes=2, l2=l2,
                                              parameters=parameters)
    metrics = compute_group_metrics(records[-1].eval_predictions, eval_y, eval_metadata, n_groups=4)
    return {
        "accuracy": round(100 * metrics.average, 2),
        "worst_group_accuracy": round(100 * metrics.worst_group, 2),
        "group_accuracy": {name: round(100 * float(value), 2)
                           for name, value in zip(GROUP_NAMES, metrics.group_accuracy)},
        "converged": convergence["converged"],
    }, parameters


def ranked(names: list[str], values: torch.Tensor, top: int) -> dict[str, list]:
    order = values.argsort(descending=True)
    entry = lambda index: {"concept": names[index], "value": round(float(values[index]), 5)}
    return {"highest": [entry(int(i)) for i in order[:top]], "lowest": [entry(int(i)) for i in order.flip(0)[:top]]}


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    ridges = [float(value) for value in args.ridges.split(",")]
    eval_split = resolve_evaluation_split(None, args.final_test)
    output_dir = run_directory(args.seed, args.study, args.model, args.attempt_id)
    print(f"[results] Readout JSON: {output_dir / 'readout.json'}", flush=True)

    adapter_class = dataset_class(DATASET)
    dataset = adapter_class(str(args.data_folder))
    eval_transform = dataset_transforms(adapter_class.image_size, adapter_class.mean, adapter_class.std)[2]
    train = dataset.get_subset("train", transform=eval_transform)
    evaluation = dataset.get_subset(eval_split, transform=eval_transform)
    position = {int(index): row for row, index in enumerate(train.indices)}
    balanced = torch.tensor([position[int(index)] for index in dataset.get_subset("ds_train").indices])

    payload = torch.load(args.concept_maps, map_location="cpu", weights_only=True)
    names = [str(name) for name in payload["factor_names"]]
    maps = aligned_maps(payload, train.indices)
    del payload

    encoder, _ = build_ijepa_encoder(args.model, args.ckpt)
    encoder = encoder.to(args.device).eval()
    print(f"[INFO] Extracting {len(train)} training images", flush=True)
    tokens, train_y, train_metadata = extract_tokens(encoder, train, args)
    print(f"[INFO] Extracting {len(evaluation)} {eval_split} images", flush=True)
    eval_tokens, eval_y, eval_metadata = extract_tokens(encoder, evaluation, args)
    del encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    side = int(round(tokens.shape[1] ** 0.5))
    targets = resize_maps(maps, side).half()
    folds = image_folds(len(tokens), args.folds, args.seed)

    print("[INFO] Cross-fitting the patch-concept ridge", flush=True)
    readout = cross_fit_readout(tokens, targets, folds, ridges, device=args.device)
    best = readout["best_ridge"]
    controls = {}
    if not args.skip_controls:
        controls = {
            "image_level_only": round(pooled_control(tokens, targets, folds, best, args.device), 4),
            "shuffled_maps": round(shuffled_control(tokens, targets, folds, best, args.seed, args.device), 4),
        }
    variance = targets.float().var(dim=(0, 1))
    readable = readout["per_factor"].clone()
    readable[variance < 1e-4] = float("nan")  # factors that never appear tell nothing

    # Probe 1: mean-pooled tokens, the standard I-JEPA linear evaluation without crop noise.
    pooled_train, pooled_eval = mean_pool(tokens), mean_pool(eval_tokens)
    mean_metrics, mean_parameters = probe(pooled_train[balanced], train_y[balanced], pooled_eval, eval_y,
                                          eval_metadata, args.l2)
    # Probe 2: the concept bottleneck. Training images use out-of-fold predictions, so the ridge never
    # saw the CLIP maps of the images it scores.
    eval_concepts_per_patch = torch.cat([
        predict_concepts(eval_tokens[start:start + 256].to(args.device), readout["full_fit"]).cpu()
        for start in range(0, len(eval_tokens), 256)
    ])
    eval_concepts = eval_concepts_per_patch.mean(dim=1)
    bottleneck_metrics, bottleneck_parameters = probe(readout["oof_concept_vectors"][balanced], train_y[balanced],
                                                      eval_concepts, eval_y, eval_metadata, args.l2)
    concept_weight = ((bottleneck_parameters["weight"][1] - bottleneck_parameters["weight"][0])
                      / bottleneck_parameters["scale"]).float()

    # Decision shares of the mean-pool probe: positive means evidence for the image's true class.
    direction, offset = margin_direction(mean_parameters["weight"].float(), mean_parameters["scale"].float(),
                                         mean_parameters["mean"].float())
    shares = torch.cat([
        concept_decision_shares(eval_tokens[start:start + 256], eval_concepts_per_patch[start:start + 256],
                                direction, offset)
        for start in range(0, len(eval_tokens), 256)
    ])
    signed = shares * (2.0 * eval_y.float() - 1.0)[:, None]
    groups = eval_metadata[:, 0].long() + 2 * eval_metadata[:, 1].long()
    present = eval_concepts.mean(dim=0) > 1.0 / len(names)

    result = {
        "schema": "ijepa-concept-readout-v1",
        "model": args.model,
        "eval_split": eval_split,
        "concept_maps": str(args.concept_maps),
        "factors": len(names),
        "patch_grid": side,
        "train_images": len(tokens),
        "probe_train_images": len(balanced),
        "readout": {
            "explained_variance": round(readout["explained_variance"], 4),
            "per_ridge": {f"{ridge:g}": round(value, 4) for ridge, value in readout["per_ridge"].items()},
            "best_ridge": best,
            "folds": args.folds,
            "controls": controls,
            "best_read_factors": ranked(names, torch.nan_to_num(readable, nan=-1e9), args.top)["highest"],
            "worst_read_factors": ranked(names, torch.nan_to_num(readable, nan=1e9), args.top)["lowest"],
        },
        "probes": {"mean_pool": mean_metrics, "concept_bottleneck": bottleneck_metrics},
        "bottleneck_weights_towards_waterbird": ranked(names, concept_weight, args.top),
        "decision_shares": {
            "meaning": "mean share of the mean-pool probe margin per concept region, signed towards the "
                       "true class; negative pushes towards the wrong class",
            "all_images": ranked(names, torch.where(present, signed.mean(dim=0), torch.zeros(len(names))), args.top),
            "per_group": {
                name: ranked(names, signed[groups == group].mean(dim=0), args.top)
                for group, name in enumerate(GROUP_NAMES) if bool((groups == group).any())
            },
        },
        "notes": "Ridge chosen by held-out explained variance among the listed strengths on the same folds. "
                 "Concept maps exist for training images only; val and test concepts come from the ridge.",
    }
    path = atomic_write_json(output_dir / "readout.json", result)
    ridge_path = scratch_root() / "features" / "Spur_SpLiCE" / args.study / args.model / args.attempt_id / "ridge.pt"
    ridge_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": readout["full_fit"][0].float(), "intercept": readout["full_fit"][1].float(),
                "factor_names": names, "ridge": best, "concept_maps": str(args.concept_maps)}, ridge_path)
    print(f"[results] Readout: {path}")
    print(f"[results] Ridge: {ridge_path}")
    print(f"[results] EV {result['readout']['explained_variance']} controls {controls}; "
          f"mean-pool WGA {mean_metrics['worst_group_accuracy']}, "
          f"bottleneck WGA {bottleneck_metrics['worst_group_accuracy']}")
    return path


if __name__ == "__main__":
    main()
