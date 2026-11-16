"""Train one supervised run of the concept-curves project (docs/PROJECT_PLAN.md).

``--mode erm`` trains with plain cross-entropy and records the concept learning curves: after every
early epoch the model's reliance on every concept unit is measured by occluding the unit's regions.
``--mode swap`` reads the curves of an ERM run, flags the units that led them and trains from fresh
weights while the flagged regions are exchanged between images.

    python -m cospro.cli.train_concept_curves --dataset metashift --seed 1 --mode erm \\
        --concept-maps <scratch>/features/Spur_SpLiCE/metashift/concept_maps/<name>/concept_maps.pt \\
        --study curves_metashift --arm erm
    python -m cospro.cli.train_concept_curves --dataset metashift --seed 1 --mode swap \\
        --concept-maps <same file> --study curves_metashift --arm swap --curves-arm erm --flag-rule top

Outputs under ``outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/``: ``run.json`` (configuration,
evaluations with worst-group accuracy, final numbers), ``concept_curves.json`` (the audit) and, for a
swap run, ``flagged_units.json``. No checkpoint is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cospro.config.settings import wandb_entity
from cospro.curves.core import (
    FLAG_RULES,
    build_units,
    concept_shares,
    describe_units,
    detection_auroc,
    flag_units,
    flagged_area,
    hard_regions,
    load_unit_maps,
    onset_scores,
    reliance_audit,
    unit_presence,
    unit_typing,
)
from cospro.curves.swap import flagged_masks, swap_flagged
from cospro.curves.training import (
    FALLBACK_DEFAULTS,
    TRAINING_DEFAULTS,
    AuditBatches,
    Classifier,
    PlainImages,
    TrainViews,
    evaluate,
    train_epoch,
)
from cospro.data.registry import canonical_dataset_name, dataset_class, dataset_names
from cospro.models.resnet import RESNET_MODEL_NAMES
from cospro.tracking.artifacts import atomic_write_json, make_attempt_id, run_directory, seed_run

WGA_KEY = "curves/{split}/wga"
#: The reported number of a run averages this many last evaluations.
FINAL_EVALUATIONS = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=canonical_dataset_name, choices=dataset_names())
    parser.add_argument("--data-folder", default=os.environ.get("DATA_FOLDER", "./datasets"))
    parser.add_argument("--concept-maps", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--mode", choices=("erm", "swap"), default="erm")
    parser.add_argument("--model", choices=RESNET_MODEL_NAMES, default="resnet18_large")
    parser.add_argument("--epochs", type=int, default=None, help="Default per model, see the plan.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--unit-count", type=int, default=12, help="Concept units; 0 uses single factors.")
    parser.add_argument("--early-epochs", type=int, default=None, help="T0: the audit runs after each of these.")
    parser.add_argument("--late-audit-every", type=int, default=10, help="Audit interval after the early window.")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--final-test", action="store_true", help="One evaluation on the test split after training.")
    # Swap options.
    parser.add_argument("--curves", type=Path, default=None, help="concept_curves.json of the ERM run to flag from.")
    parser.add_argument("--curves-arm", default="erm", help="Arm of this study and seed whose curves are read "
                        "when --curves is not given.")
    parser.add_argument("--flag-rule", choices=FLAG_RULES, default="top")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--max-flagged-area", type=float, default=0.75)
    parser.add_argument("--flag-units", default="", help="Comma-separated unit ids for --flag-rule units.")
    parser.add_argument("--flag-seed", type=int, default=0)
    parser.add_argument("--swap-probability", type=float, default=0.5)
    parser.add_argument("--shuffle-maps", action="store_true", help="Control: every image carries another image's regions.")
    # Run identity and runtime.
    parser.add_argument("--study", default="curves_adhoc")
    parser.add_argument("--arm", default="")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="CoSpRo")
    parser.add_argument("--max-train-images", type=int, default=0, help="Smoke tests: train on the first images only.")
    args = parser.parse_args(argv)
    defaults = TRAINING_DEFAULTS.get(args.model, FALLBACK_DEFAULTS)
    for name in ("epochs", "learning_rate", "weight_decay", "early_epochs"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    args.early_epochs = min(args.early_epochs, args.epochs)
    args.arm = args.arm or args.mode
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_curves(study: str, seed: int, arm: str, early_epochs: int, root=None) -> Path:
    """The newest ``concept_curves.json`` of an arm of this study and seed that covers the early window."""

    directory = seed_run(seed, study, arm, root=root)
    for candidate in sorted(directory.glob("*/concept_curves.json"), reverse=True):
        epochs = json.loads(candidate.read_text(encoding="utf-8")).get("epochs", [])
        if epochs and max(epochs) >= early_epochs:
            return candidate
    raise FileNotFoundError(f"No concept_curves.json with {early_epochs} early epochs under {directory}. "
                            "Run the ERM arm of this seed first.")


def audit_epochs(args: argparse.Namespace) -> set[int]:
    late = range(args.early_epochs, args.epochs + 1, max(args.late_audit_every, 1))
    return {0, args.epochs, *range(1, args.early_epochs + 1), *late}


def evaluation_epochs(args: argparse.Namespace) -> list[int]:
    return sorted({args.epochs, *range(args.eval_every, args.epochs + 1, max(args.eval_every, 1))})


def final_numbers(evaluations: list[dict], split: str) -> dict:
    """Mean of the last ``FINAL_EVALUATIONS`` evaluations: WGA, accuracy and group accuracies."""

    last = [entry[split] for entry in evaluations[-FINAL_EVALUATIONS:]]
    groups = len(last[0]["group_acc"])
    return {
        f"{split}_wga": sum(entry["wga"] for entry in last) / len(last),
        f"{split}_acc": sum(entry["acc"] for entry in last) / len(last),
        f"{split}_group_acc": [sum(entry["group_acc"][group] for entry in last) / len(last) for group in range(groups)],
        "evaluations_averaged": len(last),
    }


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = run_directory(args.seed, args.study, args.arm, make_attempt_id(), root=args.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[results] Run record: {run_dir / 'run.json'}", flush=True)

    adapter_class = dataset_class(args.dataset)
    adapter = adapter_class.from_config(adapter_class.Config(root_dir=args.data_folder))
    size, mean, std = adapter_class.image_size, adapter_class.mean, adapter_class.std
    train_subset = adapter.get_subset("train")
    if args.max_train_images:
        train_subset.indices = train_subset.indices[:args.max_train_images]
    train_views = TrainViews(train_subset, size, mean, std)
    train_plain = PlainImages(train_subset, size, mean, std)
    workers = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda",
               "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train_views, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              generator=torch.Generator().manual_seed(args.seed), **workers)
    audit_loader = AuditBatches(DataLoader(train_plain, batch_size=args.batch_size, shuffle=False, **workers))
    eval_loader = DataLoader(PlainImages(adapter.get_subset(args.eval_split), size, mean, std),
                             batch_size=args.batch_size, shuffle=False, **workers)

    # Concept units and regions. A swap run takes the unit assignment of the ERM run it flags from.
    maps, factor_names = load_unit_maps(str(args.concept_maps), args.dataset, train_views.indices)
    source_curves = None
    if args.mode == "swap":
        curves_path = args.curves or find_curves(args.study, args.seed, args.curves_arm, args.early_epochs,
                                                 root=args.output_root)
        source_curves = json.loads(Path(curves_path).read_text(encoding="utf-8"))
        unit_of_factor = torch.tensor(source_curves["unit_of_factor"], dtype=torch.long)
        if len(unit_of_factor) != maps.shape[1]:
            raise ValueError(f"{curves_path} was recorded for other concept maps than {args.concept_maps}.")
    else:
        unit_of_factor = build_units(maps, args.unit_count, seed=0)
    regions = hard_regions(maps, unit_of_factor)
    units = describe_units(maps, unit_of_factor, regions, factor_names)
    del maps
    audited = [entry["unit"] for entry in units if entry["audited"]]
    # Hidden labels score the audit; training reads them only under the oracle rule, an upper-bound control.
    typing = unit_typing(unit_presence(regions), train_plain.labels,
                         train_plain.metadata[:, adapter_class.spurious_metadata_index].long())

    transform_batch = None
    flags = None
    if args.mode == "swap":
        if args.flag_rule == "oracle":
            flagged = [entry["unit"] for entry in typing if entry["type"] == "attribute" and units[entry["unit"]]["audited"]]
            if not flagged:
                raise ValueError("The oracle rule found no attribute unit.")
        else:
            explicit = [int(unit) for unit in args.flag_units.split(",") if unit.strip()]
            flagged = flag_units(source_curves["onset"], source_curves["units"], rule=args.flag_rule, tau=args.tau,
                                 max_area=args.max_flagged_area, seed=args.flag_seed, explicit=explicit)
        flags = {"rule": args.flag_rule, "tau": args.tau, "flagged": flagged,
                 "names": [units[unit]["name"] for unit in flagged],
                 "types": [typing[unit]["type"] for unit in flagged],
                 "source_curves": str(curves_path), "mean_flagged_area": round(flagged_area(flagged, units), 4),
                 "shuffled_maps": args.shuffle_maps}
        atomic_write_json(run_dir / "flagged_units.json", flags)
        print(f"[INFO] Flagged units ({args.flag_rule}): {flags['names']}", flush=True)
        swap_regions = regions
        if args.shuffle_maps:
            swap_regions = regions[torch.randperm(len(regions), generator=torch.Generator().manual_seed(0))]
        swap_generator = torch.Generator().manual_seed(args.seed + 1)

        def transform_batch(views, rows, boxes):
            batch_regions = swap_regions[torch.as_tensor(rows, dtype=torch.long)].to(views.device)
            masks = flagged_masks(batch_regions, flagged, boxes, tuple(views.shape[-2:]))
            return swap_flagged(views, masks, args.swap_probability, generator=swap_generator)

    model = Classifier(args.model, adapter_class.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    record = {"config": config, "status": "running", "train_images": len(train_views), "evaluations": [],
              "training": [], "flags": flags}
    curves = {"dataset": args.dataset, "seed": args.seed, "model": args.model, "mode": args.mode,
              "concept_maps": str(args.concept_maps), "unit_count": int(unit_of_factor.max()) + 1,
              "early_epochs": args.early_epochs, "unit_of_factor": unit_of_factor.tolist(), "units": units,
              "epochs": [], "reliance": [], "squared": [], "share": [], "onset": None,
              "evaluation_only": {"typing": typing}}
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, entity=wandb_entity(), group=args.study,
                               name=f"{args.dataset}_s{args.seed}_{args.arm}", config=config)
        wandb_run.define_metric(WGA_KEY.format(split=args.eval_split), summary="last")

    def audit(epoch: int) -> None:
        result = reliance_audit(model, audit_loader, regions, audited, device, amp=args.amp)
        curves["epochs"].append(epoch)
        curves["reliance"].append(result["reliance"])
        curves["squared"].append(result["squared"])
        curves["share"].append(concept_shares(result["reliance"], audited))
        if epoch >= 1:
            curves["onset"] = onset_scores(curves["epochs"], curves["share"], args.early_epochs)
        if epoch >= args.early_epochs:
            try:
                flagged_top = flag_units(curves["onset"], units, rule="top", tau=args.tau,
                                         max_area=args.max_flagged_area)
            except ValueError:  # no positive reliance in the early window
                flagged_top = []
            score = detection_auroc(curves["onset"], units, typing)
            curves["evaluation_only"].update({
                "detection_auroc": None if math.isnan(score) else score,
                "top_flags": flagged_top,
                "top_flag_types": [typing[unit]["type"] for unit in flagged_top],
            })
        atomic_write_json(run_dir / "concept_curves.json", curves)

    audits, evaluations = audit_epochs(args), evaluation_epochs(args)
    audit(0)
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            metrics = train_epoch(model, train_loader, optimizer, scaler, device, args.amp, transform_batch)
            schedule.step()
            record["training"].append({"epoch": epoch, **{key: round(value, 5) for key, value in metrics.items()}})
            logged = {f"curves/train/{key}": value for key, value in metrics.items()}
            if epoch in audits:
                audit(epoch)
            if epoch in evaluations:
                result = evaluate(model, eval_loader, device, args.amp)
                record["evaluations"].append({"epoch": epoch, args.eval_split: result})
                logged.update({WGA_KEY.format(split=args.eval_split): result["wga"],
                               f"curves/{args.eval_split}/acc": result["acc"]})
                print(f"[INFO] epoch {epoch}: loss {metrics['loss']:.3f} {args.eval_split} WGA {result['wga']:.3f} "
                      f"acc {result['acc']:.3f} groups {result['group_acc']} "
                      f"swapped {metrics['swapped_fraction']:.2f} ({time.time() - started:.0f} s)", flush=True)
                atomic_write_json(run_dir / "run.json", record)
            if wandb_run is not None:
                wandb_run.log(logged, step=epoch)
        record["final"] = final_numbers(record["evaluations"], args.eval_split)
        if args.final_test:
            test_loader = DataLoader(PlainImages(adapter.get_subset("test"), size, mean, std),
                                     batch_size=args.batch_size, shuffle=False, **workers)
            record["final"]["test"] = evaluate(model, test_loader, device, args.amp)
        record["status"] = "complete"
        if wandb_run is not None:
            wandb_run.summary[WGA_KEY.format(split=args.eval_split)] = record["final"][f"{args.eval_split}_wga"]
    except Exception as error:
        record["status"] = "failed"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        atomic_write_json(run_dir / "run.json", record)
        if wandb_run is not None:
            wandb_run.finish()
    print(f"[results] Final: {json.dumps(record['final'])}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
