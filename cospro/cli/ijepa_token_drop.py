"""Concept shortcut attribution at inference for a frozen I-JEPA encoder on Waterbirds.

For each concept unit: remove the patches the unit wins in every evaluation image from the encoder's input,
encode again and classify with fixed linear probes. The verdict comes from the shortcut index: how much the
removal helps the conflict groups (landbird on water, waterbird on land) over the aligned groups, minus the
same index for removing as many patches elsewhere (the shuffled control). A positive net index marks a
shortcut the probe relied on, a negative one a concept it needed. The WGA change is reported beside it; it
hides the shortcut whenever the worst group loses a little while the other conflict group gains. See
``cospro.evaluation.token_drop_attribution``.

Unit granularity: 12 units match the CSA arms; ``--units 24`` or ``--units 0`` (single factors) separate
scene concepts such as water from the birds they appear with.

    python -m cospro.cli.ijepa_token_drop --model ijepa_vith14_in1k --data_folder ~/Datasets \\
        --concept_maps <scratch>/features/Spur_SpLiCE/waterbirds/concept_maps/<name>/concept_maps.pt --final_test

Where the regions come from. Units are built from the CLIP/SpLiCE maps of the training images, as in CSA.
Those maps exist for training images only, so by default a patch-level ridge, fitted on the training
images, reads every evaluation patch's concept mass from the I-JEPA tokens; the JSON reports how well the
ridge's hard regions match the CLIP ones on held-out training images. ``--eval_concept_maps`` replaces the
ridge with CLIP maps of the evaluation images when they exist.

Labels: the units, the regions, the ridge and the reliance statistic read none. Class labels train the
probes; group labels enter the WGA effects and the evaluation-only unit typing, as for every WGA of the
study. Output: ``outputs/seeds/<study>/seed_<NN>/<model>/<attempt>/token_drop.json``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cospro.cli.ijepa_concept_readout import GROUP_NAMES, aligned_maps, extract_tokens
from cospro.data.base import dataset_transforms
from cospro.data.registry import dataset_class
from cospro.evaluation.logistic_probe import fit_logistic_probe
from cospro.evaluation.patch_concept_readout import (
    cross_fit_readout, fold_moments, image_folds, mean_pool, predict_concepts, resize_maps,
)
from cospro.evaluation.protocol import resolve_evaluation_split
from cospro.evaluation.token_drop_attribution import (
    MIN_PRESENCE, concept_units, drop_masks, group_accuracy, hard_regions, keep_pooled, paired_effect, presence,
    region_agreement, shortcut_index, shuffled_regions, spearman, unit_maps, unit_typing,
)
from cospro.methods.partial_finetune import logistic_logits
from cospro.models.ijepa import IJEPA_MODEL_NAMES, build_ijepa_encoder
from cospro.tracking.artifacts import atomic_write_json, run_directory

DATASET = "waterbirds"
PROBE_SPLITS = ("ds_train", "cb_train", "train")
#: Groups where class and background disagree: landbird on water and waterbird on land.
CONFLICT_GROUPS = (1, 2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ijepa_vith14_in1k", choices=IJEPA_MODEL_NAMES)
    parser.add_argument("--ckpt", default="", help="Official I-JEPA checkpoint; empty downloads it to scratch.")
    parser.add_argument("--data_folder", default=os.environ.get("DATA_FOLDER", ""))
    parser.add_argument("--concept_maps", required=True, type=Path,
                        help="concept_maps.pt of the training images (cospro.cli.build_concept_maps).")
    parser.add_argument("--eval_concept_maps", type=Path, default=None,
                        help="Optional CLIP concept maps of the evaluation images, with the same factors.")
    parser.add_argument("--final_test", action="store_true", help="Evaluate on test instead of val.")
    parser.add_argument("--units", type=int, default=12, help="Concept units as in CSA; 0 uses single factors.")
    parser.add_argument("--unit_seed", type=int, default=0, help="Seed of the unit clustering; CSA uses 0.")
    parser.add_argument("--probes", default="ds_train,cb_train",
                        help=f"Comma-separated probe training splits among {PROBE_SPLITS}.")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 of the logistic probes.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridges", default="0.01,0.1,1,10,100")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Resamples of the WGA-effect intervals.")
    parser.add_argument("--skip_shuffled", action="store_true", help="Skip the shuffled-region control.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", default="ijepa_token_drop")
    parser.add_argument("--attempt_id", default="local")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    args.probe_splits = [split.strip() for split in args.probes.split(",") if split.strip()]
    if not args.probe_splits or set(args.probe_splits) - set(PROBE_SPLITS):
        parser.error(f"--probes must list splits among {PROBE_SPLITS}.")
    return args


def unit_names(maps: torch.Tensor, unit_of_factor: torch.Tensor, names: list[str], top: int = 3) -> list[list[str]]:
    """The ``top`` factors of every unit by their total map mass over the training images."""

    mass = torch.zeros(maps.shape[1], dtype=torch.float64)
    for start in range(0, len(maps), 512):
        mass += maps[start:start + 512].double().sum(dim=(0, 2, 3))
    result = []
    for unit in range(int(unit_of_factor.max()) + 1):
        members = torch.nonzero(unit_of_factor == unit).flatten()
        order = members[mass[members].argsort(descending=True)][:top]
        result.append([names[int(index)] for index in order])
    return result


def map_regions(maps: torch.Tensor, unit_of_factor: torch.Tensor, side: int | None = None) -> torch.Tensor:
    """Hard regions [images, patches, units] of CLIP maps [images, factors, s, s], resampled to ``side``."""

    if side is not None:
        per_patch = resize_maps(maps, side)
    else:
        per_patch = maps.float().flatten(2).transpose(1, 2)
    return torch.cat([hard_regions(unit_maps(per_patch[start:start + 512], unit_of_factor))
                      for start in range(0, len(per_patch), 512)])


@torch.no_grad()
def encode(encoder, images: torch.Tensor, keep: torch.Tensor | None, args) -> torch.Tensor:
    """Pooled features of cached ``images`` [n, 3, H, W] with the patches outside ``keep`` removed."""

    pooled = []
    for start in range(0, len(images), args.batch_size):
        batch = images[start:start + args.batch_size].to(args.device).float()
        mask = None if keep is None else keep[start:start + args.batch_size].to(args.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=batch.is_cuda):
            tokens = encoder.patch_tokens(batch, mask)
        pooled.append(keep_pooled(tokens, mask).cpu())
    return torch.cat(pooled)


@torch.no_grad()
def cache_evaluation(encoder, subset, readout_fit, unit_of_factor, args) -> dict:
    """Images (float16), labels, metadata, intact pooled features and ridge unit mass of the evaluation split."""

    loader = DataLoader(subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    images, labels, metadata, pooled, mass = [], [], [], [], []
    for batch, batch_labels, batch_metadata in loader:
        batch = batch.to(args.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=batch.is_cuda):
            tokens = encoder.patch_tokens(batch)
        pooled.append(tokens.float().mean(dim=1).cpu())
        mass.append(unit_maps(predict_concepts(tokens, readout_fit), unit_of_factor).cpu())
        images.append(batch.half().cpu())
        labels.append(batch_labels)
        metadata.append(batch_metadata)
    return {"images": torch.cat(images), "labels": torch.cat(labels), "metadata": torch.cat(metadata),
            "pooled": torch.cat(pooled), "unit_mass": torch.cat(mass)}


def fit_probe(features: torch.Tensor, labels: torch.Tensor, l2: float) -> dict:
    parameters: dict = {}
    fit_logistic_probe(features, labels, features[:2], labels[:2], num_classes=2, l2=l2, parameters=parameters)
    return parameters


def heldout_agreement(tokens, targets, folds, ridge, regions_reference, unit_of_factor, device) -> dict:
    """Ridge hard regions on fold 0 against the CLIP hard regions, with the ridge fitted on the other folds."""

    split = (folds != 0).long()
    moments = fold_moments(tokens, targets, split, 2, device=device)
    fit = tuple(part.cpu() for part in moments[1].solve(ridge))
    rows = torch.nonzero(folds == 0).flatten()
    predicted = torch.cat([
        hard_regions(unit_maps(predict_concepts(tokens[rows[start:start + 128]].to(device), fit), unit_of_factor).cpu())
        for start in range(0, len(rows), 128)
    ])
    return {"images": len(rows), **region_agreement(predicted, regions_reference[rows])}


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    started = time.time()
    ridges = [float(value) for value in args.ridges.split(",")]
    eval_split = resolve_evaluation_split(None, args.final_test)
    output_dir = run_directory(args.seed, args.study, args.model, args.attempt_id)
    print(f"[results] Attribution JSON: {output_dir / 'token_drop.json'}", flush=True)

    adapter_class = dataset_class(DATASET)
    dataset = adapter_class(str(args.data_folder))
    eval_transform = dataset_transforms(adapter_class.image_size, adapter_class.mean, adapter_class.std)[2]
    train = dataset.get_subset("train", transform=eval_transform)
    evaluation = dataset.get_subset(eval_split, transform=eval_transform)
    position = {int(index): row for row, index in enumerate(train.indices)}

    payload = torch.load(args.concept_maps, map_location="cpu", weights_only=True)
    names = [str(name) for name in payload["factor_names"]]
    maps = aligned_maps(payload, train.indices)
    del payload
    unit_of_factor = concept_units(maps, args.units, args.unit_seed)
    units = int(unit_of_factor.max()) + 1
    labels_of_units = unit_names(maps, unit_of_factor, names)
    print(f"[INFO] {len(names)} factors in {units} units", flush=True)

    encoder, _ = build_ijepa_encoder(args.model, args.ckpt)
    encoder = encoder.to(args.device).eval()
    print(f"[INFO] Extracting {len(train)} training images", flush=True)
    tokens, train_y, train_metadata = extract_tokens(encoder, train, args)
    side = int(round(tokens.shape[1] ** 0.5))
    targets = resize_maps(maps, side).half()
    train_regions = map_regions(maps, unit_of_factor, side)
    typing = unit_typing(presence(map_regions(maps, unit_of_factor)), train_y, train_metadata[:, 0])

    folds = image_folds(len(tokens), args.folds, args.seed)
    print("[INFO] Fitting the patch-concept ridge", flush=True)
    readout = cross_fit_readout(tokens, targets, folds, ridges, device=args.device)
    readout_fit = readout["full_fit"]
    agreement = heldout_agreement(tokens, targets, folds, readout["best_ridge"], train_regions, unit_of_factor,
                                  args.device)
    pooled_train = mean_pool(tokens)
    del tokens, targets

    probes = {}
    for split in args.probe_splits:
        rows = torch.tensor([position[int(index)] for index in dataset.get_subset(split).indices])
        probes[split] = {"images": len(rows), "parameters": fit_probe(pooled_train[rows], train_y[rows], args.l2)}
    del pooled_train

    print(f"[INFO] Caching {len(evaluation)} {eval_split} images", flush=True)
    cached = cache_evaluation(encoder, evaluation, readout_fit, unit_of_factor, args)
    eval_y = cached["labels"].long()
    groups = cached["metadata"][:, 0].long() + 2 * cached["metadata"][:, 1].long()
    if args.eval_concept_maps is not None:
        eval_payload = torch.load(args.eval_concept_maps, map_location="cpu", weights_only=True)
        if [str(name) for name in eval_payload["factor_names"]] != names:
            raise ValueError(f"{args.eval_concept_maps} holds other factors than {args.concept_maps}.")
        regions = map_regions(aligned_maps(eval_payload, evaluation.indices), unit_of_factor, side)
        region_source = "clip_maps"
    else:
        regions = hard_regions(cached["unit_mass"])
        region_source = "ridge_from_ijepa_tokens"
    shown = presence(regions)
    intact = cached["pooled"]
    intact_logits = {split: logistic_logits(probe["parameters"], intact).float() for split, probe in probes.items()}

    def measure(region_of_unit: torch.Tensor, unit: int) -> tuple[dict, dict, dict]:
        keep, valid = drop_masks(region_of_unit, shown[:, unit])
        rows = torch.nonzero(valid).flatten()
        occluded = intact.clone()
        if len(rows):
            occluded[rows] = encode(encoder, cached["images"][rows], keep[rows], args)
        shift = 1 - torch.nn.functional.cosine_similarity(intact[rows], occluded[rows], dim=1)
        statistics = {"images": len(rows), "embedding_shift": round(float(shift.mean()) if len(rows) else 0.0, 5),
                      "removed_share": round(float((~keep[rows]).float().mean()) if len(rows) else 0.0, 4)}
        logits = {split: logistic_logits(probe["parameters"], occluded).float() for split, probe in probes.items()}
        effects = {split: paired_effect(intact_logits[split], logits[split], eval_y, groups, args.bootstrap,
                                        args.seed + unit)
                   for split in probes}
        return statistics, effects, logits

    rows_out = []
    for unit in range(units):
        rate = float(shown[:, unit].float().mean())
        entry = {"unit": unit, "name": " / ".join(labels_of_units[unit]), "factors": labels_of_units[unit],
                 "eval_presence": round(rate, 4),
                 "eval_area": round(float(regions[:, :, unit][shown[:, unit]].float().mean()) if rate else 0.0, 4),
                 "typing": typing[unit], "attributed": rate >= MIN_PRESENCE}
        if entry["attributed"]:
            start = time.time()
            entry["removal"], entry["effects"], removed_logits = measure(regions[:, :, unit], unit)
            control_logits = {split: None for split in probes}
            if not args.skip_shuffled:
                entry["shuffled_removal"], entry["shuffled_effects"], control_logits = measure(
                    shuffled_regions(regions, unit, args.seed), unit)
            for split, effect in entry["effects"].items():
                low, high = effect["wga_effect_ci95"]
                effect["wga_verdict"] = "raises WGA" if low > 0 else "lowers WGA" if high < 0 else "neutral"
            entry["shortcut_index"] = {
                split: shortcut_index(intact_logits[split], removed_logits[split], control_logits[split], eval_y,
                                      groups, CONFLICT_GROUPS, args.bootstrap, args.seed + unit)
                for split in probes}
            summary = ", ".join(f"{split} WGA {entry['effects'][split]['wga_effect']:+.2f} index "
                                f"{index[index['verdict_from']]:+.2f} {index['verdict']}"
                                for split, index in entry["shortcut_index"].items())
            print(f"[INFO] unit {unit} ({entry['name']}, {typing[unit]['type']}): shift "
                  f"{entry['removal']['embedding_shift']:.4f}, {summary}, {time.time() - start:.0f} s", flush=True)
        rows_out.append(entry)

    attributed = [entry for entry in rows_out if entry["attributed"]]
    shifts = [entry["removal"]["embedding_shift"] for entry in attributed]
    correlation = {split: {
        "wga_effect": round(spearman(shifts, [entry["effects"][split]["wga_effect"] for entry in attributed]), 4),
        "shortcut_index": round(spearman(shifts, [entry["shortcut_index"][split][entry["shortcut_index"][split]
                                                                                ["verdict_from"]]
                                                  for entry in attributed]), 4),
    } for split in probes}
    ranking = {split: [{"unit": entry["unit"], "name": entry["name"], "type": entry["typing"]["type"],
                        "index": entry["shortcut_index"][split][entry["shortcut_index"][split]["verdict_from"]],
                        "ci95": entry["shortcut_index"][split][entry["shortcut_index"][split]["verdict_from"] + "_ci95"],
                        "verdict": entry["shortcut_index"][split]["verdict"]}
                       for entry in sorted(attributed, reverse=True,
                                           key=lambda item: item["shortcut_index"][split][
                                               item["shortcut_index"][split]["verdict_from"]])]
               for split in probes}
    baseline = {}
    for split, logits in intact_logits.items():
        accuracy = group_accuracy(logits.argmax(dim=1), eval_y, groups)
        baseline[split] = {"images": probes[split]["images"],
                           "accuracy": round(100 * float((logits.argmax(dim=1) == eval_y).float().mean()), 2),
                           "worst_group_accuracy": round(float(accuracy.nan_to_num(nan=float("inf")).min()), 2),
                           "group_accuracy": [round(float(value), 2) for value in accuracy]}
    result = {
        "schema": "ijepa-token-drop-v2",
        "model": args.model,
        "eval_split": eval_split,
        "eval_images": len(eval_y),
        "group_names": list(GROUP_NAMES),
        "concept_maps": str(args.concept_maps),
        "factors": len(names),
        "units": units,
        "unit_seed": args.unit_seed,
        "patch_grid": side,
        "region_source": region_source,
        "ridge": {"best_ridge": readout["best_ridge"], "explained_variance": round(readout["explained_variance"], 4),
                  "heldout_region_agreement_with_clip": agreement},
        "probes": baseline,
        "spearman_shift_vs": correlation,
        "shortcut_ranking": ranking,
        "attribution": rows_out,
        "meaning": {
            "wga_effect": "WGA after removing the unit's patches minus WGA of the intact images, in points; "
                          "the interval is a paired bootstrap over images within groups",
            "wga_verdict": "from the WGA-effect interval",
            "shortcut_index": "mean accuracy change (points) over the conflict groups (landbird on water, waterbird on "
                              "land) minus that over the aligned groups; net subtracts the same index of the "
                              "shuffled control; also for the true-class margin",
            "verdict": "shortcut when the net accuracy index interval lies above 0, needed when below, neutral "
                       "otherwise (the removed index when the control is skipped)",
            "embedding_shift": "mean 1 - cos between intact and removed features over the images that show the unit; "
                               "label-free reliance",
            "shuffled_effects": "control: each image loses another image's region of the unit",
            "typing": "evaluation only, from hidden labels and the CLIP maps of the training images, as in CSA",
        },
        "seconds": round(time.time() - started),
    }
    path = atomic_write_json(output_dir / "token_drop.json", result)
    print(f"[results] Attribution: {path}")
    return path


if __name__ == "__main__":
    main()
