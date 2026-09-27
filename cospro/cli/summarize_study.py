"""Summarize every run record of one study into a JSON and a Markdown table.

Reads ``outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json`` and writes
``outputs/reports/<study>/summary.{json,md}``: one row per run and one line per dataset and arm with
the mean and spread over seeds. ``scripts/run_training.sbatch`` refreshes it after every run.

    python -m cospro.cli.summarize_study --study concept_factors_metashift

The headline number is ``val WGA (last 4)``: worst-group accuracy of the validation probe averaged
over the last four periodic probes (SSL epochs 425 to 500 at the default frequency), which is steadier
than the single last probe on small validation groups.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from cospro.tracking.artifacts import atomic_write_json, report, resolve_output_root

CONFIG_KEYS = (
    "dataset", "model", "epochs", "batch_size", "learning_rate", "weight_decay", "temp", "splice_mode",
    "splice_weight", "factor_distill_weight", "factor_condition_fraction", "factor_targets",
    "factor_merge_similarity", "factor_min_frequency", "factor_start_epoch", "latetvg_prune_rate",
    "latetvg_layers", "concept_factor_count", "concept_factor_groups",
)
LAST_PROBES = 4


def probe_trajectory(record: dict) -> list[dict[str, float]]:
    """The last validation probe of every SSL epoch that ran one, in epoch order."""

    by_epoch: dict[int, dict[str, float]] = {}
    for event in record.get("metrics", []):
        values = event.get("values", {})
        if event.get("stage") == "linear_probe" and "ssl_epoch" in values and "eval_worst_group_accuracy" in values:
            by_epoch[int(values["ssl_epoch"])] = {
                "epoch": int(values["ssl_epoch"]),
                "wga": float(values["eval_worst_group_accuracy"]),
                "acc": float(values.get("eval_accuracy", float("nan"))),
            }
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def last_ssl_value(record: dict, key: str) -> float | None:
    for event in reversed(record.get("metrics", [])):
        if event.get("stage") == "ssl" and key in event.get("values", {}):
            return float(event["values"][key])
    return None


def run_row(path: Path, root: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    config = record.get("config", {})
    final = record.get("final_metrics", {})
    trajectory = probe_trajectory(record)
    tail = trajectory[-LAST_PROBES:]
    parts = path.relative_to(root).parts
    return {
        "run": path.parent.relative_to(root).as_posix(),
        "seed": int(config.get("seed", -1)),
        "arm": parts[1] if len(parts) > 3 else str(config.get("arm", "")),
        "status": record.get("status"),
        "config": {key: config.get(key) for key in CONFIG_KEYS if key in config},
        "val_wga_last": final.get("Last linear val worst-group acc"),
        "val_acc_last": final.get("Last linear val acc"),
        "val_wga_last4": statistics.mean(point["wga"] for point in tail) if tail else None,
        "val_acc_last4": statistics.mean(point["acc"] for point in tail) if tail else None,
        "val_group_accuracies": final.get("Linear val group accuracies"),
        "spurious_probe_val_wga": final.get("Spurious probe last val worst-group acc"),
        "factor_explained_variance": last_ssl_value(record, "SSL relational_factor_explained_variance"),
        "factor_heldout_explained_variance": last_ssl_value(
            record, "SSL relational_factor_heldout_explained_variance"),
        "trajectory": trajectory,
        "wandb": (record.get("wandb") or {}).get("url"),
    }


def spread(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize(rows: list[dict]) -> list[dict]:
    arms: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["status"] == "complete":
            arms.setdefault((str(row["config"].get("dataset")), row["arm"]), []).append(row)
    summary = []
    for (dataset, arm), members in sorted(arms.items()):
        summary.append({
            "dataset": dataset,
            "arm": arm,
            "seeds": sorted(member["seed"] for member in members),
            "val_wga_last4": spread([m["val_wga_last4"] for m in members if m["val_wga_last4"] is not None]),
            "val_wga_last": spread([m["val_wga_last"] for m in members if m["val_wga_last"] is not None]),
            "val_acc_last4": spread([m["val_acc_last4"] for m in members if m["val_acc_last4"] is not None]),
            "factor_explained_variance": spread(
                [m["factor_explained_variance"] for m in members if m["factor_explained_variance"] is not None]
            ),
            "factor_heldout_explained_variance": spread(
                [m["factor_heldout_explained_variance"] for m in members
                 if m.get("factor_heldout_explained_variance") is not None]
            ),
            "val_group_accuracies": [
                statistics.mean(values) for values in zip(*[m["val_group_accuracies"] for m in members
                                                            if m["val_group_accuracies"]])
            ],
        })
    return summary


def number(value: dict[str, float | None]) -> str:
    if value["mean"] is None:
        return ""
    return f"{value['mean']:.1f}" + ("" if value["sd"] is None else f" ± {value['sd']:.1f}")


def markdown(study: str, summary: list[dict], rows: list[dict]) -> str:
    lines = [f"# {study}", "",
             "Validation probe; `last 4` averages the last four periodic probes. ± is the standard deviation "
             "over seeds. `val group acc` lists the mean accuracy of every (class, attribute) group in the "
             "dataset's group order. `held-out expl. var.` is the factor variance a ridge regression fitted "
             "on the other half of the batch explains, which memorizing images cannot raise.", ""]
    for dataset in sorted({entry["dataset"] for entry in summary}):
        lines += [f"## {dataset}", "",
                  "| arm | seeds | val WGA (last 4) | val WGA (last) | val acc (last 4) | val group acc | "
                  "factor expl. var. | held-out expl. var. |",
                  "|---|---|---|---|---|---|---|---|"]
        entries = [entry for entry in summary if entry["dataset"] == dataset]
        entries.sort(key=lambda entry: -(entry["val_wga_last4"]["mean"] or 0))
        for entry in entries:
            explained = entry["factor_explained_variance"]["mean"]
            explained_text = "" if explained is None else f"{explained:.2f}"
            heldout = entry["factor_heldout_explained_variance"]["mean"]
            heldout_text = "" if heldout is None else f"{heldout:.2f}"
            groups = " / ".join(f"{value:.0f}" for value in entry["val_group_accuracies"])
            lines.append(
                f"| {entry['arm']} | {','.join(map(str, entry['seeds']))} | {number(entry['val_wga_last4'])} | "
                f"{number(entry['val_wga_last'])} | {number(entry['val_acc_last4'])} | {groups} | "
                f"{explained_text} | {heldout_text} |"
            )
        lines.append("")
    unfinished = [row for row in rows if row["status"] != "complete"]
    if unfinished:
        lines += ["## Unfinished runs", ""] + [f"- `{row['run']}`: {row['status']}" for row in unfinished] + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    args = parser.parse_args(argv)
    root = resolve_output_root() / "seeds" / args.study
    rows = [run_row(path, root) for path in sorted(root.glob("seed_*/*/*/run.json"))]
    summary = summarize(rows)
    destination = report(args.study)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "summary.json", {"study": args.study, "arms": summary, "runs": rows})
    (destination / "summary.md").write_text(markdown(args.study, summary, rows), encoding="utf-8")
    print(f"[results] {len(rows)} runs of {args.study}: {destination / 'summary.md'}")
    return destination / "summary.md"


if __name__ == "__main__":
    main()
