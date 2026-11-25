"""Table of fine-tuning runs grouped by setting, ranked by last-epochs validation WGA.

    python -m cospro.cli.summarize_finetune [--study ijepa_finetune] [--head dfr_head]

Reads every ``run.json`` (schema ``ijepa-finetune-v2``) under ``outputs/seeds/<study>/`` and averages
the last-epochs means over seeds. Validation decides the setting; test columns appear only for runs
made with ``--final_test`` and are for reporting.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from cospro.tracking.artifacts import resolve_output_root

SETTING = ("model", "train_split", "dfr_split", "trainable_blocks", "epochs", "lr", "head_lr", "weight_decay")


def load_runs(study: str) -> list[dict]:
    runs = []
    for path in sorted((resolve_output_root() / "seeds" / study).glob("seed_*/*/*/run.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == "ijepa-finetune-v2":
            runs.append(payload)
    return runs


def cell(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f}±{statistics.stdev(values):.1f}"


def table(runs: list[dict], head: str) -> list[str]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        groups[tuple(run["config"][key] for key in SETTING)].append(run)

    def metric(run: dict, split: str, name: str) -> float | None:
        report = run["last_epochs"].get(head, {}).get(split)
        return None if report is None else report[name]["mean"]

    rows = []
    for setting, members in groups.items():
        columns = {}
        for split in ("val", "test"):
            for name in ("accuracy", "worst_group_accuracy"):
                columns[split, name] = [value for value in (metric(run, split, name) for run in members)
                                        if value is not None]
        rows.append((setting, members, columns))
    rows.sort(key=lambda row: (row[0][0], -statistics.mean(row[2]["val", "worst_group_accuracy"] or [0.0])))
    lines = [f"head: {head}; values are means over the last epochs, then mean±std over seeds",
             f"{'model':22} {'split':8} {'dfr':8} {'blk':>3} {'ep':>4} {'lr':>7} {'n':>2} "
             f"{'val acc':>10} {'val WGA':>10} {'test acc':>10} {'test WGA':>10}"]
    for setting, members, columns in rows:
        model, split, dfr, blocks, epochs, lr = setting[:6]
        lines.append(f"{model:22} {split:8} {dfr:8} {blocks:>3} {epochs:>4} {lr:>7g} {len(members):>2} "
                     f"{cell(columns['val', 'accuracy']):>10} {cell(columns['val', 'worst_group_accuracy']):>10} "
                     f"{cell(columns['test', 'accuracy']):>10} {cell(columns['test', 'worst_group_accuracy']):>10}")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="ijepa_finetune")
    parser.add_argument("--head", choices=("finetuned_head", "dfr_head", "both"), default="both")
    args = parser.parse_args(argv)
    runs = load_runs(args.study)
    if not runs:
        print(f"No ijepa-finetune-v2 runs under outputs/seeds/{args.study}/")
        return
    for head in (("finetuned_head", "dfr_head") if args.head == "both" else (args.head,)):
        print("\n".join(table(runs, head)) + "\n")


if __name__ == "__main__":
    main()
