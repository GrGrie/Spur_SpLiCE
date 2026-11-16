"""Summarize a concept-curves study: the kill-test tables of docs/PROJECT_PLAN.md, Section 7.

Reads the run records under ``outputs/seeds/<study>/`` and writes ``outputs/reports/<study>/summary.json``
and ``summary.md``: validation worst-group accuracy per arm with paired wins over ERM, the detection
AUROC of the audit per seed, the agreement of the onset scores between seeds and the audit report (every
unit with its words, onset score, flag and hidden-label type). It reads JSON only, so it runs on any
machine after ``git pull``.

    python -m cospro.cli.summarize_concept_curves --study curves_metashift
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from cospro.curves.core import spearman
from cospro.tracking.artifacts import atomic_write_json, report, resolve_output_root


def newest_complete_runs(study_root: Path) -> dict[str, dict[int, dict]]:
    """Per arm and seed the newest complete run: ``{"dir", "run", "curves"}``."""

    runs: dict[str, dict[int, dict]] = {}
    for record_path in sorted(study_root.glob("seed_*/*/*/run.json")):
        run = json.loads(record_path.read_text(encoding="utf-8"))
        if run.get("status") != "complete":
            continue
        seed = int(record_path.parents[2].name.split("_")[1])
        arm = record_path.parents[1].name
        curves_path = record_path.with_name("concept_curves.json")
        curves = json.loads(curves_path.read_text(encoding="utf-8")) if curves_path.is_file() else None
        # Attempt names start with a UTC timestamp, so the sorted glob ends on the newest attempt.
        runs.setdefault(arm, {})[seed] = {"dir": str(record_path.parent), "run": run, "curves": curves}
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def arm_table(runs: dict[str, dict[int, dict]], split: str, reference: str) -> list[dict]:
    rows = []
    for arm, seeds in sorted(runs.items()):
        wga = {seed: entry["run"]["final"][f"{split}_wga"] for seed, entry in seeds.items()}
        accuracy = [entry["run"]["final"][f"{split}_acc"] for entry in seeds.values()]
        groups = [entry["run"]["final"][f"{split}_group_acc"] for entry in seeds.values()]
        mean, spread = mean_std(list(wga.values()))
        row = {"arm": arm, "seeds": sorted(wga), "wga": mean, "wga_std": spread, "acc": statistics.fmean(accuracy),
               "group_acc": [statistics.fmean(column) for column in zip(*groups)]}
        paired = [wga[seed] - runs[reference][seed]["run"]["final"][f"{split}_wga"]
                  for seed in wga if arm != reference and seed in runs.get(reference, {})]
        if paired:
            row.update({"gain_over_reference": statistics.fmean(paired),
                        "wins_over_reference": f"{sum(gain > 0 for gain in paired)}/{len(paired)}"})
        flags = [entry["run"].get("flags") for entry in seeds.values() if entry["run"].get("flags")]
        if flags:
            row["flagged"] = {str(seed): entry["run"]["flags"]["names"] for seed, entry in sorted(seeds.items())
                              if entry["run"].get("flags")}
        rows.append(row)
    return rows


def audit_summary(seeds: dict[int, dict]) -> dict:
    """Detection AUROC, top flags and seed agreement of one audited arm."""

    per_seed, onsets = [], {}
    for seed, entry in sorted(seeds.items()):
        curves = entry["curves"]
        if not curves or curves.get("onset") is None:
            continue
        evaluation = curves["evaluation_only"]
        onsets[seed] = curves["onset"]
        per_seed.append({"seed": seed, "detection_auroc": evaluation.get("detection_auroc"),
                         "top_flags": [curves["units"][unit]["name"] for unit in evaluation.get("top_flags", [])],
                         "top_flag_types": evaluation.get("top_flag_types", [])})
    order = sorted(onsets)
    agreement = [spearman(onsets[first], onsets[second]) for position, first in enumerate(order)
                 for second in order[position + 1:]]
    agreement = [value for value in agreement if not math.isnan(value)]
    report_rows = []
    if order:
        curves = seeds[order[0]]["curves"]
        typing = curves["evaluation_only"]["typing"]
        for unit in curves["units"]:
            if unit["audited"]:
                scores = [onsets[seed][unit["unit"]] for seed in order if len(onsets[seed]) > unit["unit"]]
                report_rows.append({"unit": unit["unit"], "name": unit["name"], "presence": unit["presence"],
                                    "area": unit["area"], "onset": statistics.fmean(scores),
                                    "type": typing[unit["unit"]]["type"]})
        report_rows.sort(key=lambda row: -row["onset"])
    return {"per_seed": per_seed, "seed_agreement": statistics.fmean(agreement) if agreement else None,
            "units": report_rows}


def render(study: str, split: str, table: list[dict], audits: dict[str, dict]) -> str:
    lines = [f"# {study}", "", f"Mean of the last four evaluations on `{split}`. Group order as the dataset defines it.", "",
             "| arm | seeds | WGA | acc | group acc | gain over erm | wins |", "|---|---|---|---|---|---|---|"]
    for row in table:
        groups = " / ".join(f"{100 * value:.0f}" for value in row["group_acc"])
        gain = f"{100 * row['gain_over_reference']:+.1f}" if "gain_over_reference" in row else ""
        lines.append(f"| {row['arm']} | {len(row['seeds'])} | {100 * row['wga']:.1f} +- {100 * row['wga_std']:.1f} | "
                     f"{100 * row['acc']:.1f} | {groups} | {gain} | {row.get('wins_over_reference', '')} |")
    for arm, audit in audits.items():
        if not audit["per_seed"]:
            continue
        lines += ["", f"## Audit of `{arm}`", ""]
        agreement = audit["seed_agreement"]
        lines.append("Seed agreement of the onset scores (mean Spearman): "
                     + ("n/a" if agreement is None else f"{agreement:.2f}"))
        lines += ["", "| seed | detection AUROC | flagged units | types |", "|---|---|---|---|"]
        for entry in audit["per_seed"]:
            score = entry["detection_auroc"]
            lines.append(f"| {entry['seed']} | {'n/a' if score is None else f'{score:.2f}'} | "
                         f"{'; '.join(entry['top_flags'])} | {', '.join(entry['top_flag_types'])} |")
        lines += ["", "| unit | words | presence | area | mean onset | hidden-label type |", "|---|---|---|---|---|---|"]
        for unit in audit["units"]:
            lines.append(f"| {unit['unit']} | {unit['name']} | {unit['presence']:.2f} | {unit['area']:.2f} | "
                         f"{unit['onset']:.3f} | {unit['type']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--reference", default="erm")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    runs = newest_complete_runs(resolve_output_root(args.output_root) / "seeds" / args.study)
    if not runs:
        raise SystemExit(f"No complete run of study {args.study}.")
    table = arm_table(runs, args.split, args.reference)
    audits = {arm: audit_summary(seeds) for arm, seeds in sorted(runs.items())
              if any(entry["run"]["config"]["mode"] == "erm" for entry in seeds.values())}
    directory = report(args.study, root=args.output_root)
    atomic_write_json(directory / "summary.json", {"study": args.study, "split": args.split, "arms": table,
                                                    "audits": audits})
    summary = directory / "summary.md"
    summary.write_text(render(args.study, args.split, table, audits), encoding="utf-8")
    print(f"[results] Summary: {summary}")
    return summary


if __name__ == "__main__":
    main()
