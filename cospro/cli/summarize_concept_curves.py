"""Summarize the concept audits of a study (docs/PROJECT_PLAN.md, Section 7).

Reads every ``concept_curves.json`` and ``flagged_units.json`` under ``outputs/seeds/<study>/`` and writes
``outputs/reports/<study>/concept_curves.json`` and ``concept_curves.md``: per arm and seed the detection
AUROC of the onset scores and the flagged units with their hidden-label types, the agreement of the onset
scores between seeds and the audit report (every unit with its words, onset score and type). Worst-group
accuracy per arm stays in the study summary of ``cospro.cli.summarize_study``. This command reads JSON
only, so it runs on any machine after ``git pull``.

    python -m cospro.cli.summarize_concept_curves --study factors_metashift
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from cospro.curves.core import spearman
from cospro.tracking.artifacts import atomic_write_json, report, resolve_output_root


def newest_audits(study_root: Path) -> dict[str, dict[int, dict]]:
    """Per arm and seed the audit of the newest attempt: ``{"curves", "flags", "path"}``."""

    audits: dict[str, dict[int, dict]] = {}
    for curves_path in sorted(study_root.glob("seed_*/*/*/**/concept_curves.json")):
        relative = curves_path.relative_to(study_root).parts
        seed, arm = int(relative[0].split("_")[1]), relative[1]
        flags_path = curves_path.with_name("flagged_units.json")
        # Attempt folders sort by submission, so the last one seen per arm and seed is the newest.
        audits.setdefault(arm, {})[seed] = {
            "curves": json.loads(curves_path.read_text(encoding="utf-8")),
            "flags": json.loads(flags_path.read_text(encoding="utf-8")) if flags_path.is_file() else None,
            "path": str(curves_path),
        }
    return audits


def summarize_arm(seeds: dict[int, dict]) -> dict:
    per_seed, onsets = [], {}
    for seed, entry in sorted(seeds.items()):
        curves, flags = entry["curves"], entry["flags"]
        if curves.get("onset") is None:
            continue
        onsets[seed] = curves["onset"]
        per_seed.append({
            "seed": seed, "last_audit_epoch": max(curves["epochs"]),
            "detection_auroc": curves["evaluation_only"].get("detection_auroc"),
            "rule": None if flags is None else flags["rule"],
            "flagged": [] if flags is None else flags["names"],
            "flagged_types": [] if flags is None else (flags.get("types") or []),
        })
    order = sorted(onsets)
    agreement = [spearman(onsets[first], onsets[second]) for position, first in enumerate(order)
                 for second in order[position + 1:]]
    agreement = [value for value in agreement if not math.isnan(value)]
    units = []
    if order:
        curves = seeds[order[0]]["curves"]
        typing = curves["evaluation_only"].get("typing")
        for unit in curves["units"]:
            if unit["audited"]:
                units.append({"unit": unit["unit"], "name": unit["name"], "presence": unit["presence"],
                              "area": unit["area"],
                              "onset": statistics.fmean(onsets[seed][unit["unit"]] for seed in order),
                              "type": "n/a" if typing is None else typing[unit["unit"]]["type"]})
        units.sort(key=lambda row: -row["onset"])
    scores = [entry["detection_auroc"] for entry in per_seed if entry["detection_auroc"] is not None]
    return {"per_seed": per_seed, "mean_detection_auroc": statistics.fmean(scores) if scores else None,
            "seed_agreement": statistics.fmean(agreement) if agreement else None, "units": units}


def render(study: str, arms: dict[str, dict]) -> str:
    def number(value) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = [f"# Concept audits of {study}", "",
             "Detection AUROC: how well the onset score separates attribute units from class units. Types come from "
             "hidden labels and are used for this report only.", ""]
    for arm, summary in arms.items():
        if not summary["per_seed"]:
            continue
        lines += [f"## {arm}", "", f"Mean detection AUROC {number(summary['mean_detection_auroc'])}, seed agreement "
                  f"of the onset scores (mean Spearman) {number(summary['seed_agreement'])}.", "",
                  "| seed | audited to epoch | detection AUROC | rule | flagged units | types |", "|---|---|---|---|---|---|"]
        for entry in summary["per_seed"]:
            lines.append(f"| {entry['seed']} | {entry['last_audit_epoch']} | {number(entry['detection_auroc'])} | "
                         f"{entry['rule'] or ''} | {'; '.join(entry['flagged'])} | {', '.join(entry['flagged_types'])} |")
        lines += ["", "| unit | words | presence | area | mean onset | hidden-label type |", "|---|---|---|---|---|---|"]
        for unit in summary["units"]:
            lines.append(f"| {unit['unit']} | {unit['name']} | {unit['presence']:.2f} | {unit['area']:.2f} | "
                         f"{unit['onset']:.3f} | {unit['type']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    audits = newest_audits(resolve_output_root(args.output_root) / "seeds" / args.study)
    if not audits:
        raise SystemExit(f"No concept_curves.json under the study {args.study}.")
    arms = {arm: summarize_arm(seeds) for arm, seeds in sorted(audits.items())}
    directory = report(args.study, root=args.output_root)
    atomic_write_json(directory / "concept_curves.json", {"study": args.study, "arms": arms})
    summary = directory / "concept_curves.md"
    summary.write_text(render(args.study, arms), encoding="utf-8")
    print(f"[results] Concept audit summary: {summary}")
    return summary


if __name__ == "__main__":
    main()
