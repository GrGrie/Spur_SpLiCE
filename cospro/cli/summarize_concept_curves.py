"""Summarize the concept audits of a study (docs/PROJECT_PLAN.md, Section 7).

Reads every ``concept_curves.json`` and ``flagged_units.json`` under ``outputs/seeds/<study>/`` and writes
``outputs/reports/<study>/concept_curves.json`` and ``concept_curves.md``: per arm and seed the detection
AUROC of the onset scores and the flagged units with their hidden-label types, the agreement of the onset
scores between seeds and the audit report (every unit with its words, onset score and type). The report opens
with the verdicts of the Early Concept Swap kill test (docs/PROJECT_ECS.md, Section 3): stage K1 from the
audits of the arm ``curves``, stage K2 from the validation WGA of every ECS arm against ``simclr`` and the
check that the reliance on the flagged units falls once the swap starts. When the study
holds the single-unit arms ``curves_unit_<k>``, the report opens with the concept attribution table of
docs/PROJECT_CSA.md: per unit the onset rank and the paired change of validation WGA against SimCLR.
Worst-group accuracy per arm stays in the study summary of ``cospro.cli.summarize_study``. This command reads JSON
only, so it runs on any machine after ``git pull``.

    python -m cospro.cli.summarize_concept_curves --study factors_metashift
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from cospro.cli.summarize_study import run_row
from cospro.curves.core import spearman
from cospro.tracking.artifacts import atomic_write_json, report, resolve_output_root

# Arms that train as plain SimCLR, in order of preference as the reference of the attribution table.
REFERENCE_ARMS = ("simclr", "curves_audit")
UNIT_ARM = re.compile(r"curves_unit_\d+")
# Arms of the ECS kill test besides `curves_*` ablations, and the gain over SimCLR that stage K2 asks for.
ECS_ARMS = ("curves",)
K2_MIN_GAIN = 5.0


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


def probe_results(study_root: Path) -> dict[str, dict[int, dict]]:
    """Per arm and seed the validation probe of the newest complete run: ``{"wga", "acc"}`` (last four probes)."""

    results: dict[str, dict[int, dict]] = {}
    for path in sorted(study_root.glob("seed_*/*/*/run.json")):
        row = run_row(path, study_root)
        if row["status"] == "complete" and row["val_wga_last4"] is not None:
            results.setdefault(row["arm"], {})[row["seed"]] = {"wga": row["val_wga_last4"], "acc": row["val_acc_last4"]}
    return results


def attribution(audits: dict[str, dict[int, dict]], results: dict[str, dict[int, dict]]) -> dict | None:
    """Concept Shortcut Attribution (docs/PROJECT_CSA.md): the effect of swapping one unit on the probe.

    Every arm ``curves_unit_<k>`` is compared with the reference arm on the seeds both finished. The effect of
    unit ``k`` is the mean paired difference of validation WGA. ``onset_effect_spearman`` relates the onset
    score of a unit to its effect across units.
    """

    reference_arm = next((arm for arm in REFERENCE_ARMS if arm in results), None)
    unit_arms = {int(arm.rsplit("_", 1)[1]): arm for arm in results if UNIT_ARM.fullmatch(arm)}
    if reference_arm is None or not unit_arms:
        return None
    reference, rows = results[reference_arm], []
    for unit, arm in sorted(unit_arms.items()):
        seeds = sorted(set(results[arm]) & set(reference))
        entries = [audits[arm][seed]["curves"] for seed in seeds if seed in audits.get(arm, {})]
        entries = [curves for curves in entries if curves.get("onset") is not None and unit < len(curves["units"])]
        if not seeds or not entries or not entries[0]["units"][unit]["audited"]:
            continue
        gains = [results[arm][seed]["wga"] - reference[seed]["wga"] for seed in seeds]
        typing = entries[0]["evaluation_only"].get("typing")
        rows.append({
            "unit": unit, "name": entries[0]["units"][unit]["name"], "seeds": seeds,
            "presence": entries[0]["units"][unit]["presence"], "area": entries[0]["units"][unit]["area"],
            "onset": statistics.fmean(curves["onset"][unit] for curves in entries),
            "type": "n/a" if typing is None else typing[unit]["type"],
            "wga_effect": statistics.fmean(gains),
            "wga_effect_sd": statistics.stdev(gains) if len(gains) > 1 else None,
            "positive_seeds": sum(gain > 0 for gain in gains),
            "accuracy_effect": statistics.fmean(results[arm][seed]["acc"] - reference[seed]["acc"] for seed in seeds),
        })
    if not rows:
        return None
    rows.sort(key=lambda row: -row["onset"])
    for rank, row in enumerate(rows, start=1):
        row["onset_rank"] = rank
    correlation = spearman([row["onset"] for row in rows], [row["wga_effect"] for row in rows]) if len(rows) > 2 else None
    return {"reference_arm": reference_arm, "units": rows,
            "onset_effect_spearman": None if correlation is None or math.isnan(correlation) else correlation}


def render_attribution(result: dict) -> list[str]:
    correlation = result["onset_effect_spearman"]
    lines = ["## Concept attribution", "",
             f"Each row swaps one concept unit alone after the early window (arm `curves_unit_<k>`) and reports the "
             f"paired change of validation WGA against `{result['reference_arm']}`, in points. Spearman correlation of "
             f"onset score and WGA effect across units: {'n/a' if correlation is None else f'{correlation:.2f}'}.", "",
             "| onset rank | unit | words | onset | area | hidden-label type | WGA effect | sd | seeds up | accuracy effect |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in result["units"]:
        deviation = "n/a" if row["wga_effect_sd"] is None else f"{row['wga_effect_sd']:.1f}"
        lines.append(f"| {row['onset_rank']} | {row['unit']} | {row['name']} | {row['onset']:.3f} | {row['area']:.2f} | "
                     f"{row['type']} | {row['wga_effect']:+.1f} | {deviation} | {row['positive_seeds']}/{len(row['seeds'])} | "
                     f"{row['accuracy_effect']:+.1f} |")
    return [*lines, ""]


def majority(count: int, total: int) -> bool:
    """Three of four seeds, generalized: at least three quarters of the seeds."""

    return total > 0 and count >= math.ceil(0.75 * total)


def flagged_share_change(entry: dict) -> dict | None:
    """Summed share of the flagged units at the end of the early window and at the last audit of a run.

    Theory check of docs/PROJECT_ECS.md, Section 5: once the swap starts, the encoder's reliance on the
    flagged units has to fall.
    """

    curves, flags = entry["curves"], entry["flags"]
    if not flags or not flags.get("flagged"):
        return None
    epochs, early = curves["epochs"], int(curves["early_epochs"])
    if early not in epochs or max(epochs) <= early:
        return None
    before = sum(curves["share"][epochs.index(early)][unit] for unit in flags["flagged"])
    after = sum(curves["share"][-1][unit] for unit in flags["flagged"])
    return {"at_early_end": before, "at_last_audit": after, "last_audit_epoch": max(epochs), "fell": after < before}


def k1_verdict(seeds: dict[int, dict]) -> dict:
    """Stage K1 of the ECS kill test on the arm ``curves``: does the audit find the spurious concept?"""

    rows = []
    for seed, entry in sorted(seeds.items()):
        curves, flags = entry["curves"], entry["flags"]
        typing = curves["evaluation_only"].get("typing")
        if curves.get("onset") is None or typing is None or flags is None:
            continue
        audited = [unit["unit"] for unit in curves["units"] if unit["audited"]]
        leader = max(audited, key=lambda unit: (curves["onset"][unit], -unit))
        types = [typing[unit]["type"] for unit in flags["flagged"]]
        rows.append({"seed": seed, "auroc": curves["evaluation_only"].get("detection_auroc"),
                     "leader_type": typing[leader]["type"], "flagged_types": types,
                     "clean_flags": "attribute" in types and "class" not in types})
    scores = [row["auroc"] for row in rows if row["auroc"] is not None]
    if not rows or not scores:
        return {"verdict": "pending", "seeds": rows, "mean_auroc": None}
    mean = statistics.fmean(scores)
    high = sum(score >= 0.7 for score in scores)
    clean = sum(row["clean_flags"] for row in rows)
    class_leads = sum(row["leader_type"] == "class" for row in rows)
    if mean < 0.55 or class_leads * 2 > len(rows):
        verdict = "fail"
    elif majority(high, len(scores)) and majority(clean, len(rows)):
        verdict = "pass"
    else:
        verdict = "weak"
    return {"verdict": verdict, "seeds": rows, "mean_auroc": mean, "seeds_auroc_at_least_0.7": high,
            "seeds_with_clean_flags": clean, "seeds_where_a_class_unit_leads": class_leads}


def gain_over(results: dict[str, dict[int, dict]], arm: str, reference: str) -> dict | None:
    """Paired WGA gain of ``arm`` over ``reference`` on the seeds both finished."""

    if arm not in results or reference not in results:
        return None
    seeds = sorted(set(results[arm]) & set(results[reference]))
    if not seeds:
        return None
    gains = [results[arm][seed]["wga"] - results[reference][seed]["wga"] for seed in seeds]
    return {"seeds": seeds, "mean": statistics.fmean(gains), "up": sum(gain > 0 for gain in gains)}


def k2_verdict(results: dict[str, dict[int, dict]]) -> dict:
    """Stage K2 of the ECS kill test: does the swap raise WGA? Thresholds of docs/PROJECT_ECS.md, Section 3."""

    method = gain_over(results, "curves", "simclr")
    random_control = gain_over(results, "curves", "curves_random")
    oracle = gain_over(results, "curves_oracle", "simclr")
    arms = {arm: gain_over(results, arm, "simclr") for arm in sorted(results)
            if arm in ECS_ARMS or (arm.startswith("curves_") and not UNIT_ARM.fullmatch(arm))}

    def passes(gain: dict | None) -> bool:
        return gain is not None and gain["mean"] >= K2_MIN_GAIN and majority(gain["up"], len(gain["seeds"]))

    if method is None or (passes(method) and random_control is None):
        verdict = "pending"
    elif passes(method) and majority(random_control["up"], len(random_control["seeds"])):
        verdict = "pass"
    elif oracle is None:
        verdict = "pending"
    elif passes(oracle):
        verdict = "partial"
    else:
        verdict = "fail"
    return {"verdict": verdict, "curves_vs_simclr": method, "curves_vs_random": random_control,
            "oracle_vs_simclr": oracle, "arms_vs_simclr": {arm: gain for arm, gain in arms.items() if gain}}


def ecs_kill_test(audits: dict[str, dict[int, dict]], results: dict[str, dict[int, dict]]) -> dict | None:
    """Both stages of the ECS kill test plus the post-swap reliance check of every swapping ECS arm."""

    if "curves" not in audits and "curves" not in results:
        return None
    falls = {}
    for arm, seeds in sorted(audits.items()):
        if UNIT_ARM.fullmatch(arm):
            continue
        changes = [change for change in (flagged_share_change(entry) for entry in seeds.values()) if change]
        if changes:
            falls[arm] = {"seeds": len(changes), "fell": sum(change["fell"] for change in changes),
                          "mean_at_early_end": statistics.fmean(change["at_early_end"] for change in changes),
                          "mean_at_last_audit": statistics.fmean(change["at_last_audit"] for change in changes)}
    return {"k1": k1_verdict(audits.get("curves", {})), "k2": k2_verdict(results), "flagged_share_after_swap": falls}


def render_kill_test(result: dict) -> list[str]:
    def number(value, digits=2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    k1, k2 = result["k1"], result["k2"]
    lines = ["## Early Concept Swap kill test", "",
             f"**Stage K1 (audit finds the spurious concept): {k1['verdict']}.** Mean detection AUROC of `curves` "
             f"{number(k1['mean_auroc'])}.", "",
             "| seed | detection AUROC | type of the leading unit | types of the flagged units | clean flags |",
             "|---|---|---|---|---|"]
    for row in k1["seeds"]:
        lines.append(f"| {row['seed']} | {number(row['auroc'])} | {row['leader_type']} | {', '.join(row['flagged_types'])} | "
                     f"{'yes' if row['clean_flags'] else 'no'} |")
    lines += ["", f"**Stage K2 (swap raises WGA): {k2['verdict']}.** Paired change of validation WGA (last four "
              f"probes) against `simclr`, in points; a pass needs at least +{K2_MIN_GAIN:g} and a gain on three "
              "quarters of the seeds, and `curves` above `curves_random` on three quarters of the seeds.", "",
              "| arm | seeds | mean WGA change | seeds up |", "|---|---|---|---|"]
    for arm, gain in k2["arms_vs_simclr"].items():
        lines.append(f"| {arm} | {len(gain['seeds'])} | {gain['mean']:+.1f} | {gain['up']}/{len(gain['seeds'])} |")
    control = k2["curves_vs_random"]
    if control:
        lines += ["", f"`curves` against `curves_random`: {control['mean']:+.1f} points, up on {control['up']} of "
                  f"{len(control['seeds'])} seeds."]
    if result["flagged_share_after_swap"]:
        lines += ["", "**Reliance on the flagged units after the swap starts** (summed share; the theory predicts a fall).",
                  "", "| arm | seeds | share at end of early window | share at last audit | seeds where it fell |",
                  "|---|---|---|---|---|"]
        for arm, row in result["flagged_share_after_swap"].items():
            lines.append(f"| {arm} | {row['seeds']} | {row['mean_at_early_end']:.3f} | {row['mean_at_last_audit']:.3f} | "
                         f"{row['fell']}/{row['seeds']} |")
    return [*lines, ""]


def render(study: str, arms: dict[str, dict], attributed: dict | None = None, kill_test: dict | None = None) -> str:
    def number(value) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = [f"# Concept audits of {study}", "",
             "Detection AUROC: how well the onset score separates attribute units from class units. Types come from "
             "hidden labels and are used for this report only.", ""]
    if kill_test is not None:
        lines += render_kill_test(kill_test)
    if attributed is not None:
        lines += render_attribution(attributed)
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
    study_root = resolve_output_root(args.output_root) / "seeds" / args.study
    audits = newest_audits(study_root)
    if not audits:
        raise SystemExit(f"No concept_curves.json under the study {args.study}.")
    arms = {arm: summarize_arm(seeds) for arm, seeds in sorted(audits.items())}
    results = probe_results(study_root)
    attributed = attribution(audits, results)
    kill_test = ecs_kill_test(audits, results)
    directory = report(args.study, root=args.output_root)
    atomic_write_json(directory / "concept_curves.json", {"study": args.study, "arms": arms, "attribution": attributed,
                                                         "ecs_kill_test": kill_test})
    summary = directory / "concept_curves.md"
    summary.write_text(render(args.study, arms, attributed, kill_test), encoding="utf-8")
    print(f"[results] Concept audit summary: {summary}")
    return summary


if __name__ == "__main__":
    main()
