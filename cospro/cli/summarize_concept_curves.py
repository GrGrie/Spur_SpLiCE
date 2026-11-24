"""Summarize the concept audits of a study (docs/PROJECT_PLAN.md, Section 7).

Reads every ``concept_curves.json`` and ``flagged_units.json`` under ``outputs/seeds/<study>/`` and writes
``outputs/reports/<study>/concept_curves.json`` and ``concept_curves.md``: per arm and seed the detection
AUROC of the onset scores and the flagged units with their hidden-label types, the agreement of the onset
scores between seeds and the audit report (every unit with its words, onset score and type). The report opens
with the verdicts of the Early Concept Swap kill test (docs/PROJECT_ECS.md, Section 3): stage K1 from the
audits of the arm ``curves``, stage K2 from the validation WGA of every ECS arm against ``simclr`` and the
check that the reliance on the flagged units falls once the swap starts. When the study
holds the single-unit arms ``curves_unit_<k>``, the report holds the concept attribution of docs/PROJECT_CSA.md:
per unit the onset rank, the paired change of validation WGA against SimCLR with its verdict (shortcut, needed or
neutral), the change against the other unit arms of the same seed, the faithfulness check of the reliance curve
and the verdicts of hypotheses H1 to H3. Runs with ``concept_attribution_val.json``
(``cospro.cli.attribute_concepts``) add the attribution at inference: per unit the change of validation WGA when
the frozen encoder and probe see the evaluation images with that concept occluded, net of a size-matched control.
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
REFERENCE_ARMS = ("simclr", "curves_audit", "csa_audit")
# Audit-only arms (training equals SimCLR), the no-swap reference of the faithfulness check H3.
AUDIT_ARMS = ("csa_audit", "curves_audit")
UNIT_ARM = re.compile(r"curves_unit_\d+")
# Fail outcome of CSA: every effect stays neutral at this many seeds (docs/PROJECT_CSA.md, Section 4).
CSA_FAIL_SEEDS = 8
H2_MIN_SPEARMAN = 0.5
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
    baseline = next((audits[arm] for arm in AUDIT_ARMS if arm in audits), None)
    for unit, arm in sorted(unit_arms.items()):
        seeds = sorted(set(results[arm]) & set(reference))
        entries = [audits[arm][seed]["curves"] for seed in seeds if seed in audits.get(arm, {})]
        entries = [curves for curves in entries if curves.get("onset") is not None and unit < len(curves["units"])]
        if not seeds or not entries or not entries[0]["units"][unit]["audited"]:
            continue
        gains = [results[arm][seed]["wga"] - reference[seed]["wga"] for seed in seeds]
        typing = entries[0]["evaluation_only"].get("typing")
        verdict = effect_verdict(gains)
        rows.append({
            "verdict": verdict["verdict"], "wga_effect_se": verdict["se"], "gains": gains,
            "reliance": unit_share_change({seed: audits[arm][seed] for seed in seeds if seed in audits.get(arm, {})},
                                          unit, baseline),
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
    correlation = None if correlation is None or math.isnan(correlation) else correlation
    contrast = unit_contrast(results, {row["unit"]: f"curves_unit_{row['unit']}" for row in rows})
    for row in rows:
        row["relative"] = contrast["units"].get(row["unit"])
    return {"reference_arm": reference_arm, "units": rows, "onset_effect_spearman": correlation,
            "unit_contrast": {key: value for key, value in contrast.items() if key != "units"},
            "hypotheses": csa_hypotheses(rows, correlation)}


def effect_verdict(values: list[float]) -> dict:
    """Shortcut, needed or neutral (docs/PROJECT_CSA.md, Section 2).

    Beyond noise means the same sign on three quarters of the seeds and a mean of at least twice its standard
    error. A positive effect marks a shortcut (hiding the concept helps the worst group), a negative one a
    concept the encoder needs.
    """

    if len(values) < 2:
        return {"verdict": "neutral", "se": None}
    mean, se = statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))
    up, down = sum(value > 0 for value in values), sum(value < 0 for value in values)
    if mean > 0 and majority(up, len(values)) and mean >= 2 * se:
        return {"verdict": "shortcut", "se": se}
    if mean < 0 and majority(down, len(values)) and -mean >= 2 * se:
        return {"verdict": "needed", "se": se}
    return {"verdict": "neutral", "se": se}


def unit_share_change(entries: dict[int, dict], unit: int, baseline: dict[int, dict] | None = None) -> dict | None:
    """Faithfulness check H3 for one unit arm: its own share at the end of the early window and at the last audit.

    With an audit-only arm of the same seeds (``baseline``, no swap), the unit's last share is compared with that
    arm's last share of the unit, so the check removes how much the share drifts in plain SimCLR. Without it the
    reference is the unit arm's own share at the end of the early window.
    """

    changes = []
    for seed, entry in sorted(entries.items()):
        curves = entry["curves"]
        epochs, early = curves.get("epochs") or [], curves.get("early_epochs")
        if early is None or early not in epochs or max(epochs) <= early:
            continue
        reference = curves["share"][epochs.index(early)][unit]
        plain = (baseline or {}).get(seed)
        if plain is not None and max(epochs) in (plain["curves"].get("epochs") or []):
            reference = plain["curves"]["share"][plain["curves"]["epochs"].index(max(epochs))][unit]
        changes.append((curves["share"][epochs.index(early)][unit], curves["share"][-1][unit], reference))
    if not changes:
        return None
    fell = sum(after < reference for _, after, reference in changes)
    return {"seeds": len(changes), "fell": fell, "holds": majority(fell, len(changes)),
            "against": "audit-only arm" if baseline else "early window",
            "at_early_end": statistics.fmean(before for before, _, _ in changes),
            "at_last_audit": statistics.fmean(after for _, after, _ in changes),
            "reference": statistics.fmean(reference for _, _, reference in changes)}


def unit_contrast(results: dict[str, dict[int, dict]], unit_arms: dict[int, str]) -> dict:
    """WGA of every unit arm against the mean of all unit arms of the same seed, and a test that the unit matters.

    The reference is the mean of the unit arms on the seeds every unit arm finished, so one noisy reference run
    cannot shift every effect at once. ``f_statistic`` and ``p_value`` come from the two-way layout unit by seed
    without interaction: a large p value says the identity of the swapped unit explains no more WGA than the
    run-to-run noise does.
    """

    seeds = sorted(set.intersection(*(set(results[arm]) for arm in unit_arms.values()))) if unit_arms else []
    if len(unit_arms) < 2 or len(seeds) < 2:
        return {"seeds": seeds, "units": {}, "f_statistic": None, "p_value": None, "residual_sd": None}
    table = {unit: [results[arm][seed]["wga"] for seed in seeds] for unit, arm in unit_arms.items()}
    seed_mean = [statistics.fmean(table[unit][column] for unit in table) for column in range(len(seeds))]
    grand = statistics.fmean(seed_mean)
    units = {}
    for unit, values in table.items():
        differences = [value - mean for value, mean in zip(values, seed_mean)]
        units[unit] = {"mean": statistics.fmean(differences), "up": sum(value > 0 for value in differences),
                       "seeds": len(seeds), "verdict": effect_verdict(differences)["verdict"]}
    rows, columns = len(table), len(seeds)
    unit_means = {unit: statistics.fmean(values) for unit, values in table.items()}
    between = columns * sum((mean - grand) ** 2 for mean in unit_means.values())
    residual = sum((table[unit][column] - unit_means[unit] - seed_mean[column] + grand) ** 2
                   for unit in table for column in range(columns))
    freedom = (rows - 1) * (columns - 1)
    if residual <= 0:
        return {"seeds": seeds, "units": units, "f_statistic": None, "p_value": None, "residual_sd": 0.0}
    f_statistic = (between / (rows - 1)) / (residual / freedom)
    try:
        from scipy.stats import f as f_distribution

        p_value = float(f_distribution.sf(f_statistic, rows - 1, freedom))
    except ImportError:
        p_value = None
    return {"seeds": seeds, "units": units, "f_statistic": f_statistic, "p_value": p_value,
            "residual_sd": math.sqrt(residual / freedom)}


def csa_hypotheses(rows: list[dict], correlation: float | None) -> dict:
    """H1 to H3 of docs/PROJECT_CSA.md, Section 4, and the outcome of the training-time attribution.

    H3 holds when the swapped unit's own share falls on three quarters of the seeds for three quarters of the
    unit arms.
    """

    h1 = (any(row["type"] == "attribute" and row["verdict"] == "shortcut" for row in rows)
          and any(row["type"] == "class" and row["verdict"] == "needed" for row in rows))
    checks = [row["reliance"]["holds"] for row in rows if row.get("reliance")]
    h3 = bool(checks) and majority(sum(checks), len(checks))
    h2 = correlation is not None and correlation >= H2_MIN_SPEARMAN
    neutral = all(row["verdict"] == "neutral" for row in rows)
    fewest = min(len(row["seeds"]) for row in rows)
    if h1 and h3:
        outcome = "pass"
    elif neutral and fewest >= CSA_FAIL_SEEDS:
        outcome = "fail"
    else:
        outcome = "weak"
    return {"h1": h1, "h2": h2, "h3": h3, "h3_units_holding": sum(checks), "h3_units": len(checks),
            "all_neutral": neutral, "fewest_seeds": fewest, "outcome": outcome}


def inference_attributions(study_root: Path, split: str = "val") -> dict[str, dict[int, dict]]:
    """Per arm and seed the newest ``concept_attribution_<split>.json`` (cospro.cli.attribute_concepts)."""

    found: dict[str, dict[int, dict]] = {}
    for path in sorted(study_root.glob(f"seed_*/*/*/concept_attribution_{split}.json")):
        relative = path.relative_to(study_root).parts
        found.setdefault(relative[1], {})[int(relative[0].split("_")[1])] = json.loads(path.read_text(encoding="utf-8"))
    return found


def summarize_inference(arm: str, seeds: dict[int, dict], audits: dict[str, dict[int, dict]],
                        training: dict | None) -> dict:
    """Per unit the WGA change of the occlusion at inference over seeds, net of the control, with its verdict."""

    by_unit: dict[int, list[dict]] = {}
    for seed, record in sorted(seeds.items()):
        for row in record["units"]:
            by_unit.setdefault(row["unit"], []).append({"seed": seed, **row})
    onsets = {seed: entry["curves"]["onset"] for seed, entry in audits.get(arm, {}).items()
              if entry["curves"].get("onset") is not None}
    trained = {row["unit"]: row["wga_effect"] for row in (training or {}).get("units", [])}
    units = []
    for unit, rows in sorted(by_unit.items()):
        net = [row["net_change"]["wga"] for row in rows]
        verdict = effect_verdict(net)
        units.append({
            "unit": unit, "name": rows[0]["name"], "type": rows[0]["type"], "seeds": [row["seed"] for row in rows],
            "eval_presence": statistics.fmean(row["eval_presence"] for row in rows),
            "wga_change": statistics.fmean(row["change"]["wga"] for row in rows),
            "control_wga_change": statistics.fmean(row["control_change"]["wga"] for row in rows),
            "net_wga_change": statistics.fmean(net), "net_wga_se": verdict["se"],
            "net_up": sum(value > 0 for value in net),
            "net_acc_change": statistics.fmean(row["net_change"]["acc"] for row in rows),
            "embedding_shift": statistics.fmean(row["embedding_shift"] for row in rows),
            "control_shift": statistics.fmean(row["control_shift"] for row in rows),
            "net_group_change": [statistics.fmean(values)
                                 for values in zip(*(row["net_change"]["group_acc"] for row in rows))],
            "onset": statistics.fmean(onsets[seed][unit] for seed in onsets) if onsets else None,
            "training_effect": trained.get(unit), "verdict": verdict["verdict"],
        })

    def correlate(key: str) -> float | None:
        pairs = [(row[key], row["net_wga_change"]) for row in units if row[key] is not None]
        if len(pairs) < 3:
            return None
        value = spearman([first for first, _ in pairs], [second for _, second in pairs])
        return None if math.isnan(value) else value

    first = next(iter(seeds.values()))
    h1 = (any(row["type"] == "attribute" and row["verdict"] == "shortcut" for row in units)
          and any(row["type"] == "class" and row["verdict"] == "needed" for row in units))
    return {"arm": arm, "seeds": sorted(seeds), "group_names": first.get("group_names"),
            "clean_wga": statistics.fmean(record["clean"]["wga"] for record in seeds.values()),
            "units": units, "onset_spearman": correlate("onset"),
            "training_effect_spearman": correlate("training_effect"), "h1": h1}


def render_inference(result: dict) -> list[str]:
    def number(value, digits=2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    names = result["group_names"]
    lines = [f"## Concept attribution at inference (`{result['arm']}`, {len(result['seeds'])} seeds)", "",
             "Each row occludes one concept unit in the validation images that show it and classifies them with the "
             "frozen encoder and its probe. Changes are in points of validation WGA against the clean images; the "
             "control occludes a region of the same unit taken from another image, and the net change is the "
             f"occlusion minus the control. Mean clean WGA {result['clean_wga']:.1f}. Spearman of the net change "
             f"with the onset score {number(result['onset_spearman'])}, with the training-time effect "
             f"{number(result['training_effect_spearman'])}. **H1 at inference "
             f"{'holds' if result['h1'] else 'fails'}** (an attribute unit is a shortcut and a class unit is needed).",
             "", "| unit | words | type | eval presence | embedding shift (control) | WGA change | control | net | se | "
             "seeds up | verdict | net change per group |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in sorted(result["units"], key=lambda entry: -entry["net_wga_change"]):
        labels = names or [str(index) for index in range(len(row["net_group_change"]))]
        groups = ", ".join(f"{label.split('=')[-1]} {value:+.1f}" for label, value in zip(labels, row["net_group_change"]))
        lines.append(f"| {row['unit']} | {row['name']} | {row['type']} | {row['eval_presence']:.2f} | "
                     f"{row['embedding_shift']:.3f} ({row['control_shift']:.3f}) | {row['wga_change']:+.1f} | "
                     f"{row['control_wga_change']:+.1f} | {row['net_wga_change']:+.1f} | {number(row['net_wga_se'], 1)} | "
                     f"{row['net_up']}/{len(row['seeds'])} | {row['verdict']} | {groups} |")
    lines += ["", "Groups are (target, context) by the hidden labels."]
    return [*lines, ""]


def render_attribution(result: dict) -> list[str]:
    correlation = result["onset_effect_spearman"]
    hypotheses, contrast = result["hypotheses"], result["unit_contrast"]

    def holds(value: bool) -> str:
        return "holds" if value else "fails"

    def number(value, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = ["## Concept attribution", "",
             f"**Outcome: {hypotheses['outcome']}.** H1 (an attribute unit is a shortcut and a class unit is needed) "
             f"{holds(hypotheses['h1'])}; H2 (Spearman of onset and effect at least {H2_MIN_SPEARMAN:g}) "
             f"{holds(hypotheses['h2'])}; H3 (the swapped unit's own share falls) {holds(hypotheses['h3'])}, on "
             f"{hypotheses['h3_units_holding']} of {hypotheses['h3_units']} units. Fewest seeds per unit: "
             f"{hypotheses['fewest_seeds']}; the fail outcome needs every unit neutral at {CSA_FAIL_SEEDS} seeds.", "",
             f"Each row swaps one concept unit alone after the early window (arm `curves_unit_<k>`) and reports the "
             f"paired change of validation WGA against `{result['reference_arm']}`, in points. Spearman correlation of "
             f"onset score and WGA effect across units: {number(correlation, 2)}. Verdict: shortcut or needed when the "
             "sign holds on three quarters of the seeds and the mean is at least twice its standard error, neutral "
             "otherwise. The column *vs unit arms* compares each unit arm with the mean of all unit arms of the same "
             "seed.", "",
             f"Does the swapped unit matter? Unit-by-seed F test over {len(contrast['seeds'])} seeds: "
             f"F = {number(contrast['f_statistic'], 2)}, p = {number(contrast['p_value'], 2)}, run-to-run sd "
             f"{number(contrast['residual_sd'], 1)} points.", "",
             "| onset rank | unit | words | onset | area | hidden-label type | WGA effect | sd | seeds up | verdict | "
             "vs unit arms | accuracy effect | own share, epoch 50 to last, and its reference (seeds below it) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in result["units"]:
        relative, reliance = row.get("relative"), row.get("reliance")
        relative = "n/a" if relative is None else f"{relative['mean']:+.1f} ({relative['up']}/{relative['seeds']} up)"
        reliance = ("n/a" if reliance is None else f"{reliance['at_early_end']:.3f} to {reliance['at_last_audit']:.3f}, "
                    f"reference {reliance['reference']:.3f} ({reliance['fell']}/{reliance['seeds']})")
        lines.append(f"| {row['onset_rank']} | {row['unit']} | {row['name']} | {row['onset']:.3f} | {row['area']:.2f} | "
                     f"{row['type']} | {row['wga_effect']:+.1f} | {number(row['wga_effect_sd'], 1)} | "
                     f"{row['positive_seeds']}/{len(row['seeds'])} | {row['verdict']} | {relative} | "
                     f"{row['accuracy_effect']:+.1f} | {reliance} |")
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


def render(study: str, arms: dict[str, dict], attributed: dict | None = None, kill_test: dict | None = None,
           inference: list[dict] | None = None) -> str:
    def number(value) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = [f"# Concept audits of {study}", "",
             "Detection AUROC: how well the onset score separates attribute units from class units. Types come from "
             "hidden labels and are used for this report only.", ""]
    if kill_test is not None:
        lines += render_kill_test(kill_test)
    if attributed is not None:
        lines += render_attribution(attributed)
    for result in inference or []:
        lines += render_inference(result)
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
    inference = [summarize_inference(arm, seeds, audits, attributed)
                 for arm, seeds in sorted(inference_attributions(study_root).items())]
    directory = report(args.study, root=args.output_root)
    atomic_write_json(directory / "concept_curves.json", {"study": args.study, "arms": arms, "attribution": attributed,
                                                         "ecs_kill_test": kill_test, "inference_attribution": inference})
    summary = directory / "concept_curves.md"
    summary.write_text(render(args.study, arms, attributed, kill_test, inference), encoding="utf-8")
    print(f"[results] Concept audit summary: {summary}")
    return summary


if __name__ == "__main__":
    main()
