"""Validate and summarize the fixed CRP replication runs.

The input is deliberately a JSON manifest rather than a set of command-line
overrides.  This keeps the comparison roots, arms, weights, seeds and frozen
artifact fingerprints auditable and makes it impossible for the summary to
silently select a different run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from splice.graph_io import graph_fingerprint


REQUIRED_PROTOCOLS = {"crp_controls_cluster_v1", "crp_controls_followup_v1"}
COMMON_FIELDS = (
    "dataset",
    "epochs",
    "batch_size",
    "model",
    "learning_rate",
    "train_set_linear_layer",
    "probe_l2",
    "probe_tolerance",
    "probe_max_epochs",
    "linear_probe_mode",
    "linear_probe_freq",
    "graph_seed",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return left == right
    return math.isclose(left_number, right_number, rel_tol=tolerance, abs_tol=tolerance)


def _resolve_result(root: Path, record: dict[str, Any], ssl_epoch: int) -> Path | None:
    declared = Path(str(record.get("result", "")))
    if declared.is_file():
        return declared
    seed = record.get("seed")
    arm = record.get("arm")
    if seed is None or arm is None:
        return None
    candidates = sorted((root / f"seed{seed}" / str(arm) / "training").glob(
        f"*/probe_features_epoch_{ssl_epoch}_*_val.json"
    ))
    return candidates[-1] if len(candidates) == 1 else None


def _command_values(root: Path, seed: int, arm: str) -> dict[str, str] | None:
    command_path = root / f"seed{seed}" / arm / "command.json"
    if not command_path.is_file():
        return None
    payload = _read_json(command_path)
    command = payload.get("command", payload if isinstance(payload, list) else None)
    if not isinstance(command, list):
        return None
    values: dict[str, str] = {}
    index = 0
    while index < len(command):
        item = str(command[index])
        if item.startswith("--") and index + 1 < len(command):
            values[item[2:]] = str(command[index + 1])
            index += 2
        else:
            index += 1
    return values


def _condition(arm: str, weight: float | None) -> str:
    if arm in {"simclr", "crp_sampler_only", "raw_clip_sampler_only"}:
        return arm
    if weight is None:
        return arm
    return f"{arm}_lambda{weight:g}"


def _validate_root(
    spec: dict[str, Any],
    manifest: dict[str, Any],
    expected_fingerprints: dict[str, str],
    reference_fields: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    root = Path(spec["root"])
    name = str(spec.get("name", root))
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    experiment_path = root / "experiment.json"
    if not experiment_path.is_file():
        return [], [f"{name}: missing experiment.json: {experiment_path}"], reference_fields
    experiment = _read_json(experiment_path)
    protocol = experiment.get("protocol")
    if protocol not in REQUIRED_PROTOCOLS:
        problems.append(f"{name}: unsupported protocol {protocol!r}")
    if reference_fields is None:
        reference_fields = {field: experiment.get(field) for field in COMMON_FIELDS}
    else:
        for field in COMMON_FIELDS:
            actual = experiment.get(field)
            expected = reference_fields.get(field)
            matches = (
                _same_number(actual, expected)
                if isinstance(expected, (int, float))
                else actual == expected
            )
            if not matches:
                problems.append(f"{name}: protocol field {field}={actual!r}, expected {expected!r}")

    expected_arms = list(spec["arms"])
    expected_seeds = [int(seed) for seed in spec["seeds"]]
    expected_weights = spec.get("weights", {})
    required_graphs = set()
    if any(arm in {"crp_sampler_only", "splice_crp_kl"} for arm in expected_arms):
        required_graphs.add("crp")
    if any(arm.startswith("raw_clip_") for arm in expected_arms):
        required_graphs.add("raw_clip")

    cache_identity_path = root / "cache_identity.json"
    graph_identity_path = root / "graph_identity.json"
    if not cache_identity_path.is_file():
        problems.append(f"{name}: missing cache_identity.json")
    else:
        cache_identity = _read_json(cache_identity_path)
        if cache_identity.get("content_id") != expected_fingerprints.get("cache"):
            problems.append(
                f"{name}: cache fingerprint {cache_identity.get('content_id')!r} "
                f"!= {expected_fingerprints.get('cache')!r}"
            )
        cache_path = Path(str(cache_identity.get("path", "")))
        if not cache_path.is_file():
            cache_path = root / cache_path.name
        if cache_path.is_file() and graph_fingerprint(cache_path) != expected_fingerprints.get("cache"):
            problems.append(f"{name}: cache content fingerprint changed: {cache_path}")
    graph_identity: dict[str, str] = {}
    if not graph_identity_path.is_file():
        problems.append(f"{name}: missing graph_identity.json")
    else:
        graph_identity = _read_json(graph_identity_path)
        for graph_name in sorted(required_graphs):
            if graph_identity.get(graph_name) != expected_fingerprints.get(graph_name):
                problems.append(
                    f"{name}: {graph_name} fingerprint {graph_identity.get(graph_name)!r} "
                    f"!= {expected_fingerprints.get(graph_name)!r}"
                )
            graph_path = root / "graphs" / f"{graph_name}_graph.json"
            if graph_path.is_file() and graph_fingerprint(graph_path) != expected_fingerprints.get(graph_name):
                problems.append(f"{name}: {graph_name} content fingerprint changed: {graph_path}")

    expected_run_fingerprints = {
        "cache_fingerprint": expected_fingerprints.get("cache"),
        "graph_fingerprints": {
            name: expected_fingerprints[name] for name in sorted(required_graphs)
        },
    }
    for seed in expected_seeds:
        for arm in expected_arms:
            completed_path = root / f"seed{seed}" / arm / "completed.json"
            prefix = f"{name}: seed={seed}, arm={arm}"
            if not completed_path.is_file():
                problems.append(f"{prefix}: missing completed.json")
                continue
            try:
                record = _read_json(completed_path)
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{prefix}: invalid completed.json ({exc})")
                continue
            if record.get("status") != "complete":
                problems.append(f"{prefix}: status is {record.get('status')!r}")
                continue
            identity = record.get("run_identity", {})
            if identity.get("cache_fingerprint") != expected_run_fingerprints["cache_fingerprint"]:
                problems.append(f"{prefix}: completed cache fingerprint mismatch")
            actual_graphs = identity.get("graph_fingerprints", {})
            if any(
                actual_graphs.get(name) != fingerprint
                for name, fingerprint in expected_run_fingerprints["graph_fingerprints"].items()
            ):
                problems.append(f"{prefix}: completed graph fingerprint mismatch")
            command_values = _command_values(root, seed, arm)
            if command_values is None:
                problems.append(f"{prefix}: missing or invalid command.json")
                continue
            actual_weight = _number(command_values.get("splice_weight"))
            expected_weight = expected_weights.get(arm)
            if expected_weight is not None and not _same_number(actual_weight, expected_weight):
                problems.append(f"{prefix}: splice_weight={actual_weight!r}, expected {expected_weight!r}")
            result_path = _resolve_result(root, record, int(manifest["ssl_epoch"]))
            if result_path is None:
                problems.append(f"{prefix}: final probe result is missing")
                continue
            try:
                result = _read_json(result_path)
                metrics = result["metrics"]
                avg = float(metrics["Average over last 10 linear val acc"])
                wga = float(metrics["Average over last 10 linear val worst-group acc"])
                best_group = float(metrics["Average over last 10 linear val best-group acc"])
                converged = bool(metrics["Probe converged"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"{prefix}: invalid probe result {result_path} ({exc})")
                continue
            declared_result = str(record.get("result", ""))
            declared_path = Path(declared_result)
            if declared_result and not declared_path.is_absolute():
                declared_path = root / declared_path
            if declared_result and declared_path.resolve() != result_path.resolve():
                problems.append(f"{prefix}: completed.json result does not match final probe path")
            if record.get("seed") is not None and int(record["seed"]) != seed:
                problems.append(f"{prefix}: completed.json seed disagrees with manifest")
            if record.get("arm") is not None and str(record["arm"]) != arm:
                problems.append(f"{prefix}: completed.json arm disagrees with manifest")
            if record.get("avg_acc_last10") is not None and not _same_number(record["avg_acc_last10"], avg, 1e-6):
                problems.append(f"{prefix}: completed/final average accuracy disagreement")
            if record.get("wga_last10") is not None and not _same_number(record["wga_last10"], wga, 1e-6):
                problems.append(f"{prefix}: completed/final WGA disagreement")
            if not converged:
                problems.append(f"{prefix}: final logistic probe did not converge")
            rows.append({
                "source": name,
                "root": str(root),
                "seed": seed,
                "arm": arm,
                "weight": actual_weight,
                "condition": _condition(arm, actual_weight),
                "avg_acc_last10": avg,
                "wga_last10": wga,
                "best_group_last10": best_group,
                "result": str(result_path),
            })
    return rows, problems, reference_fields


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    baseline = {(row["seed"]): row for row in by_condition.get("simclr", [])}
    summary: dict[str, Any] = {}
    for condition, selected in sorted(by_condition.items()):
        summary[condition] = {"seeds": sorted(row["seed"] for row in selected)}
        for metric in ("avg_acc_last10", "wga_last10", "best_group_last10"):
            values = [row[metric] for row in selected]
            deltas = [
                row[metric] - baseline[row["seed"]][metric]
                for row in selected
                if row["seed"] in baseline
            ]
            summary[condition][metric] = {
                "mean": statistics.mean(values) if values else None,
                "std_across_seeds": statistics.stdev(values) if len(values) > 1 else None,
                "paired_seeds_vs_simclr": [row["seed"] for row in selected if row["seed"] in baseline],
                "paired_deltas_vs_simclr": deltas,
                "mean_delta_vs_simclr": statistics.mean(deltas) if deltas else None,
            }
    return summary


def _paired_metric(
    rows: list[dict[str, Any]],
    target: str,
    control: str,
    metric: str,
    seeds: set[int] | None = None,
) -> tuple[list[int], list[float]]:
    target_rows = {row["seed"]: row for row in rows if row["condition"] == target}
    control_rows = {row["seed"]: row for row in rows if row["condition"] == control}
    overlap = set(target_rows) & set(control_rows)
    if seeds is not None:
        overlap &= seeds
    ordered = sorted(overlap)
    return ordered, [target_rows[seed][metric] - control_rows[seed][metric] for seed in ordered]


def _gate(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    """Report every old gate contrast, preserving missing-vs-negative semantics."""

    target = "splice_crp_kl_lambda0.5"
    replication_seeds = {3, 4}

    def contrast(control: str, expected_seeds: set[int] | None = None) -> dict[str, Any]:
        available, avg = _paired_metric(rows, target, control, "avg_acc_last10", expected_seeds)
        _, wga = _paired_metric(rows, target, control, "wga_last10", expected_seeds)
        expected = sorted(expected_seeds) if expected_seeds is not None else available
        complete = available == expected
        avg_mean = statistics.mean(avg) if avg else None
        wga_mean = statistics.mean(wga) if wga else None
        positive = avg_mean is not None and wga_mean is not None and avg_mean > 0 and wga_mean > 0
        return {
            "available_seeds": available,
            "expected_seeds": expected,
            "complete": complete,
            "status": "NOT_EVALUABLE" if not complete else ("PASS_EFFECT" if positive else "FAIL_EFFECT"),
            "mean_delta_avg_acc_pp": avg_mean,
            "mean_delta_wga_pp": wga_mean,
            "avg_acc_positive": avg_mean is not None and avg_mean > 0,
            "wga_positive": wga_mean is not None and wga_mean > 0,
        }

    controls = ("simclr", "crp_sampler_only", "raw_clip_kl_lambda0.2", "raw_clip_kl_lambda0.5", "raw_clip_kl_lambda2")
    pooled_seeds = {1, 2, 3, 4}
    pooled = {control: contrast(control, pooled_seeds) for control in controls}
    replication = {control: contrast(control, replication_seeds) for control in controls}
    simclr = pooled["simclr"]
    return {
        "status": "reported_only",
        "target": target,
        "replication_seeds": sorted(replication_seeds),
        "against_simclr": {
            **simclr,
            "avg_acc_at_least_1pp": simclr["mean_delta_avg_acc_pp"] is not None and simclr["mean_delta_avg_acc_pp"] >= 1.0,
            "wga_at_least_2pp": simclr["mean_delta_wga_pp"] is not None and simclr["mean_delta_wga_pp"] >= 2.0,
        },
        "comparisons": replication,
        "full_pooled_contrasts": pooled,
        "scientific_outcomes": {
            "negative_effects": [control for control, value in pooled.items() if value["status"] == "FAIL_EFFECT"],
            "not_evaluable": [control for control, value in pooled.items() if value["status"] == "NOT_EVALUABLE"],
        },
        "note": "Descriptive only: missing comparisons are NOT_EVALUABLE, negative complete comparisons are scientific outcomes, and no winner is selected.",
    }


def summarize(config_path: Path) -> tuple[Path, bool]:
    manifest = _read_json(config_path)
    expected = manifest["expected_fingerprints"]
    output = Path(manifest["output"])
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    reference_fields: dict[str, Any] | None = None
    seen_keys: set[tuple[int, str]] = set()
    for spec in manifest["experiments"]:
        rows, root_problems, reference_fields = _validate_root(
            spec, manifest, expected, reference_fields
        )
        problems.extend(root_problems)
        for row in rows:
            key = (row["seed"], row["condition"])
            if key in seen_keys:
                problems.append(f"duplicate row: {key}")
            seen_keys.add(key)
            all_rows.append(row)

    # Every new run must be present exactly once.  This catches a malformed
    # summary manifest even when old controls are complete.
    for spec in manifest["experiments"]:
        if not spec.get("required", True):
            continue
        expected_count = len(spec["seeds"]) * len(spec["arms"])
        actual_count = sum(row["source"] == spec.get("name", spec["root"]) for row in all_rows)
        if actual_count != expected_count:
            problems.append(
                f"{spec.get('name', spec['root'])}: expected {expected_count} valid rows, got {actual_count}"
            )

    summary = _condition_summary(all_rows)
    report = {
        "status": "complete" if not problems else "incomplete",
        "protocol": manifest.get("protocol"),
        "dataset": manifest.get("dataset"),
        "ssl_epoch": manifest.get("ssl_epoch"),
        "expected_fingerprints": expected,
        "problems": problems,
        "validation_failures": problems,
        "conditions": summary,
        "gate": _gate(all_rows, summary),
        "note": "No epoch or arm is selected automatically; all valid rows are retained.",
    }
    rows_path = output / "replication_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "source", "root", "seed", "arm", "weight", "condition",
            "avg_acc_last10", "wga_last10", "best_group_last10", "result",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (output / "replication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "replication_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "rows": len(all_rows), "problems": problems}, indent=2))
    return output, not problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    _, complete = summarize(args.config)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
