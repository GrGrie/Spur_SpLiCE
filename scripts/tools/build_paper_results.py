"""Build the compact, checked paper registry from the packaged artifact tree."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from splice.artifacts import PROJECT_ROOT, resolve_output_root


ARMS = (
    "simclr",
    "crp_sampler_only",
    "raw_clip_sampler_only",
    "raw_clip_kl",
    "splice_crp_kl",
)
SEEDS = (1, 2, 3, 4)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty distribution.")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": _quantile(values, 0.0),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "q75": _quantile(values, 0.75),
        "q95": _quantile(values, 0.95),
        "max": _quantile(values, 1.0),
        "mean": statistics.fmean(values),
    }


def _core_registry(root: Path) -> tuple[list[dict], dict, dict]:
    reports = root / "reports" / "paper"
    results = _load(reports / "test_results.json")
    summary = _load(reports / "test_summary.json")
    lock_record = _load(reports / "final_test_lock.json")
    lock = lock_record["rows"]
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}

    result_by_key = {(int(row["seed"]), str(row["arm"])): row for row in results}
    lock_by_key = {(int(row["seed"]), str(row["arm"])): row for row in lock}
    if len(result_by_key) != len(results) or set(result_by_key) != expected:
        raise ValueError("Final-test results must contain exactly one row for every seed/arm.")
    if len(lock_by_key) != len(lock) or set(lock_by_key) != expected:
        raise ValueError("Final-test lock must contain exactly one row for every seed/arm.")

    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            result = result_by_key[(seed, arm)]
            locked = lock_by_key[(seed, arm)]
            status_paths = list(
                (root / "seeds" / f"seed_{seed:02d}" / "paper" / "core" / arm).glob(
                    "training/*/run_status.json"
                )
            )
            complete = [path for path in status_paths if _load(path).get("status") == "complete"]
            if len(complete) != 1:
                raise ValueError(f"Expected one complete core execution for seed={seed}, arm={arm}.")
            status = _load(complete[0])
            wandb = status.get("run_identity", {}).get("wandb", {})
            probe_paths = list(
                (root / "seeds" / f"seed_{seed:02d}" / "paper" / "test" / arm).glob("*.json")
            )
            if len(probe_paths) != 1:
                raise ValueError(f"Expected one final-test probe JSON for seed={seed}, arm={arm}.")
            probe = _load(probe_paths[0])
            convergence = probe.get("convergence")
            if not isinstance(convergence, dict) or convergence.get("converged") is not True:
                raise ValueError(f"Final-test probe did not converge for seed={seed}, arm={arm}.")
            rows.append(
                {
                    "execution_id": f"wandb:{wandb.get('id')}" if wandb.get("id") else f"core:{seed}:{arm}",
                    "phase": "final_test",
                    "seed": seed,
                    "arm": arm,
                    "ssl_epoch": 500,
                    "avg": result["avg"],
                    "wga": result["wga"],
                    "groups": result["groups"],
                    "checkpoint_sha256": locked["sha256"],
                    "args_sha256": locked["args_sha256"],
                    "wandb": wandb or None,
                    "probe_convergence": convergence,
                }
            )

    if len({row["execution_id"] for row in rows}) != len(rows):
        raise ValueError("Core execution identities are not unique.")
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        for metric in ("avg", "wga"):
            values = [float(row[metric]) for row in arm_rows]
            expected_mean = float(summary["summary"][arm][metric]["mean"])
            expected_sd = float(summary["summary"][arm][metric]["sd"])
            if not math.isclose(statistics.fmean(values), expected_mean, abs_tol=1e-8):
                raise ValueError(f"Stored {arm}/{metric} mean does not match test rows.")
            if not math.isclose(statistics.stdev(values), expected_sd, abs_tol=1e-8):
                raise ValueError(f"Stored {arm}/{metric} SD does not match test rows.")
    lock_metadata = {
        "source_git_revision": lock_record["git_revision"],
        "frozen_artifact_sha256": lock_record["graphs_sha256"],
        "probe_protocol": lock_record["probe"],
        "reporting_rule": lock_record["rule"],
    }
    return rows, summary, lock_metadata


def _mechanism(root: Path) -> dict:
    graph = _load(root / "shared" / "waterbirds" / "graphs" / "crp_graph.json")
    groups = graph["groups"]
    if len(groups) != 504:
        raise ValueError(f"Expected the canonical 504-group audit, found {len(groups)} groups.")
    passing = [
        group for group in groups
        if group["coverage"] >= graph["config"]["min_coverage"]
        and group["score"] > group["null_threshold"]
    ]
    selected = [group for group in groups if group["selected"]]
    null_quantile = float(graph["config"]["null_quantile"])
    distributions = {
        field: _distribution([float(group[field]) for group in groups])
        for field in (
            "score",
            "null_threshold",
            "null_excess_score",
            "coverage",
            "robust_positive_gain",
            "semantic_agreement",
        )
    }
    random_scores = [float(value) for group in groups for value in group["random_subspace_scores"]]
    shuffled_scores = [float(value) for group in groups for value in group["shuffled_code_scores"]]
    group_sizes: dict[str, int] = {}
    for group in groups:
        key = str(len(group["concepts"]))
        group_sizes[key] = group_sizes.get(key, 0) + 1
    return {
        "group_count": len(groups),
        "passing_per_group_null_gate": len(passing),
        "selected_after_cap": len(selected),
        "selected_group_ids": [int(group["group_id"]) for group in selected],
        "concepts_per_group_counts": group_sizes,
        "real_group_distributions": distributions,
        "null_score_distributions": {
            "random_subspace": _distribution(random_scores),
            "shuffled_codes": _distribution(shuffled_scores),
        },
        "chance_reference": {
            "per_trial_upper_tail_rate": 1.0 - null_quantile,
            "nominal_expected_exceedances_across_504_if_real_groups_were_exchangeable_nulls": (
                len(groups) * (1.0 - null_quantile)
            ),
            "observed_real_groups_passing": len(passing),
            "caveat": (
                "This is a descriptive nominal reference, not FDR control: thresholds are fitted "
                "per group and the audited real groups are neither independent nor null draws."
            ),
        },
        "use_boundary": "Post-lock mechanism analysis; it must not change the frozen graph or final-test selection.",
    }


def build(root: Path) -> dict:
    rows, final_summary, lock_metadata = _core_registry(root)
    direct = _load(root / "reports" / "followups_corrected" / "direct_transfer" / "summary.json")
    graph_ablation = _load(root / "reports" / "followups_corrected" / "graph_ablation" / "summary.json")
    panels = _load(root / "reports" / "followups_corrected" / "visual" / "concept_panels.json")
    try:
        displayed_root = root.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        displayed_root = str(root.resolve())
    return {
        "schema_version": 2,
        "scope": "Waterbirds scientific-article evidence registry",
        "done_definition": (
            "DONE means a predeclared split/protocol, unique execution identity, linked endpoint metrics, "
            "completed/converged status, and explicit limitations. It does not require a new report for "
            "every implementation detail or repeated training without a claim-changing reason."
        ),
        "evidence_boundaries": [
            "The lock has no internal timestamp; no absolute lock date is asserted.",
            "No standalone AMP-vs-FP32/RNG/BN/update equivalence report was found; none is asserted.",
            "Historical paths are provenance strings and need not resolve after packaging.",
        ],
        "sources": {
            "artifact_root": displayed_root,
            "final_test": "reports/paper/{final_test_lock,test_results,test_summary}.json",
            "mechanism": "shared/waterbirds/graphs/crp_graph.json",
            "direct_transfer": "reports/followups_corrected/direct_transfer/summary.json",
            "graph_ablation": "reports/followups_corrected/graph_ablation/summary.json",
            "panels": "reports/followups_corrected/visual/concept_panels.json",
        },
        "final_test": {
            "dataset": "waterbirds",
            "split": "test",
            "ssl_epoch": 500,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "rows": rows,
            **lock_metadata,
            **final_summary,
        },
        "development_followups": {
            "split": "validation",
            "direct_transfer": {
                key: direct[key]
                for key in (
                    "passed",
                    "winner_selection",
                    "rows",
                    "paired_deltas_vs_matched_simclr",
                )
            },
            "semantic_graph_ablation": {
                key: graph_ablation[key]
                for key in (
                    "passed",
                    "comparison",
                    "rows",
                    "paired_deltas_crp_minus_semantic",
                )
            },
        },
        "mechanism": _mechanism(root),
        "qualitative_panels": {
            "selection_rule": panels["selection_rule"],
            "selection_is_label_free": panels["selection_is_label_free"],
            "group_count": len(panels["groups"]),
            "pair_count": sum(len(group["pairs"]) for group in panels["groups"]),
            "interpretation_caveat": panels["interpretation_caveat"],
            "format_limit": (
                "The packaged panels show selected pair comparisons, not a complete "
                "anchor-to-raw-nearest-to-projected-nearest triplet for every example."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper_results.json")
    args = parser.parse_args()
    root = resolve_output_root(args.artifact_root)
    payload = build(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"[INFO] Wrote checked paper registry to {args.output}")


if __name__ == "__main__":
    main()
