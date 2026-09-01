"""Print a small go/no-go report for CRP v2 frozen-audit JSON files.

This report is intentionally label-free.  Hidden-label graph checks belong in a
separate post-hoc analysis and must not influence the audit thresholds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _median(groups: list[dict], key: str) -> float:
    values = [float(group[key]) for group in groups if isinstance(group.get(key), (int, float))]
    return statistics.median(values) if values else float("nan")


def summarize(path: Path) -> int:
    report = json.loads(path.read_text(encoding="utf-8"))
    graph_version = int(report.get("graph_version", 0))
    groups = list(report.get("groups", []))
    selected = [group for group in groups if group.get("selected")]
    null_bound = []
    beats_random = []
    beats_shuffled = []
    for group in selected:
        score = float(group.get("score", 0.0))
        random_scores = [float(value) for value in group.get("random_subspace_scores", [])]
        shuffled_scores = [float(value) for value in group.get("shuffled_code_scores", [])]
        null_bound.append(score <= max(shuffled_scores, default=float("inf")) * (1.0 + 1e-6))
        beats_random.append(score > max(random_scores, default=float("-inf")))
        beats_shuffled.append(score > max(shuffled_scores, default=float("-inf")))

    degree = report.get("degree_stats", {})
    group_sizes = [len(group.get("concepts", [])) for group in groups]
    selected_sizes = [len(group.get("concepts", [])) for group in selected]
    print(f"file={path}")
    print(f"graph_version={graph_version} samples={report.get('sample_count')} candidates={len(groups)}")
    print(f"selected={len(selected)} concepts={[word for group in selected for word in group.get('concepts', [])]}")
    print(
        "grouping="
        f"singletons={sum(size == 1 for size in group_sizes)}/{len(group_sizes)} "
        f"median_size={statistics.median(group_sizes) if group_sizes else float('nan'):.1f} "
        f"selected_median_size={statistics.median(selected_sizes) if selected_sizes else float('nan'):.1f}"
    )
    print(
        "projection="
        f"median_jaccard={_median(groups, 'mean_jaccard_at_k'):.4f} "
        f"median_top1_turnover={_median(groups, 'top1_neighbor_turnover'):.4f}"
    )
    print(
        "selected_quality="
        f"median_coverage={_median(selected, 'coverage'):.4f} "
        f"median_semantic_agreement={_median(selected, 'semantic_agreement'):.4f} "
        f"median_alignment={_median(selected, 'activation_gain_alignment'):.4f}"
    )
    print(
        "graph="
        f"coverage={float(degree.get('coverage', 0.0)):.4f} "
        f"indegree_gini={float(degree.get('indegree_gini', 0.0)):.4f} "
        f"effective_donor_count={float(degree.get('effective_donor_count', 0.0)):.1f}"
    )
    if selected:
        print(
            "nulls="
            f"beats_random={sum(beats_random)}/{len(selected)} "
            f"beats_shuffled={sum(beats_shuffled)}/{len(selected)} "
            f"null_bound={sum(null_bound)}/{len(selected)}"
        )

    stale_artifact = graph_version < 2
    weak_projection = (
        _median(groups, "mean_jaccard_at_k") > 0.90
        and _median(groups, "top1_neighbor_turnover") < 0.10
    )
    null_failure = not selected or any(null_bound) or sum(beats_shuffled) < len(selected)
    if stale_artifact:
        print("GO_NO_GO=NO_GO artifact predates the activation/gain alignment null fix; rerun the audit")
        return 1
    if weak_projection:
        print("GO_NO_GO=NO_GO projection is too weak by the frozen-audit criterion")
        return 1
    if null_failure:
        print("GO_NO_GO=NO_GO selected groups do not beat the shuffled-code null")
        return 1
    print("GO_NO_GO=PROVISIONAL_GO run the one-seed CRP control matrix; do not scale to multi-seed yet")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=Path)
    args = parser.parse_args()
    return max(summarize(path) for path in args.json_files)


if __name__ == "__main__":
    raise SystemExit(main())
