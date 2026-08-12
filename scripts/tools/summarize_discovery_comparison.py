"""Collect the concept-discovery comparison into one table.

Reads the per-method JSON files written by ``waterbirds_SpLiCE_discovery_array.sbatch``
and reports, for every discovery signal, which concepts it selected and how much
zeroing them changes worst-group accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHOD_LABELS = {
    "oracle": "conditional_group (uses group metadata)",
    "errcontrast": "error_contrast (class labels only)",
    "gradprobe": "gradient_probe (published estimator)",
    "paper": "fixed list (SpLiCE appendix)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize the SpLiCE concept-discovery comparison")
    parser.add_argument("--dataset", default="waterbirds")
    parser.add_argument("--results_dir", default="outputs")
    parser.add_argument(
        "--methods",
        default="oracle,errcontrast,gradprobe,paper",
        help="Comma-separated run labels matching the sbatch array tasks.",
    )
    parser.add_argument("--out_path", default="")
    return parser.parse_args()


def load_result(results_dir: Path, dataset: str, method: str) -> dict | None:
    path = results_dir / f"{dataset}_discovery_{method}_cbm.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]

    rows = []
    for method in methods:
        payload = load_result(results_dir, args.dataset, method)
        if payload is None:
            print(f"[WARN] Missing results for {method!r}; has that array task finished?")
            continue
        baseline = payload["baseline"]["worst_group_accuracy"] * 100
        intervened = payload["probe_intervention"]["worst_group_accuracy"] * 100
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "eval_split": payload["eval_split"],
                "concepts": [entry["concept"] for entry in payload["intervention_concepts"]],
                "baseline_wg": baseline,
                "intervention_wg": intervened,
                "delta_wg": intervened - baseline,
                "baseline_avg": payload["baseline"]["average_accuracy"] * 100,
                "intervention_avg": payload["probe_intervention"]["average_accuracy"] * 100,
            }
        )

    if not rows:
        raise SystemExit("No results found. Run the discovery array first.")

    width = max(len(row["label"]) for row in rows)
    print()
    print("=" * (width + 46))
    print(f"{'discovery signal':<{width}}  {'WG base':>8} {'WG abl.':>8} {'delta':>8}")
    print("-" * (width + 46))
    for row in sorted(rows, key=lambda item: item["delta_wg"], reverse=True):
        print(
            f"{row['label']:<{width}}  {row['baseline_wg']:8.2f} "
            f"{row['intervention_wg']:8.2f} {row['delta_wg']:+8.2f}"
        )
    print("=" * (width + 46))
    print(f"eval split: {rows[0]['eval_split']}")
    print()
    for row in rows:
        print(f"{row['label']}:")
        print(f"    concepts: {', '.join(row['concepts'])}")
    print()
    print("A group-free signal that matches the oracle row is the result worth reporting.")

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote comparison table to {out_path}")


if __name__ == "__main__":
    main()
