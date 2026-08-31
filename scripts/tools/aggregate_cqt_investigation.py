"""Aggregate a CQT q/dinoq graph sweep into one self-contained JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch


def _csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def _canonical_number(value: str) -> str:
    return format(float(value), ".15g")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _topology_sha256(graph: dict) -> str:
    digest = hashlib.sha256()
    for key in ("neighbor_indices", "weights"):
        tensor = graph[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_result(graph_path: Path, q: str, dinoq: str) -> dict:
    summary_path = graph_path.with_suffix(".json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    graph = torch.load(graph_path, map_location="cpu", weights_only=True)
    if graph.get("artifact") != "splice_cqt_v1_teacher_graph":
        raise ValueError(f"Unexpected artifact in {graph_path}: {graph.get('artifact')}")
    config = graph.get("config", {})
    if summary.get("artifact") != graph["artifact"] or summary.get("config") != config:
        raise ValueError(f"Graph and summary disagree for {graph_path}")
    if float(config.get("min_quotient_efficacy")) != float(q):
        raise ValueError(f"Unexpected q in {graph_path}: {config.get('min_quotient_efficacy')}")
    if float(config.get("dino_damage_quantile")) != float(dinoq):
        raise ValueError(
            f"Unexpected dinoq in {graph_path}: {config.get('dino_damage_quantile')}"
        )
    edge_count = int(graph.get("degree_stats", {}).get("edge_count", 0))
    return {
        "q": float(q),
        "dinoq": float(dinoq),
        "graph_path": graph_path.as_posix(),
        "graph_sha256": _file_sha256(graph_path),
        "topology_sha256": _topology_sha256(graph),
        "graph_empty": edge_count == 0,
        **summary,
    }


def aggregate(
    root: Path,
    output: Path,
    dataset: str,
    seed: int,
    q_values: list[str],
    dinoq_values: list[str],
    allow_missing: bool,
) -> dict:
    results, missing = [], []
    for q in q_values:
        for dinoq in dinoq_values:
            variant = f"q{_canonical_number(q)}_dinoq{_canonical_number(dinoq)}"
            graph_path = root / variant / f"teacher_graph_seed{seed}.pt"
            if not graph_path.is_file() or not graph_path.with_suffix(".json").is_file():
                missing.append({"q": float(q), "dinoq": float(dinoq)})
                continue
            results.append(_load_result(graph_path, q, dinoq))

    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing {len(missing)} sweep outputs; first missing: {missing[0]}")

    by_topology: dict[str, list[dict[str, float]]] = defaultdict(list)
    for result in results:
        by_topology[result["topology_sha256"]].append(
            {"q": result["q"], "dinoq": result["dinoq"]}
        )
    duplicates = [
        {"topology_sha256": digest, "configurations": configurations}
        for digest, configurations in by_topology.items()
        if len(configurations) > 1
    ]

    payload = {
        "artifact": "splice_cqt_investigation_v1",
        "dataset": dataset,
        "seed": seed,
        "q_values": [float(value) for value in q_values],
        "dinoq_values": [float(value) for value in dinoq_values],
        "expected_count": len(q_values) * len(dinoq_values),
        "completed_count": len(results),
        "complete": not missing,
        "missing": missing,
        "empty_graph_count": sum(result["graph_empty"] for result in results),
        "unique_topology_count": len(by_topology),
        "duplicate_topology_groups": duplicates,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--q-values", required=True, type=_csv_values)
    parser.add_argument("--dinoq-values", required=True, type=_csv_values)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    payload = aggregate(
        args.root,
        args.output,
        args.dataset,
        args.seed,
        args.q_values,
        args.dinoq_values,
        args.allow_missing,
    )
    print(
        f"[INFO] Wrote {payload['completed_count']}/{payload['expected_count']} "
        f"CQT investigations to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
