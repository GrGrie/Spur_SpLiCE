"""Portable JSON storage for CRP teacher graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


GRAPH_TENSOR_DTYPES = {
    "neighbor_indices": torch.long,
    "weights": torch.float32,
    "edge_confidences": torch.float32,
    "group_ids": torch.long,
    "intervention_gains": torch.float32,
    "anchor_confidence": torch.float32,
    "confidence": torch.float32,
    "edge_source": torch.long,
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_graph_json(graph: dict, path: str | Path) -> Path:
    """Atomically write a complete, human-inspectable teacher graph."""

    output_path = Path(path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("Teacher graphs must use a .json file extension.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_ready(graph), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def load_graph_json(path: str | Path) -> dict:
    """Load a teacher graph and restore tensors used by training."""

    graph_path = Path(path)
    if graph_path.suffix.lower() != ".json":
        raise ValueError("Teacher graphs must use a .json file extension.")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict):
        raise ValueError("Teacher graph JSON must contain an object at the top level.")
    for key, dtype in GRAPH_TENSOR_DTYPES.items():
        if key in graph:
            graph[key] = torch.as_tensor(graph[key], dtype=dtype)
    return graph


def graph_fingerprint(path: str | Path) -> str:
    """Return a compact content identity used to protect exact resume semantics."""

    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
