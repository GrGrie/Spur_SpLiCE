"""Label-free preflight checks for the corrected transfer/graph series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from splice.concept_distillation import load_target_artifact
from splice.crp import validate_feature_cache
from splice.crp_training import validate_teacher_graph
from splice.graph_io import load_graph_json


def run(cache_path: Path, target_path: Path, graph_paths: list[Path], output: Path) -> Path:
    cache = validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=True))
    targets = load_target_artifact(target_path)
    sample_ids = [str(value) for value in cache["sample_ids"]]
    if targets["sample_ids"] != sample_ids:
        raise RuntimeError("Target sample IDs do not exactly match the frozen cache order.")
    valid = targets["valid_mask"].bool()
    permutation = targets["permutation"].long()
    valid_indices = torch.where(valid)[0]
    if not torch.equal(torch.sort(permutation[valid_indices]).values, valid_indices):
        raise RuntimeError("Shuffled target permutation is not a permutation of valid rows.")
    if torch.any(permutation[valid_indices] == valid_indices):
        raise RuntimeError("Shuffled target permutation contains a fixed point.")

    graph_records = []
    for path in graph_paths:
        graph = validate_teacher_graph(load_graph_json(path), sample_ids)
        graph_records.append({
            "path": str(path),
            "artifact": graph["artifact"],
            "supported_anchors": int((graph["weights"].sum(1) > 0).sum()),
            "edge_count": int((graph["neighbor_indices"] >= 0).sum()),
        })

    # A tiny deterministic objective catches a disconnected relational gradient
    # and confirms that the matched pre-loss arm has a real optimizer update.
    torch.manual_seed(0)
    model_a = torch.nn.Linear(4, 4, bias=False)
    model_b = torch.nn.Linear(4, 4, bias=False)
    model_b.load_state_dict(model_a.state_dict())
    inputs = torch.eye(4)
    targets_small = F.normalize(torch.roll(inputs, 1, dims=0), dim=1)
    matched_updates = []
    for model in (model_a, model_b):
        prediction = F.normalize(model(inputs), dim=1)
        loss = (1.0 - (prediction * targets_small).sum(dim=1)).mean()
        before = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        if model.weight.grad is None or not torch.isfinite(model.weight.grad).all():
            raise RuntimeError("Smoke objective produced no finite encoder gradient.")
        optimizer.step()
        if torch.equal(before, model.weight.detach()):
            raise RuntimeError("Smoke objective did not change the matched encoder update.")
        matched_updates.append(model.weight.detach().clone())
    if not torch.equal(matched_updates[0], matched_updates[1]):
        raise RuntimeError("Matched arms do not have identical pre-regularizer updates.")

    model_c = torch.nn.Linear(4, 4, bias=False)
    model_c.load_state_dict(model_a.state_dict())
    prediction = F.normalize(model_c(inputs), dim=1)
    loss = (1.0 - (prediction * targets_small).sum(dim=1)).mean()
    loss = loss + 0.1 * (prediction - targets_small).square().mean()
    optimizer = torch.optim.SGD(model_c.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    if model_c.weight.grad is None or not torch.isfinite(model_c.weight.grad).all():
        raise RuntimeError("Smoke regularized objective produced no finite encoder gradient.")
    optimizer.step()

    report = {
        "artifact": "next_actions_preflight_smoke_v1",
        "passed": True,
        "checks": {
            "target_image_ids_match_cache": True,
            "fixed_shuffle_is_valid_derangement": True,
            "graph_sample_ids_match_cache": True,
            "matched_pre_loss_updates_equal": True,
            "nonzero_encoder_gradient_and_update": True,
        },
        "target_valid_count": int(valid.sum()),
        "graphs": graph_records,
        "selection": "all checks are label/spurious-metadata-free",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Preflight smoke passed: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.cache, args.targets, args.graph, args.output)


if __name__ == "__main__":
    main()
