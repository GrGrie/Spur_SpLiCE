"""Read-only labelled diagnostics of a fixed graph; never used by graph builders."""
from __future__ import annotations

import torch


def group_graph_diagnostics(graph, targets, contexts):
    """Include unsupported sources in coverage and per-source mass denominators."""
    y, a = torch.as_tensor(targets).cpu(), torch.as_tensor(contexts).cpu()
    neighbors = graph["neighbor_indices"].cpu()
    weights = graph["weights"].cpu()
    q = graph["anchor_confidence"].cpu()
    if y.ndim != 1 or a.shape != y.shape or len(y) != len(neighbors):
        raise ValueError("Diagnostic annotations must align with graph rows.")
    valid = (neighbors >= 0) & (weights > 0)
    safe = neighbors.clamp_min(0)
    same_y, same_a = y[:, None] == y[safe], a[:, None] == a[safe]
    categories = {
        "same_target_same_context": same_y & same_a,
        "same_target_cross_context": same_y & ~same_a,
        "wrong_target_same_context": ~same_y & same_a,
        "wrong_target_cross_context": ~same_y & ~same_a,
    }
    mass = q[:, None] * weights * valid
    report = {}
    for pair in torch.unique(torch.stack((y, a), dim=1), dim=0):
        sources = (y == pair[0]) & (a == pair[1])
        count = int(sources.sum())
        edges = int(valid[sources].sum())
        total_mass = float(mass[sources].sum())
        detail = {}
        for name, mask in categories.items():
            selected = valid & mask
            selected_mass = float((mass * mask)[sources].sum())
            detail[name] = {
                "edge_fraction": int(selected[sources].sum()) / edges if edges else None,
                "confidence_weighted_mass_fraction": selected_mass / total_mass if total_mass else None,
                "confidence_weighted_mass_per_source": selected_mass / count,
                "source_fraction_with_donor": float(selected[sources].any(1).float().mean()),
            }
        report[f"target={int(pair[0])},context={int(pair[1])}"] = {
            "sources": count, "edges": edges,
            "supported_source_fraction": float(valid[sources].any(1).float().mean()),
            "confidence_weighted_mass_per_source": total_mass / count,
            "relations": detail,
        }
    return report


def diagnose_fixed_graphs(graphs, dataset_name, data_folder):
    """Load annotations only after graph construction; return a separate report."""
    from experiments.spurious_eval.datasets.registry import get_dataset_spec

    dataset = get_dataset_spec(dataset_name)["dataset"](data_folder)
    result = {"usage": "posthoc_only_not_for_selection", "graphs": {}}
    train_ids = {int(i) for i in dataset.get_subset("train", transform=None).indices}
    for name, graph in graphs.items():
        indices = []
        for sample_id in graph["sample_ids"]:
            prefix, index = str(sample_id).rsplit(":", 1)
            if prefix != dataset_name or int(index) not in train_ids:
                raise ValueError("Graph sample IDs do not match the train dataset.")
            indices.append(int(index))
        # Supported dataset registry entries use metadata column zero for context.
        result["graphs"][name] = group_graph_diagnostics(
            graph, dataset.y_array[indices], dataset.metadata_array[indices, 0])
    return result
