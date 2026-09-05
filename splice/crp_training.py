"""Label-free CRP graph distillation for the SimCLR student."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from splice.crp import CRP_GRAPH_VERSION, CRP_V4_GRAPH_VERSION, CrpAuditConfig, GRAPH_VERSION
from splice.graph_io import graph_fingerprint, load_graph_json


TEACHER_GRAPH_ARTIFACTS = {
    "splice_raw_clip_matched_teacher_graph",
    "splice_crp_v2_teacher_graph",
    "splice_crp_v3_teacher_graph",
    "splice_crp_v4_teacher_graph",
    "splice_safe_crp_teacher_graph",
}
REQUIRED_GRAPH_KEYS = {
    "artifact",
    "graph_version",
    "sample_ids",
    "neighbor_indices",
    "weights",
    "confidence",
}
FORBIDDEN_ANNOTATION_KEYS = {
    "a",
    "attribute",
    "attributes",
    "label",
    "labels",
    "metadata",
    "spurious",
    "target",
    "targets",
    "y",
}


def validate_teacher_graph(graph: dict, expected_sample_ids: Sequence[str] | None = None) -> dict:
    """Validate the sparse CRP graph before it can influence training."""

    if not isinstance(graph, dict):
        raise ValueError("CRP teacher graph must be a dictionary.")

    def collect_keys(value) -> set[str]:
        if not isinstance(value, dict):
            return set()
        keys = {str(key).lower() for key in value}
        for nested in value.values():
            keys.update(collect_keys(nested))
        return keys

    forbidden = FORBIDDEN_ANNOTATION_KEYS.intersection(collect_keys(graph))
    if forbidden:
        raise ValueError(f"CRP teacher graph contains forbidden annotation keys: {sorted(forbidden)}")
    missing = REQUIRED_GRAPH_KEYS.difference(graph)
    if missing:
        raise ValueError(f"CRP teacher graph is missing required keys: {sorted(missing)}")
    config = graph.get("config", {})
    if isinstance(config, dict):
        unsupported = set(config).difference(CrpAuditConfig.__dataclass_fields__)
        if unsupported:
            raise ValueError(f"CRP teacher graph contains unsupported settings: {sorted(unsupported)}")
    if graph["artifact"] not in TEACHER_GRAPH_ARTIFACTS:
        raise ValueError(f"Unexpected relational teacher artifact type: {graph['artifact']!r}.")
    expected_versions = {
        "splice_raw_clip_matched_teacher_graph": 1,
        "splice_crp_v2_teacher_graph": GRAPH_VERSION,
        "splice_crp_v3_teacher_graph": CRP_GRAPH_VERSION,
        "splice_crp_v4_teacher_graph": CRP_V4_GRAPH_VERSION,
        "splice_safe_crp_teacher_graph": 1,
    }
    expected_version = expected_versions[graph["artifact"]]
    if graph["graph_version"] != expected_version:
        raise ValueError(
            f"Unsupported relational graph version {graph['graph_version']!r}; expected {expected_version}."
        )

    sample_ids = [str(sample_id) for sample_id in graph["sample_ids"]]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("CRP graph sample_ids must be non-empty and unique.")
    if expected_sample_ids is not None and sample_ids != [str(value) for value in expected_sample_ids]:
        raise ValueError(
            "CRP graph sample_ids do not exactly match the SSL train split. "
            "Rebuild the frozen cache and graph for this dataset configuration."
        )

    n_samples = len(sample_ids)
    indices = torch.as_tensor(graph["neighbor_indices"]).detach().long().cpu()
    weights = torch.as_tensor(graph["weights"]).detach().float().cpu()
    confidence = torch.as_tensor(graph["confidence"]).detach().float().cpu().view(-1)
    if indices.ndim != 2 or weights.shape != indices.shape:
        raise ValueError("CRP neighbor_indices and weights must be aligned rank-2 tensors.")
    if indices.shape[0] != n_samples or confidence.shape != (n_samples,):
        raise ValueError("CRP graph tensors must contain one row per sample_id.")
    if not torch.isfinite(weights).all() or not torch.isfinite(confidence).all():
        raise ValueError("CRP graph contains non-finite weights or confidence values.")

    valid = indices >= 0
    if torch.any(indices < -1) or torch.any(indices[valid] >= n_samples):
        raise ValueError("CRP neighbor index is outside the sample_id range.")
    if torch.any(weights < 0) or torch.any(weights[~valid] != 0):
        raise ValueError("CRP weights must be non-negative and zero for padded edges.")
    rows = torch.arange(n_samples).view(-1, 1).expand_as(indices)
    if torch.any(indices[valid] == rows[valid]):
        raise ValueError("CRP teacher graph must not contain self-edges.")
    row_sums = weights.sum(dim=1)
    supported = row_sums > 0
    if torch.any(valid & (weights <= 0)):
        raise ValueError("Every valid CRP edge must have positive weight.")
    if torch.any(~torch.isclose(row_sums[supported], torch.ones_like(row_sums[supported]), atol=1e-5)):
        raise ValueError("Supported CRP graph rows must be row-stochastic.")
    if torch.any(valid.any(dim=1) != supported):
        raise ValueError("CRP graph rows cannot contain zero-weight or weight-only edges.")

    anchor_confidence = graph.get("anchor_confidence", confidence)
    anchor_confidence = torch.as_tensor(anchor_confidence).detach().float().cpu().view(-1)
    if anchor_confidence.shape != (n_samples,) or not torch.isfinite(anchor_confidence).all():
        raise ValueError("CRP anchor_confidence must contain one finite value per sample.")
    if torch.any(anchor_confidence < 0) or torch.any(anchor_confidence > 1 + 1e-6):
        raise ValueError("CRP anchor_confidence must be in [0, 1].")
    if torch.any((anchor_confidence > 0) != supported):
        raise ValueError("CRP anchor confidence support must match graph edge support.")

    if graph["artifact"] == "splice_safe_crp_teacher_graph":
        from splice.crp_safe_graph import SAFE_CRP_GRAPH_VERSION, SafeCrpGraphConfig

        if graph["graph_version"] != SAFE_CRP_GRAPH_VERSION:
            raise ValueError("Unsupported safe CRP graph version.")
        SafeCrpGraphConfig.from_mapping(graph.get("safe_config"))
        for fingerprint_key in ("source_crp_fingerprint", "source_raw_fingerprint"):
            if not isinstance(graph.get(fingerprint_key), str) or not graph[fingerprint_key]:
                raise ValueError(f"Safe CRP graph requires {fingerprint_key}.")
        safe_shape = indices.shape
        edge_source = torch.as_tensor(graph.get("edge_source"), dtype=torch.long).detach().cpu()
        safe_groups = torch.as_tensor(graph.get("group_ids"), dtype=torch.long).detach().cpu()
        safe_gains = torch.as_tensor(graph.get("intervention_gains"), dtype=torch.float32).detach().cpu()
        safe_confidences = torch.as_tensor(graph.get("edge_confidences"), dtype=torch.float32).detach().cpu()
        if any(value.shape != safe_shape for value in (edge_source, safe_groups, safe_gains, safe_confidences)):
            raise ValueError("Safe CRP provenance tensors must align with neighbor_indices.")
        if torch.any(edge_source < 0) or torch.any(edge_source > 2):
            raise ValueError("Safe CRP edge_source contains an unknown value.")
        if torch.any(edge_source[~valid] != 0) or torch.any(edge_source[valid] == 0):
            raise ValueError("Safe CRP edge_source must mark padding and supported edges.")
        if torch.any((edge_source == 1) & ((safe_groups != -1) | (safe_gains != 0) | (safe_confidences != 0))):
            raise ValueError("Raw safe edges must have zero CRP provenance.")
        if torch.any((edge_source == 2) & ((safe_groups < 0) | (safe_gains <= 0) | (safe_confidences <= 0))):
            raise ValueError("Safe replacement edges must have positive CRP provenance.")
        stats = graph.get("degree_stats", {})
        replacement_count = int((edge_source == 2).sum())
        treated_count = int((edge_source == 2).any(dim=1).sum())
        if int(stats.get("safe_replaced_edges", -1)) != replacement_count:
            raise ValueError("Safe replacement count does not match edge provenance.")
        if int(stats.get("safe_treated_anchors", -1)) != treated_count:
            raise ValueError("Safe treated-anchor count does not match edge provenance.")
        if torch.any((edge_source == 2).sum(dim=1) > 1):
            raise ValueError("Safe CRP graph permits at most one replacement per row.")

    return {
        **graph,
        "sample_ids": sample_ids,
        "neighbor_indices": indices,
        "weights": weights,
        "confidence": confidence,
        "anchor_confidence": anchor_confidence.clamp(0, 1),
    }


def load_teacher_graph(
    path: str | Path,
    dataset_name: str,
    source_indices: Sequence[int],
) -> tuple[dict, str]:
    """Load a graph and bind its rows to the exact training subset order."""

    graph_path = Path(path)
    if not graph_path.is_file():
        raise FileNotFoundError(f"CRP teacher graph not found: {graph_path}")
    expected_sample_ids = [f"{dataset_name}:{int(index)}" for index in source_indices]
    graph = load_graph_json(graph_path)
    graph = validate_teacher_graph(graph, expected_sample_ids)
    return graph, graph_fingerprint(graph_path)


def build_crp_concept_report(graph: dict) -> dict:
    """Summarize which CRP concepts actually contribute edges to SSL training."""

    if graph.get("artifact") not in {
        "splice_crp_v2_teacher_graph",
        "splice_crp_v3_teacher_graph",
        "splice_crp_v4_teacher_graph",
    }:
        raise ValueError("CRP concept reports require a CRP teacher graph.")
    weights = torch.as_tensor(graph["weights"], dtype=torch.float32)
    group_ids = torch.as_tensor(
        graph.get("group_ids", torch.full_like(graph["neighbor_indices"], -1)),
        dtype=torch.long,
    )
    edge_confidences = torch.as_tensor(
        graph.get("edge_confidences", torch.zeros_like(weights)), dtype=torch.float32
    )
    gains = torch.as_tensor(
        graph.get("intervention_gains", torch.zeros_like(weights)), dtype=torch.float32
    )
    group_reports = []
    concept_reports = []
    for group in graph.get("groups", []):
        group_id = int(group["group_id"])
        edge_mask = (weights > 0) & (group_ids == group_id)
        used_rows = edge_mask.any(dim=1)
        edge_count = int(edge_mask.sum())
        evidence_mass = float((weights[edge_mask] * edge_confidences[edge_mask]).sum())
        mean_gain = float(gains[edge_mask].mean()) if edge_count else 0.0
        report = {
            "group_id": group_id,
            "concepts": [str(value) for value in group.get("concepts", [])],
            "selected": bool(group.get("selected", False)),
            "audit_score": float(group.get("score", 0.0)),
            "null_threshold": float(group.get("null_threshold", 0.0)),
            "null_excess_score": float(group.get("null_excess_score", 0.0)),
            "null_excess_ratio": float(group.get("null_excess_ratio", 0.0)),
            "coverage": float(group.get("coverage", 0.0)),
            "robust_positive_gain": float(group.get("robust_positive_gain", 0.0)),
            "semantic_agreement": float(group.get("semantic_agreement", 0.0)),
            "training_edge_count": edge_count,
            "training_anchor_count": int(used_rows.sum()),
            "training_evidence_mass": evidence_mass,
            "mean_training_edge_gain": mean_gain,
        }
        group_reports.append(report)
        for concept in report["concepts"]:
            concept_reports.append(
                {
                    "concept": concept,
                    "group_id": group_id,
                    "teacher_projected": report["selected"],
                    "training_edge_count": edge_count,
                    "training_evidence_mass": evidence_mass,
                    "audit_score": report["audit_score"],
                    "null_excess_score": report["null_excess_score"],
                }
            )

    group_reports.sort(
        key=lambda item: (
            -item["training_evidence_mass"],
            -item["null_excess_score"],
            item["group_id"],
        )
    )
    concept_reports.sort(
        key=lambda item: (
            -item["training_evidence_mass"],
            -item["null_excess_score"],
            item["concept"],
        )
    )
    return {
        "artifact": "splice_crp_concept_report_v1",
        "interpretation": (
            "teacher_projected means that the concept direction was removed only while "
            "constructing CRP teacher relations; it does not assert literal erasure from "
            "the trained student representation."
        ),
        "selected_group_count": sum(report["selected"] for report in group_reports),
        "teacher_projected_concepts": sorted(
            {
                concept
                for report in group_reports
                if report["selected"]
                for concept in report["concepts"]
            }
        ),
        "important_concepts": concept_reports,
        "groups": group_reports,
    }


def save_crp_concept_report(graph: dict, graph_path: str | Path) -> Path:
    report = build_crp_concept_report(graph)
    source_path = Path(graph_path)
    output_path = source_path.with_name(f"{source_path.stem}.concepts.json")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


class IndexedCrpDataset(Dataset):
    """Return augmented images and graph rows without reading annotations."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.transform = getattr(dataset, "transform", None)
        self.source_indices = getattr(dataset, "indices", None)
        self.source_dataset = getattr(dataset, "dataset", None)
        if self.source_indices is None or self.source_dataset is None or self.transform is None:
            raise ValueError("CRP training requires an indexed subset with an image transform.")
        if not hasattr(self.source_dataset, "get_input"):
            raise ValueError("CRP source dataset must expose get_input without annotations.")

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def collate(self):
        return getattr(self.dataset, "collate", None)

    def __getitem__(self, index: int):
        source_index = int(self.source_indices[index])
        image = self.source_dataset.get_input(source_index)
        return self.transform(image), int(index)


class CrpGraphBatchSampler(Sampler[list[int]]):
    """Group anchors with weighted graph neighbours at fixed epoch cost.

    Every train sample occurs exactly once per epoch, matching the SimCLR
    baseline's image and optimizer-step budget.  The random anchor order changes
    which supported rows get an explicitly paired neighbour across epochs.
    """

    def __init__(
        self,
        neighbor_indices: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
        generator: torch.Generator,
    ) -> None:
        if batch_size < 2:
            raise ValueError("CRP graph training requires batch_size >= 2.")
        self.neighbor_indices = neighbor_indices
        self.weights = weights
        self.batch_size = int(batch_size)
        self.generator = generator
        self.supported = weights.sum(dim=1) > 0

    def __len__(self) -> int:
        return math.ceil(len(self.neighbor_indices) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(len(self.neighbor_indices), generator=self.generator).tolist()
        unused = torch.ones(len(order), dtype=torch.bool)
        cursor = 0
        while cursor < len(order):
            batch: list[int] = []
            while len(batch) < self.batch_size:
                while cursor < len(order) and not bool(unused[order[cursor]]):
                    cursor += 1
                if cursor >= len(order):
                    break
                anchor = order[cursor]
                cursor += 1
                batch.append(anchor)
                unused[anchor] = False
                if len(batch) >= self.batch_size or not bool(self.supported[anchor]):
                    continue

                neighbor_row = self.neighbor_indices[anchor]
                row_weights = self.weights[anchor].clone()
                valid = neighbor_row >= 0
                if torch.any(valid):
                    available = torch.zeros_like(valid)
                    available[valid] = unused[neighbor_row[valid]]
                    row_weights[~available] = 0
                if float(row_weights.sum()) <= 0:
                    continue
                position = int(torch.multinomial(row_weights, 1, generator=self.generator).item())
                donor = int(neighbor_row[position])
                batch.append(donor)
                unused[donor] = False
            if batch:
                yield batch


def build_crp_training_loader(
    dataset,
    graph: dict,
    batch_size: int,
    num_workers: int,
    generator: torch.Generator,
    worker_init_fn=None,
) -> DataLoader:
    indexed_dataset = IndexedCrpDataset(dataset)
    batch_sampler = CrpGraphBatchSampler(
        graph["neighbor_indices"],
        graph["weights"],
        batch_size,
        generator,
    )
    loader = DataLoader(
        indexed_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=indexed_dataset.collate,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=generator,
    )
    loader.crp_graph = graph
    return loader


class CrpRelationalRegularizer:
    """Confidence-weighted KL distillation from a fixed CRP teacher graph."""

    enabled = True
    requires_clip_distillation = False
    requires_crp_indices = True

    def __init__(
        self,
        graph: dict,
        weight: float,
        temperature: float,
        start_epoch: int,
        warmup_epochs: int,
        decay_start_epoch: int = 0,
        decay_end_epoch: int = 0,
    ) -> None:
        if weight < 0:
            raise ValueError("CRP relational weight must be non-negative.")
        if temperature <= 0:
            raise ValueError("CRP student temperature must be positive.")
        if start_epoch < 0 or warmup_epochs < 0:
            raise ValueError("CRP start and warmup epochs must be non-negative.")
        if decay_start_epoch < 0 or decay_end_epoch < 0:
            raise ValueError("CRP decay epochs must be non-negative.")
        if bool(decay_start_epoch) != bool(decay_end_epoch):
            raise ValueError("CRP decay start/end must both be zero or both be set.")
        if decay_end_epoch and decay_end_epoch <= decay_start_epoch:
            raise ValueError("CRP decay end must be greater than decay start.")
        self.neighbor_indices = graph["neighbor_indices"]
        self.weights = graph["weights"]
        self.anchor_confidence = graph["anchor_confidence"]
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.start_epoch = int(start_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.decay_start_epoch = int(decay_start_epoch)
        self.decay_end_epoch = int(decay_end_epoch)
        self.epoch = 0
        self.last_diagnostics: dict[str, float] = {}

    def _set_diagnostics(
        self,
        *,
        supported_fraction: float = 0.0,
        mean_confidence: float = 0.0,
        unweighted_kl: float = 0.0,
        weighted_kl: float = 0.0,
    ) -> None:
        self.last_diagnostics = {
            "scheduled_weight": self.scheduled_weight,
            "supported_anchor_fraction": supported_fraction,
            "mean_anchor_confidence": mean_confidence,
            "unweighted_kl": unweighted_kl,
            "confidence_weighted_kl": weighted_kl,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def scheduled_weight(self) -> float:
        if self.epoch <= self.start_epoch:
            return 0.0
        if self.warmup_epochs == 0:
            scheduled = self.weight
        else:
            progress = (self.epoch - self.start_epoch) / self.warmup_epochs
            scheduled = self.weight * min(1.0, max(0.0, progress))
        if self.decay_end_epoch and self.epoch >= self.decay_start_epoch:
            remaining = (self.decay_end_epoch - self.epoch) / (
                self.decay_end_epoch - self.decay_start_epoch
            )
            scheduled *= min(1.0, max(0.0, remaining))
        return scheduled

    def __call__(self, embeddings: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[0] % 2:
            raise ValueError("CRP regularization expects two aligned views per batch sample.")
        batch_size = embeddings.shape[0] // 2
        sample_indices = torch.as_tensor(sample_indices).detach().long().cpu().view(-1)
        if sample_indices.shape != (batch_size,):
            raise ValueError("CRP regularization requires one graph-row index per batch sample.")
        if batch_size < 2 or self.scheduled_weight <= 0:
            self._set_diagnostics()
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

        first, second = embeddings.float().split(batch_size, dim=0)
        student = F.normalize(F.normalize(first, dim=1) + F.normalize(second, dim=1), dim=1)
        positions: dict[int, list[int]] = {}
        for position, sample_index in enumerate(sample_indices.tolist()):
            positions.setdefault(sample_index, []).append(position)

        teacher = torch.zeros(batch_size, batch_size, dtype=torch.float32)
        confidence = torch.zeros(batch_size, dtype=torch.float32)
        for anchor_position, anchor_index in enumerate(sample_indices.tolist()):
            for neighbor_index, edge_weight in zip(
                self.neighbor_indices[anchor_index].tolist(),
                self.weights[anchor_index].tolist(),
            ):
                if neighbor_index < 0 or edge_weight <= 0 or neighbor_index not in positions:
                    continue
                neighbor_positions = positions[neighbor_index]
                per_occurrence = edge_weight / len(neighbor_positions)
                teacher[anchor_position, neighbor_positions] += per_occurrence
            if float(teacher[anchor_position].sum()) > 0:
                confidence[anchor_position] = self.anchor_confidence[anchor_index]

        row_sums = teacher.sum(dim=1)
        supported = row_sums > 0
        if not torch.any(supported):
            self._set_diagnostics()
            return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
        teacher[supported] /= row_sums[supported].unsqueeze(1)

        sample_indices_device = sample_indices.to(embeddings.device)
        same_sample = sample_indices_device.view(-1, 1) == sample_indices_device.view(1, -1)
        logits = (student @ student.T) / self.temperature
        logits = logits.masked_fill(same_sample, -torch.inf)
        log_student = F.log_softmax(logits, dim=1)
        teacher = teacher.to(embeddings.device)
        confidence = confidence.to(embeddings.device)
        positive = teacher > 0
        log_teacher = torch.zeros_like(teacher)
        log_teacher[positive] = teacher[positive].log()
        kl_terms = torch.zeros_like(teacher)
        kl_terms[positive] = teacher[positive] * (log_teacher[positive] - log_student[positive])
        per_anchor_kl = kl_terms.sum(dim=1)
        unweighted_kl = per_anchor_kl[supported].mean()
        loss = (confidence[supported] * per_anchor_kl[supported]).mean()
        self._set_diagnostics(
            supported_fraction=float(supported.float().mean()),
            mean_confidence=float(confidence[supported].mean()),
            unweighted_kl=float(unweighted_kl.detach()),
            weighted_kl=float(loss.detach()),
        )
        return (self.scheduled_weight * loss).to(dtype=embeddings.dtype)
