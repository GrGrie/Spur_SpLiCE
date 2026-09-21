"""CoSpRo relational distillation: a frozen teacher graph shapes the student's relations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from cospro.methods.base import LoaderContext, LossTerms, TrainingMethod, register_method
from cospro.compat import LEGACY_RELATIONAL_MODE, LEGACY_TEACHER_GRAPH_ARTIFACTS
from cospro.pipeline import COSPRO_TEACHER_GRAPH_ARTIFACT
from cospro.methods.relational_graph import (
    CoSpRoRelationalRegularizer,
    build_cospro_training_loader,
    load_teacher_graph,
    save_cospro_concept_report,
)


@register_method
class CoSpRoRelational(TrainingMethod):
    """Graph-aware batches plus the confidence-weighted relational KL term."""

    name = "cospro_relational"
    modes = ("cospro_relational", LEGACY_RELATIONAL_MODE)
    needs_sample_indices = True

    def __init__(
        self,
        *,
        graph_path: str = "",
        expected_fingerprint: str | None = None,
        weight: float = 0.0,
        temperature: float = 0.1,
        start_epoch: int = 0,
        warmup_epochs: int = 0,
        decay_start_epoch: int = 0,
        decay_end_epoch: int = 0,
        simclr_weight: float = 1.0,
        regularizer: CoSpRoRelationalRegularizer | None = None,
    ) -> None:
        self.graph_path = graph_path
        self.expected_fingerprint = expected_fingerprint
        self.simclr_weight = simclr_weight
        self._settings = {
            "weight": weight, "temperature": temperature, "start_epoch": start_epoch,
            "warmup_epochs": warmup_epochs, "decay_start_epoch": decay_start_epoch,
            "decay_end_epoch": decay_end_epoch,
        }
        self.regularizer = regularizer
        self.graph: dict | None = None
        self.graph_empty = False

    @classmethod
    def from_config(cls, config, **resolved) -> "CoSpRoRelational":
        return cls(
            graph_path=config.cospro.cospro_teacher_graph,
            expected_fingerprint=resolved.get("graph_fingerprint"),
            weight=config.method.splice_weight,
            temperature=config.cospro.cospro_temperature,
            start_epoch=config.cospro.cospro_start_epoch,
            warmup_epochs=config.cospro.cospro_warmup_epochs,
            decay_start_epoch=config.cospro.cospro_decay_start_epoch,
            decay_end_epoch=config.cospro.cospro_decay_end_epoch,
            simclr_weight=config.ssl.simclr_weight,
        )

    def wrap_loader(self, loader, context: LoaderContext):
        source_indices = getattr(loader.dataset, "indices", None)
        if source_indices is None:
            raise ValueError("CoSpRo training requires an SSL dataset with stable source indices.")
        graph, loaded_graph_fingerprint = load_teacher_graph(self.graph_path, context.dataset, source_indices)
        if self.expected_fingerprint is not None and loaded_graph_fingerprint != self.expected_fingerprint:
            raise ValueError("Relational teacher graph changed after argument validation; restart the run.")
        self.expected_fingerprint = loaded_graph_fingerprint
        self.graph = graph
        stats = graph.get("degree_stats", {})
        if graph["artifact"] in {COSPRO_TEACHER_GRAPH_ARTIFACT, *LEGACY_TEACHER_GRAPH_ARTIFACTS}:
            report_path = save_cospro_concept_report(graph, self.graph_path)
            concept_report = json.loads(report_path.read_text(encoding="utf-8"))
            top_concepts = [
                item["concept"]
                for item in concept_report["important_concepts"]
                if item["training_edge_count"] > 0
            ][:10]
            print(
                f"[INFO] CoSpRo concept report: path={report_path}, "
                f"teacher_projected={concept_report['teacher_projected_concepts']}, "
                f"top_training_concepts={top_concepts}",
                flush=True,
            )
        print(
            f"[INFO] Loaded {graph['artifact']} teacher graph: "
            f"edges={stats.get('edge_count', int((graph['neighbor_indices'] >= 0).sum()))}, "
            f"coverage={stats.get('coverage', float((graph['weights'].sum(dim=1) > 0).float().mean())):.4f}, "
            f"path={self.graph_path}",
            flush=True,
        )
        if not torch.any(graph["weights"].sum(dim=1) > 0):
            if self.simclr_weight == 0:
                raise ValueError(
                    "KL-only relational training requires a non-empty teacher graph; "
                    "the resolved graph contains no supported anchors."
                )
            self.graph_empty = True
            print(
                "[WARNING] Relational teacher graph is empty; using the standard SimCLR "
                "DataLoader and disabling relational regularization.",
                flush=True,
            )
            return loader
        self.regularizer = CoSpRoRelationalRegularizer(graph, **self._settings)
        return build_cospro_training_loader(
            loader.dataset, graph, context.batch_size, context.num_workers, loader.generator,
            worker_init_fn=context.worker_init_fn,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.regularizer is not None:
            self.regularizer.set_epoch(epoch)

    def extra_loss(self, *, model, embeddings, sample_indices) -> LossTerms | None:
        if self.regularizer is None:
            return None
        if sample_indices is None:
            raise ValueError("CoSpRo relational regularization requires graph-row sample indices.")
        value = self.regularizer(embeddings, sample_indices)
        return LossTerms(value=value, diagnostics=self.regularizer.last_diagnostics)

    def diagnostics(self) -> Mapping[str, float]:
        return getattr(self.regularizer, "last_diagnostics", {})

    def provenance(self) -> dict[str, Any]:
        if self.graph is None:
            return {"relational_graph_empty": self.graph_empty}
        return {
            "cospro_graph_fingerprint": self.expected_fingerprint,
            "teacher_graph_artifact": self.graph["artifact"],
            "teacher_graph_config": self.graph.get("config", {}),
            "teacher_graph_degree_stats": self.graph.get("degree_stats", {}),
            "teacher_graph_selected_group_ids": self.graph.get("selected_group_ids", []),
            "teacher_graph_removed_concepts": sorted(
                {
                    concept
                    for group in self.graph.get("groups", [])
                    if group.get("selected")
                    for concept in group.get("concepts", [])
                }
            ),
            "relational_graph_empty": self.graph_empty,
        }

    def input_artifacts(self) -> list[Path]:
        return [Path(self.graph_path)] if self.graph_path else []
