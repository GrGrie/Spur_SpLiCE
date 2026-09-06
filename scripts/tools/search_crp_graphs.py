"""Frozen, label-free CRP grouping search and deterministic graph selection."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch

from scripts.tools.build_crp_baseline_graphs import build_matched_raw_clip_graph
from splice.crp import CrpAuditConfig, run_frozen_audit, validate_feature_cache
from splice.crp_graph_selection import (
    evaluate_grouping_gates,
    group_partition_jaccard,
    lexicographic_candidate_key,
    replacement_edges,
    selected_group_sets,
    set_jaccard,
)
from splice.crp_group_screen import (
    MiniInterventionConfig,
    ReconstructionScreenConfig,
    build_group_screen,
)
from splice.crp_safe_graph import (
    SafeCrpGraphConfig,
    build_safe_crp_graph,
    validate_safe_crp_graph,
)
from splice.graph_io import graph_fingerprint, load_graph_json, save_graph_json


SEARCH_PROTOCOL = "crp_graph_search_v1"
DEFAULT_GATES = {
    "min_reconstruction_coverage": 0.99,
    "min_compression_gain": 0.05,
    "max_compression_gain": 0.50,
    "max_largest_group_size": 16,
    "max_largest_group_fraction": 0.05,
    "min_text_cosine_p10": 0.55,
    "min_coactivation_cosine_p10": 0.05,
    "min_top1_turnover": 0.10,
    "max_jaccard_for_change": 0.90,
    "min_jaccard_guard": 0.50,
}


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_search_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("protocol") != SEARCH_PROTOCOL:
        raise ValueError(f"Expected protocol={SEARCH_PROTOCOL!r}.")
    required = ("dataset", "data_folder", "cache", "output", "grouping_grid", "audit", "graph_seeds")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Search config is missing: {missing}")
    if list(config["graph_seeds"]) != [0, 1, 2]:
        raise ValueError("graph_seeds must be exactly [0, 1, 2].")
    grid = config["grouping_grid"]
    if list(grid.get("text_similarity_threshold", [])) != [0.70, 0.75, 0.80]:
        raise ValueError("The text similarity grid must be [0.70, 0.75, 0.80].")
    if list(grid.get("coactivation_threshold", [])) != [0.15, 0.25, 0.35]:
        raise ValueError("The coactivation grid must be [0.15, 0.25, 0.35].")
    budgets = config.get("treatment_mass_budgets", [0.02, 0.05, 0.10])
    if list(budgets) != [0.02, 0.05, 0.10] or config.get("primary_treatment_mass_budget") != 0.05:
        raise ValueError("Treatment budgets must be [0.02, 0.05, 0.10] with primary budget 0.05.")
    if config.get("selection_rule_version") != "crp_graph_selection_v1":
        raise ValueError("Unexpected selection rule version.")
    return config


def candidate_grid(config: dict) -> list[dict]:
    grid = config["grouping_grid"]
    candidates = []
    for text_threshold, coactivation_threshold in itertools.product(
        grid["text_similarity_threshold"], grid["coactivation_threshold"]
    ):
        candidate_id = f"text_{text_threshold:.2f}_coact_{coactivation_threshold:.2f}".replace(".", "p")
        candidates.append({
            "candidate_id": candidate_id,
            "text_similarity_threshold": float(text_threshold),
            "coactivation_threshold": float(coactivation_threshold),
            "singleton_control": text_threshold == 0.80 and coactivation_threshold == 0.35,
        })
    return candidates


def resolve_candidate(task_id: int, config: dict) -> dict:
    candidates = candidate_grid(config)
    if task_id < 0 or task_id >= len(candidates):
        raise ValueError(f"Candidate task id {task_id} is outside 0..{len(candidates) - 1}.")
    return candidates[task_id]


def _audit_config(config: dict, candidate: dict, seed: int = 0, max_selected_groups: int | None = None) -> CrpAuditConfig:
    values = dict(config["audit"])
    values.update({
        "text_similarity_threshold": candidate["text_similarity_threshold"],
        "coactivation_threshold": candidate["coactivation_threshold"],
        "seed": seed,
    })
    if max_selected_groups is not None:
        values["max_selected_groups"] = max_selected_groups
    return CrpAuditConfig(**values)


def _screen_configs(config: dict) -> tuple[ReconstructionScreenConfig, MiniInterventionConfig]:
    gates = {**DEFAULT_GATES, **config.get("grouping_gates", {})}
    return (
        ReconstructionScreenConfig(
            fidelity_threshold=float(config.get("fidelity_threshold", 0.90)),
            target_image_coverage=float(gates["min_reconstruction_coverage"]),
            curve_points=int(config.get("screen_curve_points", 80)),
        ),
        MiniInterventionConfig(
            enabled=True,
            sample_count=int(config.get("mini_samples", 1024)),
            max_groups=int(config.get("mini_max_groups", 24)),
            projected_neighbors=int(config["audit"].get("projected_neighbors", 20)),
            null_trials=int(config.get("mini_null_trials", 4)),
            null_quantile=float(config.get("mini_null_quantile", 0.95)),
            activation_difference_quantile=float(config["audit"].get("activation_difference_quantile", 0.75)),
            min_intervention_gain=float(config["audit"].get("min_intervention_gain", 1e-4)),
            min_coverage=float(config["audit"].get("min_coverage", 0.01)),
            residual_splice_similarity_threshold=float(config["audit"].get("residual_splice_similarity_threshold", 0.25)),
            example_edges_per_group=0,
            seed=0,
            device="cpu",
        ),
    )


def _safe_config(config: dict, budget: float) -> SafeCrpGraphConfig:
    return SafeCrpGraphConfig(
        raw_guard_k=int(config.get("raw_guard_k", 20)),
        max_replacements_per_row=int(config.get("max_replacements_per_row", 1)),
        max_replacement_weight=float(config.get("max_replacement_weight", 0.34)),
        min_treated_anchor_fraction=float(config.get("min_treated_anchor_fraction", 0.02)),
        min_crp_weight_mass_fraction=float(config.get("min_crp_weight_mass_fraction", 0.0)),
        max_crp_training_mass_fraction=float(budget),
        max_group_training_mass_fraction=float(config.get("max_group_training_mass_fraction", 0.35)),
    )


def _graph_record(path: Path) -> dict:
    return {"path": str(path), "fingerprint": graph_fingerprint(path)}


def search_one_candidate(config_path: Path, task_id: int) -> Path:
    config = load_search_config(config_path)
    candidate = resolve_candidate(task_id, config)
    output = Path(config["output"])
    candidate_root = output / "candidates" / candidate["candidate_id"]
    candidate_root.mkdir(parents=True, exist_ok=True)
    cache = validate_feature_cache(torch.load(config["cache"], map_location="cpu", weights_only=True))
    screen_config, mini_config = _screen_configs(config)
    group_config = _audit_config(config, candidate, seed=0, max_selected_groups=0)
    screen = build_group_screen(cache, group_config, screen_config, mini_config)
    gates = evaluate_grouping_gates(
        screen["metrics"], screen.get("mini_intervention"),
        {**DEFAULT_GATES, **config.get("grouping_gates", {})},
    )
    report = {
        "artifact": "crp_graph_search_candidate_v1",
        "protocol": SEARCH_PROTOCOL,
        "candidate": candidate,
        "candidate_id": candidate["candidate_id"],
        "group_screen": screen,
        "grouping_metrics": screen["metrics"],
        "grouping_gates": gates,
        "status": "GROUPING_REJECTED" if not gates["passed"] else "GROUPING_PASSED",
        "graph_seeds": list(config["graph_seeds"]),
        "graphs": {},
    }
    atomic_write_json(candidate_root / "group_screen.json", screen)
    if not gates["passed"]:
        report["rejection_reasons"] = gates["failed_gates"]
        atomic_write_json(candidate_root / "candidate_report.json", report)
        return candidate_root

    budgets = [float(value) for value in config["treatment_mass_budgets"]]
    for graph_seed in config["graph_seeds"]:
        seed_root = candidate_root / f"graph_seed{graph_seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        audit = run_frozen_audit(cache, _audit_config(config, candidate, int(graph_seed), max_selected_groups=int(config["audit"].get("max_selected_groups", 12))))
        crp_path = seed_root / "crp_graph.json"
        save_graph_json(audit, crp_path)
        raw = build_matched_raw_clip_graph(cache, audit)
        raw_path = seed_root / "raw_clip_graph.json"
        save_graph_json(raw, raw_path)
        graph_payload = {
            "crp": _graph_record(crp_path),
            "raw_clip": _graph_record(raw_path),
            "safe": {},
        }
        for budget in budgets:
            safe = build_safe_crp_graph(
                cache, audit, raw, _safe_config(config, budget),
                source_crp_fingerprint=graph_fingerprint(crp_path),
                source_raw_fingerprint=graph_fingerprint(raw_path),
            )
            safe_path = seed_root / f"safe_graph_{budget:.2f}.json"
            save_graph_json(safe, safe_path)
            graph_payload["safe"][f"{budget:.2f}"] = _graph_record(safe_path)
        report["graphs"][str(graph_seed)] = graph_payload
    report["status"] = "GRAPH_AUDITED"
    atomic_write_json(candidate_root / "candidate_report.json", report)
    return candidate_root


def _load_graph_record(record: dict) -> dict:
    path = Path(record["path"])
    if graph_fingerprint(path) != record["fingerprint"]:
        raise ValueError(f"Graph fingerprint changed: {path}")
    return load_graph_json(path)


def _structural_invariants(raw: dict, safe: dict) -> dict[str, bool]:
    raw_indices = raw["neighbor_indices"]
    safe_indices = safe["neighbor_indices"]
    return {
        "support_preserved": bool(torch.equal(raw_indices >= 0, safe_indices >= 0)),
        "weights_preserved": bool(torch.equal(raw["weights"], safe["weights"])),
        "confidence_preserved": bool(torch.equal(raw["anchor_confidence"], safe["anchor_confidence"])),
        "row_degree_preserved": bool(torch.equal((raw_indices >= 0).sum(1), (safe_indices >= 0).sum(1))),
    }


def _evidence_density(safe: dict) -> float:
    stats = safe.get("degree_stats", {})
    total = float(stats.get("safe_training_mass_total", 0.0))
    if total <= 0:
        return 0.0
    return sum(
        float(item.get("training_mass", 0.0)) * float(item.get("confidence", 0.0))
        for item in safe.get("safe_replacements", [])
    ) / total


def assess_budget(config: dict, candidate: dict, budget: float) -> dict:
    reports = []
    for graph_seed in config["graph_seeds"]:
        seed_payload = candidate["graphs"][str(graph_seed)]
        crp = _load_graph_record(seed_payload["crp"])
        raw = _load_graph_record(seed_payload["raw_clip"])
        safe = _load_graph_record(seed_payload["safe"][f"{budget:.2f}"])
        safe = validate_safe_crp_graph(safe)
        invariants = _structural_invariants(raw, safe)
        stats = safe.get("degree_stats", {})
        reports.append({
            "graph_seed": int(graph_seed),
            "crp": crp,
            "raw": raw,
            "safe": safe,
            "invariants": invariants,
            "training_mass_fraction": float(stats.get("safe_crp_training_mass_fraction", 0.0)),
            "treated_anchor_fraction": float(stats.get("safe_treated_anchor_fraction", 0.0)),
            "effective_group_count": int(stats.get("safe_effective_contributing_group_count", 0)),
            "treatment_mass_hhi": float(stats.get("safe_treatment_mass_hhi", 1.0)),
            "evidence_density": _evidence_density(safe),
            "group_sets": selected_group_sets(crp),
            "replacement_edges": replacement_edges(safe),
            "crp_path": seed_payload["crp"]["path"],
            "raw_path": seed_payload["raw_clip"]["path"],
            "safe_path": seed_payload["safe"][f"{budget:.2f}"]["path"],
        })
    group_stability = min(
        group_partition_jaccard(left["group_sets"], right["group_sets"])
        for left, right in itertools.combinations(reports, 2)
    )
    edge_stability = min(
        set_jaccard(left["replacement_edges"], right["replacement_edges"])
        for left, right in itertools.combinations(reports, 2)
    )
    per_seed_checks = all(
        all(item["invariants"].values())
        and item["training_mass_fraction"] <= budget + 1e-12
        and item["treated_anchor_fraction"] >= float(config.get("min_treated_anchor_fraction", 0.02))
        and item["effective_group_count"] >= int(config.get("min_effective_contributing_groups", 3))
        and max(item["safe"].get("degree_stats", {}).get("safe_group_training_mass_fraction", {}).values() or [0.0])
            <= float(config.get("max_group_training_mass_fraction", 0.35)) + 1e-12
        for item in reports
    )
    hard_checks = {
        "selected_group_stability": group_stability >= float(config.get("min_selected_group_jaccard", 0.67)),
        "replacement_edge_stability": edge_stability >= float(config.get("min_replacement_edge_jaccard", 0.80)),
        "per_seed_safe_invariants": per_seed_checks,
    }
    return {
        "budget": budget,
        "graph_seed_reports": reports,
        "minimum_selected_group_jaccard": group_stability,
        "minimum_replacement_edge_jaccard": edge_stability,
        "median_training_mass_weighted_evidence_density": float(torch.tensor([item["evidence_density"] for item in reports]).median()),
        "treatment_mass_hhi": max(item["treatment_mass_hhi"] for item in reports),
        "hard_checks": hard_checks,
        "admissible": all(hard_checks.values()),
    }


def select_experiment(config_path: Path) -> Path:
    config = load_search_config(config_path)
    output = Path(config["output"])
    reports = []
    for candidate in candidate_grid(config):
        report_path = output / "candidates" / candidate["candidate_id"] / "candidate_report.json"
        if report_path.is_file():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    candidates = []
    assessed_by_id = {}
    for report in reports:
        if report.get("status") != "GRAPH_AUDITED":
            continue
        budget_report = assess_budget(config, report, float(config["primary_treatment_mass_budget"]))
        assessed = {**report, **budget_report}
        candidates.append(assessed)
        assessed_by_id[report["candidate_id"]] = assessed
    admissible = [report for report in candidates if report["admissible"]]
    selection = {
        "artifact": "crp_graph_selection_v1",
        "protocol": SEARCH_PROTOCOL,
        "selection_rule_version": config["selection_rule_version"],
        "candidate_reports": [],
    }
    report_by_id = {report["candidate_id"]: report for report in reports}
    for candidate_spec in candidate_grid(config):
        report = report_by_id.get(candidate_spec["candidate_id"], {
            "candidate_id": candidate_spec["candidate_id"],
            "status": "MISSING_SHARD",
            "rejection_reasons": ["candidate_report_missing"],
        })
        assessed = assessed_by_id.get(candidate_spec["candidate_id"], report)
        failed = list(report.get("rejection_reasons", report.get("grouping_gates", {}).get("failed_gates", [])))
        failed.extend(name for name, passed in (assessed.get("hard_checks") or {}).items() if not passed)
        selection["candidate_reports"].append({
            "candidate_id": report["candidate_id"],
            "status": assessed.get("status"),
            "admissible": bool(assessed.get("admissible", False)),
            "hard_checks": assessed.get("hard_checks"),
            "rejection_reasons": list(dict.fromkeys(failed)),
        })
    if not admissible:
        selection.update({"status": "NO_ADMISSIBLE_CANDIDATE", "reason": "No grouping/budget/stability candidate passed all fixed gates."})
        atomic_write_json(output / "selection.json", selection)
        return output / "selection.json"
    winner = sorted(admissible, key=lexicographic_candidate_key)[0]
    winner_seed0 = next(item for item in winner["graph_seed_reports"] if item["graph_seed"] == 0)
    selected_graph = output / "selected_graph.json"
    selected_raw = output / "selected_raw_clip_graph.json"
    shutil.copyfile(winner_seed0["safe_path"], selected_graph)
    shutil.copyfile(winner_seed0["raw_path"], selected_raw)
    selected_identity = {
        "candidate_id": winner["candidate_id"],
        "budget": winner["budget"],
        "safe_graph": graph_fingerprint(selected_graph),
        "raw_graph": graph_fingerprint(selected_raw),
        "source_crp": winner_seed0["safe"].get("source_crp_fingerprint") if isinstance(winner_seed0["safe"], dict) else None,
    }
    # winner_seed0 holds the loaded graph payload, not the record; recover the
    # source fingerprints from the saved graph and make the identity explicit.
    selected_identity["source_crp"] = json.loads(selected_graph.read_text(encoding="utf-8")).get("source_crp_fingerprint")
    selected_identity["source_raw"] = json.loads(selected_graph.read_text(encoding="utf-8")).get("source_raw_fingerprint")
    atomic_write_json(output / "selected_graph_identity.json", selected_identity)
    controls = dict(config.get("selected_controls", {}))
    controls.update({
        "protocol": "safe_crp_controls_v1",
        "cache": config["cache"],
        "data_folder": config["data_folder"],
        "output": str(output / "selected_controls"),
        "seeds": list(controls.get("seeds", [0])),
        "graph_seed": int(controls.get("graph_seed", 0)),
        "graph": dict(config["audit"]),
        "arms": ["raw_clip_sampler_only", "safe_crp_sampler_only", "raw_clip_kl", "safe_crp_kl"],
        "prebuilt_graph_paths": {"raw_clip": str(selected_raw), "safe_crp": str(selected_graph)},
        "prebuilt_graph_fingerprints": {"raw_clip": graph_fingerprint(selected_raw), "safe_crp": graph_fingerprint(selected_graph)},
        "safe_graph": asdict(_safe_config(config, float(config["primary_treatment_mass_budget"]))),
        "selected_candidate_id": winner["candidate_id"],
    })
    atomic_write_json(output / "selected_safe_controls.conf", controls)
    selection.update({"status": "SELECTED", "candidate_id": winner["candidate_id"], "budget": winner["budget"], "selected_graph": str(selected_graph), "selected_graph_identity": str(output / "selected_graph_identity.json")})
    atomic_write_json(output / "selection.json", selection)
    return output / "selection.json"


def _load_vector(path: Path) -> list[int]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"Annotation file must contain one vector: {path}")
        value = next(iter(value.values()))
    return [int(item) for item in torch.as_tensor(value).view(-1).tolist()]


def posthoc_only(
    config_path: Path,
    graph_path: Path,
    labels_path: Path | None = None,
    contexts_path: Path | None = None,
) -> Path:
    """Validate an already selected graph without touching selection.json."""

    config = load_search_config(config_path)
    graph = load_graph_json(graph_path)
    safe = validate_safe_crp_graph(graph)
    if (labels_path is None) != (contexts_path is None):
        raise ValueError("--labels and --contexts must be supplied together.")
    if labels_path is not None:
        labels = _load_vector(labels_path)
        contexts = _load_vector(contexts_path)
        if len(labels) != len(safe["sample_ids"]) or len(contexts) != len(labels):
            raise ValueError("Post-hoc labels/contexts must align with the selected graph rows.")
        from scripts.tools.crp_posthoc_diagnostics import group_graph_diagnostics

        diagnostics = group_graph_diagnostics(safe, labels, contexts)
        annotation_source = "explicit_posthoc_vectors"
    else:
        from scripts.tools.crp_posthoc_diagnostics import diagnose_fixed_graphs

        diagnostics = diagnose_fixed_graphs(
            {"selected_safe": safe}, config["dataset"], config["data_folder"]
        )["graphs"]["selected_safe"]
        annotation_source = "dataset_registry_posthoc"
    report = {
        "artifact": "crp_graph_search_posthoc_v1",
        "protocol": SEARCH_PROTOCOL,
        "graph": str(graph_path),
        "graph_fingerprint": graph_fingerprint(graph_path),
        "safe_stats": safe.get("degree_stats", {}),
        "group_diagnostics": diagnostics,
        "annotation_source": annotation_source,
        "selection_mutated": False,
    }
    output = Path(config["output"]) / "selected_graph_posthoc.json"
    atomic_write_json(output, report)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--array-task-id", type=int)
    modes.add_argument("--select-only", action="store_true")
    modes.add_argument("--posthoc-only", action="store_true")
    parser.add_argument("--selected-graph", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--contexts", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.array_task_id is not None:
        print(f"Search shard ready at {search_one_candidate(args.config, args.array_task_id)}")
    elif args.select_only:
        print(f"Selection written to {select_experiment(args.config)}")
    else:
        if args.selected_graph is None:
            raise ValueError("--posthoc-only requires --selected-graph.")
        print(f"Post-hoc report written to {posthoc_only(args.config, args.selected_graph, args.labels, args.contexts)}")


if __name__ == "__main__":
    main()
