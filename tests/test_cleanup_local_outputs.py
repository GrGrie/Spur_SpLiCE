import csv
import json
from pathlib import Path

from scripts.tools.cleanup_local_outputs import build_plan


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_cleanup_requires_evidence_for_checkpoints(tmp_path, monkeypatch):
    import scripts.tools.cleanup_local_outputs as cleanup

    monkeypatch.setattr(cleanup, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "outputs" / "legacy"
    kept = source / "run-a" / "last.pth"
    deleted = source / "run-b" / "last.pth"
    _write(kept, "checkpoint")
    _write(deleted, "checkpoint")
    _write(deleted.parent / "run_status.json", json.dumps({"status": "complete", "accuracy": 0.9}))

    plan = build_plan(source)

    assert [item["uri"] for item in plan["delete"]] == ["project://outputs/legacy/run-b/last.pth"]
    assert [item["uri"] for item in plan["retain"]] == ["project://outputs/legacy/run-a/last.pth"]


def test_cleanup_exports_csv_and_ignores_intermediate_checkpoint(tmp_path, monkeypatch):
    import scripts.tools.cleanup_local_outputs as cleanup

    monkeypatch.setattr(cleanup, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "outputs" / "legacy"
    study = source / "windows_splice_only_ablation"
    study.mkdir(parents=True, exist_ok=True)
    with (study / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "validation_accuracy"])
        writer.writeheader()
        writer.writerow({"run": "simclr", "validation_accuracy": "77.81"})
    final_dir = study / "training" / "waterbirds_s0_base_e100_final"
    intermediate_dir = study / "training" / "waterbirds_s0_base_e50_intermediate"
    for directory, epochs in ((final_dir, 100), (intermediate_dir, 50)):
        _write(directory / "args.json", json.dumps({"epochs": epochs, "crp_teacher_graph": ""}))
        _write(directory / "last.pth", "checkpoint")

    plan = build_plan(source)

    assert len(plan["legacy_results"]) == 1
    assert plan["legacy_results"][0]["rows"][0]["validation_accuracy"] == 77.81
    assert [item["uri"] for item in plan["delete"]] == [
        "project://outputs/legacy/windows_splice_only_ablation/training/waterbirds_s0_base_e100_final/last.pth"
    ]
    assert plan["summary"]["retain_files"] == 1


def test_cleanup_removes_reproducible_tensor_cache_without_result(tmp_path, monkeypatch):
    import scripts.tools.cleanup_local_outputs as cleanup

    monkeypatch.setattr(cleanup, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "outputs" / "legacy"
    _write(source / "splice_score_cache" / "embeddings.pt", "cache")

    plan = build_plan(source)

    assert plan["delete"][0]["reasons"] == ["reproducible-cache"]


def test_cleanup_does_not_treat_checkpoint_in_cache_directory_as_cache(tmp_path, monkeypatch):
    import scripts.tools.cleanup_local_outputs as cleanup

    monkeypatch.setattr(cleanup, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "outputs" / "legacy"
    _write(source / "cache" / "last.pt", "checkpoint")

    plan = build_plan(source)

    assert plan["delete"] == []
    assert plan["retain"][0]["kind"] == "checkpoint"
