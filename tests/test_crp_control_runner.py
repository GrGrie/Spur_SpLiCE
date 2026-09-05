import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from experiments.spurious_eval.linear_probe import build_wandb_group_metrics
from scripts.tools import run_crp_controls as runner


CONFIG_PATH = Path("scripts/run_crp_controls_cluster.conf")


def cluster_config(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["output"] = str(tmp_path / "cluster-output")
    config["cache"] = str(tmp_path / "cluster-output" / "cache.pt")
    return config


def test_cluster_config_has_eight_tasks_and_fixed_periodic_probe():
    config, arms = runner.load_and_validate_config(CONFIG_PATH)
    assert len(config["seeds"]) * len(arms) == 8
    assert config["epochs"] == 500
    assert config["linear_probe_mode"] == "periodic"
    assert config["linear_probe_freq"] == 25
    assert "--use_wandb" in runner.training_command(config, 1, "simclr", None, Path("out"))


def test_array_mapping_is_row_major():
    arms = ("simclr", "crp_sampler_only", "raw_clip_kl", "splice_crp_kl")
    assert runner.resolve_array_task(0, [1, 2], arms) == (1, "simclr")
    assert runner.resolve_array_task(3, [1, 2], arms) == (1, "splice_crp_kl")
    assert runner.resolve_array_task(4, [1, 2], arms) == (2, "simclr")
    assert runner.resolve_array_task(7, [1, 2], arms) == (2, "splice_crp_kl")
    with pytest.raises(ValueError, match="outside"):
        runner.resolve_array_task(8, [1, 2], arms)


def test_prepare_only_dispatches_without_training():
    with patch.object(sys, "argv", ["run_crp_controls", "--config", str(CONFIG_PATH), "--prepare-only"]), \
         patch.object(runner, "prepare_experiment") as prepare, \
         patch.object(runner, "run_command", side_effect=AssertionError("training must not run")):
        runner.main()
    prepare.assert_called_once_with(CONFIG_PATH)


def test_array_task_writes_only_its_arm_and_reuses_completed(tmp_path):
    config = cluster_config(tmp_path)
    output = Path(config["output"])
    prepared = {
        "config_fingerprint": "config-id",
        "cache_fingerprint": "cache-id",
        "graph_fingerprints": {"crp": "crp-id", "raw_clip": "raw-id"},
    }
    command_log = []

    def fake_run(command, log):
        command_log.append(command)
        result_path = output / "seed1" / "simclr" / "training" / "run" / "probe_features_epoch_500_ds_train_val.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({
            "convergence": {"converged": True, "epochs": 10},
            "metrics": {
                "Average over last 10 linear val acc": 50.0,
                "Average over last 10 linear val worst-group acc": 25.0,
                "Average over last 10 linear val best-group acc": 75.0,
            },
        }), encoding="utf-8")

    with patch.object(runner, "_load_prepared", return_value=(config, tuple(config["arms"]), output, prepared)), \
         patch.object(runner, "_run_identity", return_value={"id": "run"}), \
         patch.object(runner, "run_command", side_effect=fake_run):
        record = runner.run_one_arm(CONFIG_PATH, 0)

    assert record["arm"] == "simclr"
    assert (output / "seed1" / "simclr" / "command.json").is_file()
    assert not (output / "seed1" / "crp_sampler_only").exists()
    assert len(command_log) == 1

    with patch.object(runner, "_load_prepared", return_value=(config, tuple(config["arms"]), output, prepared)), \
         patch.object(runner, "_run_identity", return_value={"id": "run"}), \
         patch.object(runner, "run_command", side_effect=AssertionError("completed task must be reused")):
        reused = runner.run_one_arm(CONFIG_PATH, 0)
    assert reused == record


def test_summary_excludes_incomplete_seed(tmp_path):
    config = cluster_config(tmp_path)
    output = Path(config["output"])
    prepared = {
        "config_fingerprint": "config-id",
        "cache_fingerprint": "cache-id",
        "graph_fingerprints": {"crp": "crp-id", "raw_clip": "raw-id"},
    }
    with patch.object(runner, "_load_prepared", return_value=(config, tuple(config["arms"]), output, prepared)), \
         patch.object(runner, "_run_identity", return_value={"id": "run"}):
        for seed, arms in ((1, config["arms"]), (2, ["simclr"])):
            for arm in arms:
                result_path = output / f"seed{seed}" / arm / "training" / "probe_features_epoch_500_ds_train_val.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps({
                    "group_metrics": {"val": {"accuracy": [10, 20, 30, 40], "count": [1, 2, 3, 4]}},
                }), encoding="utf-8")
                runner._atomic_write_json(output / f"seed{seed}" / arm / "completed.json", {
                    "status": "complete", "seed": seed, "arm": arm,
                    "avg_acc_last10": 25.0, "wga_last10": 10.0,
                    "best_group_last10": 40.0, "probe_epochs": 10,
                    "result": str(result_path), "run_identity": {"id": "run"},
                })
        runner.summarize_experiment(CONFIG_PATH)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["paired_seeds"] == [1]
    assert summary["missing"]["2"] == ["crp_sampler_only", "raw_clip_kl", "splice_crp_kl"]
    assert len((output / "results.csv").read_text(encoding="utf-8").splitlines()) == 5


def test_wandb_group_names_are_target_context_and_include_all_four_groups():
    metrics = build_wandb_group_metrics(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.tensor([10, 20, 30, 40]),
        torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]),
    )
    assert metrics["Linear val group (target,context)=(0,0) acc"] == pytest.approx(10.0)
    assert metrics["Linear val group (target,context)=(0,1) acc"] == pytest.approx(20.0)
    assert metrics["Linear val group (target,context)=(1,0) acc"] == pytest.approx(30.0)
    assert metrics["Linear val group (target,context)=(1,1) acc"] == pytest.approx(40.0)
