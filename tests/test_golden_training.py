"""Golden end-to-end training runs: two CPU epochs per training mode on a synthetic Waterbirds.

Each mode runs ``spur_splice.py`` in a subprocess exactly as the cluster launchers do. The
per-epoch losses plus final probe metrics from ``run.json`` are compared with the snapshot.
Floating-point results depend on the BLAS build, so each operating system keeps its own snapshot
(``training_runs.<system>.json``). The first run on a new system creates it for review and commit.

``SPUR_SPLICE_GOLDEN_DEVICE=cuda`` runs the same modes on a GPU with AMP and channels-last memory
(the cluster configuration). ``SPUR_SPLICE_GOLDEN_REPORT`` names a JSON file for the run summaries.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from golden_support import (
    assert_close_tree,
    compare_or_update,
    synthetic_splice_cache,
    synthetic_target_bank,
    train_sample_ids,
    write_waterbirds_fixture,
)
from cospro.pipeline import CoSpRoAuditConfig, build_concept_groups, build_teacher_graph
from cospro.pipeline.graph_io import save_graph_json
from cospro.tracking.artifacts import PROJECT_ROOT

SNAPSHOT = f"training_runs.{platform.system().lower()}.json"
DEVICE = os.environ.get("SPUR_SPLICE_GOLDEN_DEVICE", "cpu")
REPORT = os.environ.get("SPUR_SPLICE_GOLDEN_REPORT", "")
ACCELERATED = "true" if DEVICE.startswith("cuda") else "false"
COMMON_ARGS = [
    "--dataset", "waterbirds", "--model", "resnet18_large", "--epochs", "2", "--batch_size", "16",
    "--num_workers", "0", "--device", DEVICE, "--amp", ACCELERATED, "--channels_last", ACCELERATED,
    "--rank_eval_freq", "0", "--linear_probe_mode", "final", "--linear_probe_max_epochs", "50",
    "--seed", "1", "--study", "golden", "--attempt_id", "golden",
    "--delete_checkpoints_after_training", "true",
]
GRAPH_CONFIG = CoSpRoAuditConfig(
    text_similarity_threshold=0.8, coactivation_threshold=0.35, projected_neighbors=6,
    graph_top_k=3, max_indegree=6, null_trials=8,
)


def mode_args(inputs: Path) -> dict[str, list[str]]:
    graph = str(inputs / "teacher_graph.json")
    relational = ["--splice_mode", "cospro_relational", "--cospro_teacher_graph", graph,
                  "--cospro_start_epoch", "0", "--cospro_warmup_epochs", "0", "--cospro_temperature", "0.25"]
    return {
        "simclr": ["--splice_mode", "none"],
        "cospro": [*relational, "--splice_weight", "0.5"],
        "cospro_kl_only": [*relational, "--splice_weight", "1.0", "--simclr_weight", "0"],
        "frozen_concept_distill": [
            "--splice_mode", "frozen_concept_distill",
            "--concept_transfer_targets", str(inputs / "targets.pt"),
            "--concept_transfer_start_epoch", "0", "--concept_transfer_warmup_epochs", "0",
        ],
        "la_ssl": ["--la_ssl", "--la_ssl_warmup_epochs", "0", "--la_ssl_update_freq", "1"],
    }


def prepare_inputs(root: Path) -> Path:
    write_waterbirds_fixture(root / "datasets")
    inputs = root / "inputs"
    inputs.mkdir()
    sample_ids, y, place = train_sample_ids()
    cache = synthetic_splice_cache(sample_ids, y, place)
    groups = build_concept_groups(cache, GRAPH_CONFIG)
    save_graph_json(build_teacher_graph(cache, groups, GRAPH_CONFIG, device="cpu"), inputs / "teacher_graph.json")
    torch.save(synthetic_target_bank(sample_ids), inputs / "targets.pt")
    return inputs


def run_mode(root: Path, inputs: Path, arm: str, args: list[str]) -> dict:
    run_dir = root / "runs" / arm
    env = {
        **os.environ,
        **({"CUDA_VISIBLE_DEVICES": ""} if DEVICE == "cpu" else {}),
        "WANDB_MODE": "disabled",
        "PYTHONHASHSEED": "0",
        "SPUR_SPLICE_SCRATCH_ROOT": str(root / "scratch"),
        "SPUR_SPLICE_OUTPUT_ROOT": str(root / "outputs"),
        "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    command = [
        sys.executable, "-u", str(PROJECT_ROOT / "spur_splice.py"), *COMMON_ARGS,
        "--data_folder", str(root / "datasets"), "--arm", arm,
        "--artifact_dir", str(run_dir / "training"), "--run_record", str(run_dir / "run.json"), *args,
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(f"{arm} failed:\n{result.stdout[-3000:]}\n{result.stderr[-5000:]}")
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def summarize(record: dict) -> dict:
    """Keep the behaviour-bearing numbers: per-epoch SSL losses and final probe metrics."""

    ssl_events = [event for event in record.get("metrics", []) if event.get("stage") == "ssl"]
    losses = [
        {key: value for key, value in event["values"].items() if "loss" in key.lower() and isinstance(value, float)}
        for event in ssl_events
    ]
    final = {
        key: value for key, value in record.get("final_metrics", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {"status": record.get("status"), "ssl_losses": losses, "final_metrics": final}


def assert_finite(testcase: unittest.TestCase, value, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(testcase, item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(testcase, item, f"{path}[{index}]")
    elif isinstance(value, float):
        testcase.assertTrue(math.isfinite(value), f"non-finite value at {path}")


class GoldenTrainingTests(unittest.TestCase):
    def test_each_training_mode_matches_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = prepare_inputs(root)
            runs = {}
            for arm, args in mode_args(inputs).items():
                with self.subTest(arm=arm):
                    summary = summarize(run_mode(root, inputs, arm, args))
                    self.assertEqual(summary["status"], "complete")
                    self.assertEqual(len(summary["ssl_losses"]), 2, "one SSL metric event per epoch")
                    self.assertTrue(any("worst" in key.lower() for key in summary["final_metrics"]), "WGA recorded")
                    assert_finite(self, summary)
                    runs[arm] = summary

        actual = {"platform": platform.system(), "runs": runs}
        if REPORT:
            Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
            Path(REPORT).write_text(
                json.dumps({**actual, "device": DEVICE, "torch": torch.__version__}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if DEVICE != "cpu":
            self.skipTest("the numeric snapshot covers CPU runs; accelerated modes completed with finite metrics")
        compare_or_update(
            SNAPSHOT, actual,
            lambda expected, current: assert_close_tree(self, expected, current, rel=1e-3, abs_=1e-4),
        )


if __name__ == "__main__":
    unittest.main()
