"""Golden snapshot of how command lines resolve into the flat training configuration.

The flat namespace produced by ``spur_splice.parse_args`` feeds the storage-name hash, args.json,
run.json, the W&B config and checkpoint options. This test pins it for every runner command of every
manifest, for standalone variants and for invalid argument sets, together with the pipeline and
linear-probe defaults. Configuration refactors must leave this snapshot unchanged.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import spur_splice
from experiments.runner import command_for, load_manifest, matrix
from cospro.cli import linear_probe
from golden_support import assert_close_tree, compare_or_update, synthetic_target_bank
from cospro.cli import run_cospro_pipeline
from cospro.tracking.artifacts import PROJECT_ROOT

GRAPHS = ("crp_graph.json", "cospro_laion_graph.json", "cospro_laion_gated_graph.json", "raw_clip_graph.json",
          "semantic_splice_graph.json")
METASHIFT_GRAPHS = ("cospro_graph.json", "cospro_laion_graph.json", "cospro_laion_gated_graph.json",
                    "raw_clip_graph.json")
CIFAR_GRAPHS = ("cospro_laion_graph.json",)
TARGET_BANK = "legacy_archives/Spur_SpLiCE/next_actions_after_transfer_2026-09-07/direct_transfer/targets_v1.pt"
VOLATILE = {"runtime_versions", "storage_name", "save_folder"}
STANDALONE = {
    "defaults": [],
    "cifar": ["--dataset", "spur_cifar10"],
    "celeba_alias": ["--dataset", "celebA", "--seed", "4"],
    "la_ssl": ["--la_ssl", "--la_ssl_eta", "0.2"],
    "final_test": ["--final_test", "--linear_eval_split", "test", "--linear_probe_mode", "final"],
    "boolean_spellings": ["--amp", "false", "--channels_last", "--cudnn_enabled", "0", "--keep_checkpoints"],
    "cosine_warm": ["--cosine", "--warm", "--batch_size", "512", "--lr_decay_epochs", "300,400"],
    "legacy_crp_flags": ["--splice_mode", "crp_relational", "--crp_teacher_graph", "{graphs}/crp_graph.json",
                         "--crp_temperature", "0.2", "--crp_start_epoch", "5", "--splice_weight", "0.3"],
    "kl_only": ["--splice_mode", "cospro_relational", "--cospro_teacher_graph", "{graphs}/crp_graph.json",
                "--simclr_weight", "0", "--splice_weight", "1", "--cospro_decay_start_epoch", "100",
                "--cospro_decay_end_epoch", "200"],
    "periodic_probe_off": ["--linear_probe_mode", "none", "--rank_eval_freq", "0"],
}
INVALID = {
    "negative_epochs": ["--epochs", "0"],
    "relational_without_graph": ["--splice_mode", "cospro_relational"],
    "kl_only_without_relational": ["--simclr_weight", "0"],
    "la_ssl_with_teacher": ["--la_ssl", "--splice_mode", "cospro_relational"],
    "cifar_large_model": ["--dataset", "spur_cifar10", "--model", "resnet18_large"],
    "cudnn_benchmark": ["--cudnn_benchmark", "true"],
    "bad_crop": ["--ssl_crop_min", "1.5"],
    "decay_half_set": ["--cospro_decay_start_epoch", "10"],
    "test_without_lock": ["--linear_eval_split", "test"],
    "keep_count": ["--checkpoint_keep_count", "3"],
}


class ResolutionFixture:
    """Temporary output and scratch roots with the graphs and the target bank the manifests name."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.outputs = root / "outputs"
        self.scratch = root / "scratch"
        self.graphs = self.outputs / "shared" / "waterbirds" / "graphs"
        self.graphs.mkdir(parents=True)
        for name in GRAPHS:
            shutil.copyfile(PROJECT_ROOT / "outputs" / "shared" / "waterbirds" / "graphs" / name, self.graphs / name)
        # Resolution reads a teacher graph's bytes for the fingerprint only, so one real graph
        # stands in for every graph the MetaShift study names.
        for dataset, names in (("metashift", METASHIFT_GRAPHS), ("spur_cifar10", CIFAR_GRAPHS)):
            directory = self.outputs / "shared" / dataset / "graphs"
            directory.mkdir(parents=True)
            for name in names:
                shutil.copyfile(self.graphs / "crp_graph.json", directory / name)
        bank = self.scratch / TARGET_BANK
        bank.parent.mkdir(parents=True)
        torch.save(synthetic_target_bank([f"waterbirds:{index}" for index in range(8)]), bank)
        self.environment = {
            "SPUR_SPLICE_SCRATCH_ROOT": str(self.scratch),
            "SPUR_SPLICE_OUTPUT_ROOT": "",
            "DATA_FOLDER": "/data",
            "WANDB_ENTITY": "",
        }

    def normalize(self, value):
        if isinstance(value, dict):
            return {key: self.normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [self.normalize(item) for item in value]
        if isinstance(value, str):
            for source, target in ((self.outputs, "<outputs>"), (self.scratch, "<scratch>"), (self.root, "<tmp>"),
                                   (PROJECT_ROOT, "<project>")):
                value = value.replace(str(source), target).replace(source.as_posix(), target)
            return value.replace("\\", "/")
        return value


def resolve(argv: list[str]) -> dict:
    with patch.object(sys, "argv", ["spur_splice.py", *argv]), patch("torch.cuda.is_available", return_value=False):
        args = spur_splice.parse_args()
    return {key: ("<volatile>" if key in VOLATILE else value) for key, value in vars(args).items()}


def resolution_error(argv: list[str]) -> str:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), patch.object(sys, "argv", ["spur_splice.py", *argv]), \
            patch("torch.cuda.is_available", return_value=False):
        try:
            spur_splice.parse_args()
        except SystemExit:
            return stderr.getvalue().strip().splitlines()[-1]
    return "<accepted>"


def build_snapshot() -> dict:
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}):
        fixture = ResolutionFixture(Path(directory))
        os.environ.update(fixture.environment)
        os.environ.pop("WANDB_ENTITY")
        artifact_dir = str(Path(directory) / "standalone")
        snapshot: dict = {"manifest_commands": {}, "standalone": {}, "invalid": {}}

        for path in sorted((PROJECT_ROOT / "experiments" / "manifests").glob("*.yaml")):
            manifest = load_manifest(path)
            manifest["_manifest_path"] = str(path)
            for seed, arm in matrix(manifest):
                variants = [False, True] if "locked_test" in manifest else [False]
                for locked in variants:
                    command, _ = command_for(manifest, seed, arm, output_root=fixture.outputs, locked_test=locked)
                    key = f"{path.stem} seed={seed:02d} arm={arm}" + (" locked_test" if locked else "")
                    snapshot["manifest_commands"][key] = fixture.normalize(resolve(command[3:]))

        for name, argv in STANDALONE.items():
            argv = [value.replace("{graphs}", str(fixture.graphs)) for value in argv]
            snapshot["standalone"][name] = fixture.normalize(resolve([*argv, "--artifact_dir", artifact_dir]))
        for name, argv in INVALID.items():
            snapshot["invalid"][name] = resolution_error([*argv, "--artifact_dir", artifact_dir])

        with patch("torch.cuda.is_available", return_value=False):
            pipeline = vars(run_cospro_pipeline.parse_args(["--data-folder", "/data"]))
            with patch.object(sys, "argv", ["linear_probe.py"]):
                probe = vars(linear_probe.parse_args())
            probe_normalized = vars(linear_probe.normalize_args(__import__("argparse").Namespace()))
        snapshot["pipeline_defaults"] = fixture.normalize({key: str(value) for key, value in pipeline.items() if key != "python"})
        snapshot["probe_cli_defaults"] = fixture.normalize(probe)
        snapshot["probe_normalized_defaults"] = fixture.normalize(probe_normalized)
        return snapshot


class ConfigResolutionTests(unittest.TestCase):
    def test_command_lines_resolve_to_the_pinned_configuration(self):
        compare_or_update(
            "resolved_configs.json", build_snapshot(),
            lambda expected, actual: assert_close_tree(self, expected, actual, rel=1e-12, abs_=0.0),
        )


if __name__ == "__main__":
    unittest.main()
