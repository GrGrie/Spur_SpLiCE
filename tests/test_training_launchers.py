"""Check Slurm shell adapters without starting training or a cluster job."""

import os
import shlex
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def launch(script, *args):
    env = {**os.environ, "PROJECT_DIR": str(ROOT), "PYTHON_BIN": "/bin/echo"}
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *args],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )


class TrainingLauncherTests(unittest.TestCase):
    def test_standalone_requires_explicit_dataset_and_seed(self):
        result = launch("run_training.sbatch", "--dataset", "celeba")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--seed N", result.stderr)

    def test_standalone_forwards_training_and_checkpoint_options(self):
        result = launch(
            "run_training.sbatch", "--dataset", "celeba", "--seed", "8",
            "--learning_rate", "0.003", "--checkpoint_dir", "/tmp/chosen",
            "--resume", "/tmp/old/last.pth",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertIn("--run_record", command)
        self.assertTrue(command[command.index("--run_record") + 1].endswith("/run.json"))
        self.assertEqual(command[-10:], [
            "--dataset", "celeba", "--seed", "8", "--learning_rate", "0.003",
            "--checkpoint_dir", "/tmp/chosen", "--resume", "/tmp/old/last.pth",
        ])

    def test_matrix_cell_does_not_keep_default_array_task(self):
        result = launch("run_experiment.sbatch", "experiments/manifests/waterbirds_cospro.json",
                        "--seed", "3", "--arm", "cospro", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertNotIn("--task", command)
        self.assertEqual(command[-5:], ["--seed", "3", "--arm", "cospro", "--dry-run"])

    def test_la_ssl_has_a_valid_default_and_accepts_seed(self):
        default = shlex.split(launch("run_la_ssl.sh", "--dry-run").stdout)
        chosen = shlex.split(launch("run_la_ssl.sh", "--seed", "4", "--dry-run").stdout)
        self.assertEqual(default[default.index("--seed") + 1], "1")
        self.assertEqual(chosen.count("--seed"), 1)
        self.assertEqual(chosen[chosen.index("--seed") + 1], "4")

    def test_cache_forwards_hyperparameters(self):
        result = launch("cache_splice_dataset.sh", "waterbirds", "--batch-size", "32",
                        "--splice-l1-penalty", "0.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertEqual(command[-4:], ["--batch-size", "32", "--splice-l1-penalty", "0.1"])

    def test_pipeline_does_not_force_large_model_on_cifar(self):
        result = launch("run_cospro_pipeline.sh", "--dataset", "spur_cifar10", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertNotIn("--model", command)


if __name__ == "__main__":
    unittest.main()
