"""Check Slurm shell adapters without starting training or a cluster job."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_bash() -> str | None:
    """Return a POSIX bash. On Windows the System32 bash is a WSL relay, so Git Bash is used."""
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
        return str(git_bash) if git_bash.is_file() else None
    return shutil.which("bash")


BASH = find_bash()


def launch(script, *args, extra_env=None):
    """Run a launcher in its local branch with ``echo`` standing in for Python.

    SLURM_* variables are dropped so the tests behave the same inside a Slurm job, and an empty
    output root keeps graphs that exist under outputs/ out of the automatic graph selection.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    with tempfile.TemporaryDirectory() as empty_output_root:
        env.update({
            "PROJECT_DIR": str(ROOT),
            "PYTHON_BIN": "/bin/echo",
            "SPUR_SPLICE_OUTPUT_ROOT": Path(empty_output_root).as_posix(),
        })
        env.update(extra_env or {})
        return subprocess.run(
            [BASH, str(ROOT / "scripts" / script), *args],
            cwd=ROOT, env=env, text=True, capture_output=True,
        )


@unittest.skipIf(BASH is None, "a POSIX bash is required to exercise the Slurm launchers")
class TrainingLauncherTests(unittest.TestCase):
    def test_standalone_auto_selects_the_only_dataset_teacher_graph(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            graph = (
                Path(temporary_directory) / "shared" / "celeba" / "graphs"
                / "groups" / "teacher_graph.json"
            )
            graph.parent.mkdir(parents=True)
            graph.write_text("{}", encoding="utf-8")
            result = launch(
                "run_training.sbatch", "--dataset", "celebA", "--seed", "1",
                extra_env={"SPUR_SPLICE_OUTPUT_ROOT": Path(temporary_directory).as_posix()},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertEqual(command[command.index("--splice_mode") + 1], "cospro_relational")
        self.assertEqual(command[command.index("--cospro_teacher_graph") + 1], graph.as_posix())
        self.assertEqual(command[-4:], ["--dataset", "celebA", "--seed", "1"])

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
        result = launch("run_experiment.sbatch", "experiments/manifests/waterbirds_cospro.yaml",
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
                        "--splice-l1-penalty", "0.1", extra_env={"DATA_FOLDER": "/data"})
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertEqual(command[-4:], ["--batch-size", "32", "--splice-l1-penalty", "0.1"])

    def test_cache_requires_a_dataset_root(self):
        env = {key: value for key, value in os.environ.items() if key != "DATA_FOLDER"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            result = launch("cache_splice_dataset.sh", "waterbirds")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DATA_FOLDER", result.stderr)

    def test_pipeline_forwards_only_the_variables_that_are_set(self):
        result = launch("run_cospro_pipeline.sh", "--dry-run", extra_env={"EPOCHS": "10", "AMP": "0"})
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertEqual(command[command.index("--epochs") + 1], "10")
        self.assertIn("--no-amp", command)
        self.assertNotIn("--batch-size", command)

    def test_pipeline_does_not_force_large_model_on_cifar(self):
        result = launch("run_cospro_pipeline.sh", "--dataset", "spur_cifar10", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertNotIn("--model", command)


if __name__ == "__main__":
    unittest.main()
