"""Check every Slurm launcher header against the cluster resource rules in AGENTS.md.

One V100 per GPU job, at most 5 CPUs, at most 8G of memory per requested CPU and logs under
outputs/SLURM/. A new launcher is checked automatically once it carries #SBATCH lines.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_CPUS = 5
MAX_MEMORY_GB_PER_CPU = 8
LAUNCHER_GLOBS = ("scripts/*.sbatch", "scripts/*.sh")
SBATCH_LINE = re.compile(r"^#SBATCH\s+--([a-z-]+)(?:[=\s]+(\S+))?")


def sbatch_options(path: Path) -> dict[str, str]:
    options = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SBATCH_LINE.match(line.strip())
        if match:
            options[match.group(1)] = match.group(2) or ""
    return options


def memory_in_gb(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)B?", value.upper())
    if match is None:
        raise ValueError(f"unparseable --mem value {value!r}")
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "": 1 / 1024, "G": 1, "T": 1024}[match.group(2)]
    return float(match.group(1)) * scale


def launchers() -> list[Path]:
    paths = sorted({path for pattern in LAUNCHER_GLOBS for path in ROOT.glob(pattern)})
    return [path for path in paths if sbatch_options(path)]


class SlurmLauncherTests(unittest.TestCase):
    def test_launchers_exist(self):
        self.assertGreater(len(launchers()), 0)

    def test_resource_requests_follow_the_cluster_rules(self):
        for path in launchers():
            options = sbatch_options(path)
            with self.subTest(launcher=path.name):
                cpus = int(options.get("cpus-per-task", "1"))
                self.assertLessEqual(cpus, MAX_CPUS, "at most 5 CPUs per job")
                self.assertIn("mem", options, "every launcher states its memory")
                self.assertLessEqual(
                    memory_in_gb(options["mem"]), MAX_MEMORY_GB_PER_CPU * cpus,
                    f"memory limit is {MAX_MEMORY_GB_PER_CPU}G per CPU",
                )
                gres = options.get("gres")
                if gres is not None:
                    self.assertRegex(gres, r"^gpu(:v100)?:1$", "GPU jobs request exactly one V100")
                for stream in ("output", "error"):
                    self.assertTrue(
                        options.get(stream, "").startswith("outputs/SLURM/"),
                        f"--{stream} writes under outputs/SLURM/",
                    )


if __name__ == "__main__":
    unittest.main()
