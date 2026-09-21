"""Pin the run storage names that spur_splice.py derives from its configuration.

The storage name selects the checkpoint folder. A resumed run must land in the folder that already
holds its checkpoints, so renaming configuration fields keeps every storage name stable.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import spur_splice
from cospro.tracking.artifacts import PROJECT_ROOT

GRAPH = "tests/fixtures/storage_identity_graph.json"
COMMON = ["--dataset", "waterbirds", "--device", "cpu", "--seed", "3", "--epochs", "500", "--batch_size", "128"]
EXPECTED = {
    "simclr": (["--splice_mode", "none"], "waterbirds_s3_base_e500_343cde76bf"),
    "cospro": (
        ["--splice_mode", "cospro_relational", "--splice_weight", "0.5", "--cospro_teacher_graph", GRAPH,
         "--cospro_temperature", "0.25"],
        "waterbirds_s3_cospro-relational_e500_b0ac17d47b",
    ),
    "legacy_crp_flags": (
        ["--splice_mode", "crp_relational", "--splice_weight", "0.5", "--crp_teacher_graph", GRAPH,
         "--crp_temperature", "0.25"],
        "waterbirds_s3_crp-v2-relational_e500_607f2041f8",
    ),
}


def storage_name(extra: list[str]) -> str:
    with tempfile.TemporaryDirectory() as directory:
        argv = ["spur_splice.py", *COMMON, *extra, "--artifact_dir", directory]
        cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            with patch.object(sys, "argv", argv):
                return spur_splice.parse_args().storage_name
        finally:
            os.chdir(cwd)


class StorageIdentityTests(unittest.TestCase):
    def test_storage_names_are_stable(self):
        for label, (extra, expected) in EXPECTED.items():
            with self.subTest(configuration=label):
                self.assertEqual(storage_name(extra), expected)


if __name__ == "__main__":
    unittest.main()
