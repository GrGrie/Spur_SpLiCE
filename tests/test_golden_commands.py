"""Golden snapshot of every command the experiment runner generates from the committed manifests."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from golden_support import compare_or_update
from experiments.runner import command_for, load_manifest, matrix
from splice.artifacts import PROJECT_ROOT

MANIFEST_DIR = PROJECT_ROOT / "experiments" / "manifests"
FIXED_ENV = {
    "DATA_FOLDER": "/data",
    "SPUR_SPLICE_SCRATCH_ROOT": str(PROJECT_ROOT / "_golden_scratch"),
    "SPUR_SPLICE_OUTPUT_ROOT": "",
}


def normalize(command: list[str], output_root: Path) -> list[str]:
    replacements = [
        (sys.executable, "<python>"),
        (str(output_root), "<outputs>"),
        (FIXED_ENV["SPUR_SPLICE_SCRATCH_ROOT"], "<scratch>"),
        (str(PROJECT_ROOT), "<project>"),
    ]
    normalized = []
    for value in command:
        for source, target in replacements:
            value = value.replace(source, target)
        normalized.append(value.replace("\\", "/"))
    return normalized


def generate_commands() -> dict[str, dict[str, list[str]]]:
    output_root = PROJECT_ROOT / "_golden_outputs"
    snapshot: dict[str, dict[str, list[str]]] = {}
    with patch.dict(os.environ, FIXED_ENV):
        for path in sorted(MANIFEST_DIR.glob("*.yaml")):
            manifest = load_manifest(path)
            manifest["_manifest_path"] = str(path)
            entries: dict[str, list[str]] = {}
            for seed, arm in matrix(manifest):
                command, _ = command_for(manifest, seed, arm, output_root=output_root)
                entries[f"seed={seed:02d} arm={arm}"] = normalize(command, output_root)
                if "locked_test" in manifest:
                    command, _ = command_for(manifest, seed, arm, output_root=output_root, locked_test=True)
                    entries[f"seed={seed:02d} arm={arm} locked_test"] = normalize(command, output_root)
            snapshot[path.name] = entries
    return snapshot


class GoldenCommandTests(unittest.TestCase):
    def test_runner_commands_match_snapshot(self):
        def compare(expected, actual):
            self.assertEqual(sorted(expected), sorted(actual), "manifest set changed")
            for manifest_name, entries in expected.items():
                self.assertEqual(sorted(entries), sorted(actual[manifest_name]), f"{manifest_name}: matrix changed")
                for key, command in entries.items():
                    self.assertEqual(command, actual[manifest_name][key], f"{manifest_name} {key}")

        compare_or_update("commands.json", generate_commands(), compare)


if __name__ == "__main__":
    unittest.main()
