"""Run a seed/arm experiment matrix from one immutable JSON manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from splice.artifacts import PROJECT_ROOT, seed_run


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"name", "seeds", "common", "arms"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    if not manifest["seeds"] or not manifest["arms"]:
        raise ValueError("Manifest must contain at least one seed and arm.")
    return manifest


def matrix(manifest: dict) -> list[tuple[int, str]]:
    return [(int(seed), str(arm)) for seed in manifest["seeds"] for arm in manifest["arms"]]


def command_for(manifest: dict, seed: int, arm: str) -> tuple[list[str], Path]:
    if arm not in manifest["arms"]:
        raise ValueError(f"Unknown arm {arm!r}")
    output = seed_run(seed, str(manifest["name"]), arm)
    values = {**manifest["common"], **manifest["arms"][arm].get("args", {})}
    values.update(seed=seed, checkpoint_dir=str(output / "training"))
    substitutions = {"project": str(PROJECT_ROOT), "seed": seed, "arm": arm, "output": str(output)}
    command = [sys.executable, "-u", str(PROJECT_ROOT / "spur_splice.py")]
    for flag in [*manifest.get("flags", []), *manifest["arms"][arm].get("flags", [])]:
        command.append(f"--{flag}")
    for key, value in values.items():
        if isinstance(value, str):
            for token, replacement in substitutions.items():
                value = value.replace("{" + token + "}", str(replacement))
            value = os.path.expandvars(value)
        command.extend((f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)))
    return command, output


def run(manifest: dict, seed: int, arm: str, dry_run: bool = False) -> Path:
    command, output = command_for(manifest, seed, arm)
    print(" ".join(command))
    if dry_run:
        return output
    output.mkdir(parents=True, exist_ok=True)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", type=int)
    selection.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    tasks = matrix(manifest)
    if args.list:
        for task, (seed, arm) in enumerate(tasks):
            print(f"{task:3d}  seed={seed:02d}  arm={arm}")
        return
    if args.task is not None:
        try:
            seed, arm = tasks[args.task]
        except IndexError as exc:
            raise SystemExit(f"task must be in 0..{len(tasks) - 1}") from exc
    elif args.seed is not None and args.arm:
        seed, arm = args.seed, args.arm
        if (seed, arm) not in tasks:
            raise SystemExit(f"seed={seed}, arm={arm!r} is not in this manifest")
    else:
        parser.error("choose --task or both --seed and --arm")
    run(manifest, seed, arm, args.dry_run)


if __name__ == "__main__":
    main()
