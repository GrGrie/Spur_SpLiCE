"""Run a seed/arm experiment matrix from one immutable JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from splice.artifacts import PROJECT_ROOT, resolve_output_root, seed_run


EXECUTION_SCHEMA_VERSION = 1


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


def manifest_fingerprint(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def command_for(
    manifest: dict,
    seed: int,
    arm: str,
    *,
    artifact_root: str | Path | None = None,
    output: Path | None = None,
) -> tuple[list[str], Path]:
    if arm not in manifest["arms"]:
        raise ValueError(f"Unknown arm {arm!r}")
    root = resolve_output_root(artifact_root)
    output = output or seed_run(seed, str(manifest["name"]), arm, root=root)
    values = {**manifest["common"], **manifest["arms"][arm].get("args", {})}
    values.update(seed=seed, checkpoint_dir=str(output / "training"))
    substitutions = {
        "project": str(PROJECT_ROOT),
        "artifacts": str(root),
        "seed": seed,
        "arm": arm,
        "output": str(output),
    }
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


def _read_command(path: Path) -> list[str]:
    try:
        command = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read existing command identity at {path}.") from exc
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise RuntimeError(f"Existing command identity at {path} is invalid.")
    return command


def _without_resume(command: list[str]) -> list[str]:
    normalized = list(command)
    while "--resume" in normalized:
        index = normalized.index("--resume")
        del normalized[index : index + 2]
    return normalized


def _require_same_command(output: Path, command: list[str]) -> None:
    existing = _read_command(output / "command.json")
    if _without_resume(existing) != _without_resume(command):
        raise RuntimeError(
            f"Refusing to reuse {output}: its command does not match this manifest execution."
        )


def _statuses(output: Path) -> list[dict]:
    statuses = []
    for path in sorted((output / "training").glob("*/run_status.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read run status at {path}.") from exc
        payload["_path"] = str(path)
        statuses.append(payload)
    return statuses


def _resume_checkpoint(output: Path) -> Path:
    checkpoints = [
        path for path in (output / "training").glob("*/*.pth")
        if path.name != "probe_tmp.pth"
    ]
    if not checkpoints:
        raise RuntimeError(f"Cannot resume {output}: no training checkpoint was found.")

    def checkpoint_rank(path: Path) -> tuple[int, float]:
        if path.name == "last.pth":
            return (sys.maxsize, path.stat().st_mtime)
        if path.stem.startswith("epoch_") and path.stem[6:].isdigit():
            return (int(path.stem[6:]), path.stat().st_mtime)
        return (-1, path.stat().st_mtime)

    return max(checkpoints, key=checkpoint_rank)


def _next_attempt_output(base_output: Path) -> Path:
    attempts = base_output / "attempts"
    number = 1
    while (attempts / f"attempt_{number:04d}").exists():
        number += 1
    return attempts / f"attempt_{number:04d}"


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    manifest: dict,
    seed: int,
    arm: str,
    dry_run: bool = False,
    *,
    existing: str = "error",
    artifact_root: str | Path | None = None,
) -> Path:
    command, base_output = command_for(manifest, seed, arm, artifact_root=artifact_root)
    output = base_output
    populated = output.exists() and any(output.iterdir())

    if existing == "new-attempt":
        output = _next_attempt_output(base_output)
        command, _ = command_for(
            manifest, seed, arm, artifact_root=artifact_root, output=output
        )
    elif populated:
        if existing == "error":
            raise RuntimeError(
                f"Execution directory already exists: {output}. "
                "Choose --existing reuse, resume, or new-attempt explicitly."
            )
        _require_same_command(output, command)
        statuses = _statuses(output)
        complete = [status for status in statuses if status.get("status") == "complete"]
        if existing == "reuse":
            if len(complete) != 1:
                raise RuntimeError(
                    f"Cannot reuse {output}: expected exactly one complete run status, found {len(complete)}."
                )
            print(f"[INFO] Reusing completed execution at {output}")
            return output
        if existing == "resume":
            if complete:
                raise RuntimeError(f"Execution at {output} is complete; use --existing reuse or new-attempt.")
            command.extend(("--resume", str(_resume_checkpoint(output))))
    elif existing in {"reuse", "resume"}:
        raise RuntimeError(f"Cannot {existing} {output}: the execution directory is empty or absent.")

    print(" ".join(command))
    if dry_run:
        return output
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "command.json", command)
    _write_json(
        output / "execution.json",
        {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "manifest_sha256": manifest_fingerprint(manifest),
            "study": str(manifest["name"]),
            "seed": seed,
            "arm": arm,
            "existing_policy": existing,
            "command": command,
        },
    )
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
    parser.add_argument(
        "--existing",
        choices=("error", "reuse", "resume", "new-attempt"),
        default="error",
        help="Explicit policy when this seed/arm execution already exists.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Artifact tree root; overrides SPUR_SPLICE_ARTIFACT_ROOT and outputs/.",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    tasks = matrix(manifest)
    if args.list:
        for task, (seed, arm) in enumerate(tasks):
            print(f"{task:3d}  seed={seed:02d}  arm={arm}")
        return
    if args.task is not None:
        if not 0 <= args.task < len(tasks):
            raise SystemExit(f"task must be in 0..{len(tasks) - 1}")
        seed, arm = tasks[args.task]
    elif args.seed is not None and args.arm:
        seed, arm = args.seed, args.arm
        if (seed, arm) not in tasks:
            raise SystemExit(f"seed={seed}, arm={arm!r} is not in this manifest")
    else:
        parser.error("choose --task or both --seed and --arm")
    run(
        manifest,
        seed,
        arm,
        args.dry_run,
        existing=args.existing,
        artifact_root=args.artifact_root,
    )


if __name__ == "__main__":
    main()
