"""Run a seed/arm experiment matrix from one immutable YAML (or legacy JSON) manifest."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

from cospro.tracking.artifacts import PROJECT_ROOT, atomic_write_json, make_attempt_id, resolve_output_root, run_directory, scratch_root
from cospro.tracking.run_recording import portable_json


EXECUTION_SCHEMA_VERSION = 1
PRIMARY_ATTEMPT_ID = "primary"
LOCKED_TEST_ATTEMPT_ID = "locked-test"


MANIFEST_SUFFIXES = (".yaml", ".yml", ".json")


def read_manifest_file(path: str | Path) -> dict:
    """Parse a manifest file; YAML is the canonical format and JSON stays readable."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        manifest = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        manifest = json.loads(text)
    else:
        raise ValueError(f"Manifest must end with one of {MANIFEST_SUFFIXES}: {path}")
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must contain a mapping at the top level: {path}")
    return manifest


def load_manifest(path: str | Path) -> dict:
    manifest = read_manifest_file(path)
    required = {"name", "seeds", "common", "arms"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    if not manifest["seeds"] or not manifest["arms"]:
        raise ValueError("Manifest must contain at least one seed and arm.")
    return expand_sweeps(manifest)


_ARM_TOKEN = re.compile(r"^[a-z0-9_.-]+$")


def _sweep_token(value: object) -> str:
    token = ("true" if value else "false") if isinstance(value, bool) else str(value).lower()
    if not _ARM_TOKEN.fullmatch(token):
        raise ValueError(f"Sweep value {value!r} cannot form an arm name; declare that arm explicitly.")
    return token


def expand_sweeps(manifest: dict) -> dict:
    """Expand the optional ``sweeps`` block into one arm per grid point.

    ``sweeps: {prefix: {base: ARM, grid: {option: [values, ...]}}}`` adds arms named
    ``prefix__option-value__option-value`` whose arguments are the base arm's arguments overridden by
    the grid point. Manifests without ``sweeps`` come back unchanged.
    """

    sweeps = manifest.get("sweeps")
    if not sweeps:
        return manifest
    from cospro.config import training_defaults  # deferred: pulls in torch

    options = set(training_defaults())
    arms = dict(manifest["arms"])
    for prefix, sweep in sweeps.items():
        base_name, grid = sweep.get("base"), sweep.get("grid")
        if base_name not in manifest["arms"]:
            raise ValueError(f"Sweep {prefix!r} names an unknown base arm {base_name!r}.")
        if not isinstance(grid, dict) or not grid or not all(isinstance(values, list) and values for values in grid.values()):
            raise ValueError(f"Sweep {prefix!r} needs a grid of non-empty value lists.")
        unknown = sorted(set(grid) - options)
        if unknown:
            raise ValueError(f"Sweep {prefix!r} varies unknown training options: {unknown}.")
        base = manifest["arms"][base_name]
        keys = list(grid)
        for point in itertools.product(*(grid[key] for key in keys)):
            name = "__".join([_sweep_token(prefix)] + [f"{key}-{_sweep_token(value)}" for key, value in zip(keys, point)])
            if name in arms:
                raise ValueError(f"Sweep {prefix!r} generates arm {name!r}, which already exists.")
            arms[name] = {**base, "args": {**base.get("args", {}), **dict(zip(keys, point))}}
    return {**manifest, "arms": arms}


def matrix(manifest: dict) -> list[tuple[int, str]]:
    return [(int(seed), str(arm)) for seed in manifest["seeds"] for arm in manifest["arms"]]


def manifest_fingerprint(manifest: dict) -> str:
    public_manifest = {key: value for key, value in manifest.items() if not key.startswith("_")}
    payload = json.dumps(public_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def command_for(
    manifest: dict,
    seed: int,
    arm: str,
    attempt_id: str | None = None,
    *,
    output_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    output: Path | None = None,
    locked_test: bool = False,
) -> tuple[list[str], Path]:
    """Build a command; ``artifact_root`` remains an alias for old local callers."""

    if arm not in manifest["arms"]:
        raise ValueError(f"Unknown arm {arm!r}")
    if output_root is not None and artifact_root is not None:
        raise ValueError("Pass only one of output_root and artifact_root")
    root = resolve_output_root(output_root if output_root is not None else artifact_root)
    attempt_id = attempt_id or (LOCKED_TEST_ATTEMPT_ID if locked_test else PRIMARY_ATTEMPT_ID)
    output = output or run_directory(seed, str(manifest["name"]), arm, attempt_id, root=root)
    attempt_id = output.name
    values = {**manifest["common"], **manifest["arms"][arm].get("args", {})}
    flags = [*manifest.get("flags", []), *manifest["arms"][arm].get("flags", [])]
    if locked_test:
        protocol = manifest.get("locked_test")
        if not isinstance(protocol, dict):
            raise ValueError("--locked-test requires a predeclared 'locked_test' manifest section.")
        protocol_args = protocol.get("args", {})
        protocol_flags = protocol.get("flags", [])
        if not isinstance(protocol_args, dict) or not isinstance(protocol_flags, list):
            raise ValueError("locked_test args must be an object and flags must be a list.")
        if (
            protocol_args.get("linear_eval_split") != "test"
            or protocol_args.get("linear_probe_mode") != "final"
            or "final_test" not in protocol_flags
        ):
            raise ValueError(
                "locked_test must select the test split, final-only probing, and the final_test guard."
            )
        values.update(protocol_args)
        flags.extend(protocol_flags)
    values.update(seed=seed, checkpoint_dir=str(output / "training"))
    substitutions = {
        "project": str(PROJECT_ROOT), "artifacts": str(root), "seed": seed,
        "arm": arm, "output": str(output),
        "attempt": attempt_id,
        "scratch": str(scratch_root()),
    }
    command = [sys.executable, "-u", str(PROJECT_ROOT / "spur_splice.py")]
    command.extend((
        "--study", str(manifest["name"]), "--arm", arm, "--attempt_id", attempt_id,
        "--run_record", str(output / "run.json"), "--artifact_dir", str(output / "training"),
        "--manifest_path", str(manifest.get("_manifest_path", "")),
    ))
    for flag in flags:
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read existing command identity at {path}.") from exc
    command = payload.get("command") if isinstance(payload, dict) else payload
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise RuntimeError(f"Existing command identity at {path} is invalid.")
    return command


def _comparable(command: list[str]) -> list[str]:
    """Normalize a command for identity checks.

    ``--resume`` depends on the checkpoint found at launch time. ``--manifest_path`` keeps only the
    file stem, so the JSON-to-YAML migration preserves identity; every manifest value is expanded
    into its own flag, so the remaining command still pins the configuration.
    """

    normalized = list(command)
    while "--resume" in normalized:
        index = normalized.index("--resume")
        del normalized[index:index + 2]
    if "--manifest_path" in normalized:
        index = normalized.index("--manifest_path") + 1
        if index < len(normalized) and normalized[index]:
            normalized[index] = Path(normalized[index].replace("\\", "/")).stem
    return normalized


def _require_same_command(output: Path, command: list[str]) -> None:
    existing = portable_json({"command": _read_command(output / "command.json")})["command"]
    requested = portable_json({"command": command})["command"]
    if _comparable(existing) != _comparable(requested):
        raise RuntimeError(f"Refusing to reuse {output}: its command does not match this manifest execution.")


def _is_complete(output: Path) -> bool:
    record = output / "run.json"
    if record.is_file():
        try:
            return json.loads(record.read_text(encoding="utf-8")).get("status") == "complete"
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cannot read run record at {record}.") from exc
    statuses = []
    for path in sorted((output / "training").glob("*/run_status.json")):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read run status at {path}.") from exc
    return len([status for status in statuses if status.get("status") == "complete"]) == 1


def _resume_checkpoint(output: Path) -> Path:
    checkpoints = [path for path in (output / "training").glob("**/*.pth") if path.name != "probe_tmp.pth"]
    if not checkpoints:
        raise RuntimeError(f"Cannot resume {output}: no local training checkpoint was found.")

    def checkpoint_rank(path: Path) -> tuple[int, float]:
        if path.name == "last.pth":
            return (sys.maxsize, path.stat().st_mtime)
        if path.stem.startswith("epoch_") and path.stem[6:].isdigit():
            return (int(path.stem[6:]), path.stat().st_mtime)
        return (-1, path.stat().st_mtime)

    return max(checkpoints, key=checkpoint_rank)


def run(
    manifest: dict,
    seed: int,
    arm: str,
    dry_run: bool = False,
    *,
    existing: str = "error",
    output_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    attempt_id: str | None = None,
    locked_test: bool = False,
) -> Path:
    if existing == "new-attempt" and attempt_id is None:
        attempt_id = make_attempt_id()
    attempt_id = attempt_id or (LOCKED_TEST_ATTEMPT_ID if locked_test else PRIMARY_ATTEMPT_ID)
    command, output = command_for(
        manifest, seed, arm, attempt_id, output_root=output_root, artifact_root=artifact_root,
        locked_test=locked_test,
    )
    populated = output.exists() and any(output.iterdir())
    if populated:
        if existing in {"error", "new-attempt"}:
            raise RuntimeError(
                f"Execution directory already exists: {output}. Choose a new attempt ID or an explicit reuse/resume policy."
            )
        _require_same_command(output, command)
        complete = _is_complete(output)
        if existing == "reuse":
            if not complete:
                raise RuntimeError(f"Cannot reuse {output}: the run is not complete.")
            print(f"[INFO] Reusing completed execution at {output}")
            return output
        if complete:
            raise RuntimeError(f"Execution at {output} is complete; use --existing reuse or new-attempt.")
        command.extend(("--resume", str(_resume_checkpoint(output))))
    elif existing in {"reuse", "resume"}:
        raise RuntimeError(f"Cannot {existing} {output}: the execution directory is empty or absent.")

    print(f"[results] Attempt directory: {output} (run.json, command.json, execution.json)", flush=True)
    print(" ".join(command))
    if dry_run:
        return output
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "command.json", portable_json({"schema": "experiment-command-v1", "command": command}))
    atomic_write_json(output / "execution.json", portable_json({
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "manifest_sha256": manifest_fingerprint(manifest), "study": str(manifest["name"]),
        "seed": seed, "arm": arm, "attempt_id": attempt_id, "existing_policy": existing,
        "locked_test": locked_test,
        "command": command,
    }))
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
    parser.add_argument("--existing", choices=("error", "reuse", "resume", "new-attempt"), default="error")
    parser.add_argument("--attempt-id", help="Stable run attempt to reuse or resume; defaults to 'primary'.")
    parser.add_argument(
        "--locked-test",
        action="store_true",
        help="Apply the manifest's predeclared final-only held-out test protocol.",
    )
    parser.add_argument(
        "--output-root", "--artifact-root", dest="output_root", type=Path,
        help="Git-facing results tree; overrides SPUR_SPLICE_OUTPUT_ROOT and outputs/.",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    manifest["_manifest_path"] = str(Path(args.manifest).resolve())
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
        manifest, seed, arm, args.dry_run, existing=args.existing,
        output_root=args.output_root, attempt_id=args.attempt_id, locked_test=args.locked_test,
    )


if __name__ == "__main__":
    main()
