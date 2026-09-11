"""Complete the two existing projection controls for one missing student seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.runner import command_for, load_manifest, run
from splice.artifacts import PROJECT_ROOT


MANIFESTS = ("waterbirds_semantic_completion.json", "waterbirds_direct_completion.json")
TARGET_SHA256 = "fde777c311169e14ac2381a418dd2d68434a0d5e923f7f2a77edc59501273123"
SEMANTIC_FINGERPRINT = "eab4f961781f8e8e9dcc7b4783969bbd"


def validate_inputs(command):
    if "--concept_transfer_targets" in command:
        path = Path(command[command.index("--concept_transfer_targets") + 1])
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _sha256(handle)
        if actual != TARGET_SHA256:
            raise ValueError(f"Target bank differs from the seed-1/3 artifact: {path}")
    else:
        path = Path(command[command.index("--cospro_teacher_graph") + 1])
        # Packaging added a trailing newline (and Windows uses CRLF). The
        # original indent=2/no-trailing-newline serialization reproduces the
        # historical fingerprint exactly, without changing any graph values.
        original_bytes = json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True).encode()
        if hashlib.blake2b(original_bytes, digest_size=16).hexdigest() != SEMANTIC_FINGERPRINT:
            raise ValueError(f"Semantic graph differs from the seed-1/3 artifact: {path}")


def _sha256(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=[2, 4])
    parser.add_argument("--targets", type=Path, help="Location of the original, checksum-verified targets_v1.pt.")
    parser.add_argument("--attempt-id", default="primary")
    parser.add_argument("--existing", choices=["error", "reuse", "resume"], default="error")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = []
    for name in MANIFESTS:
        path = PROJECT_ROOT / "experiments" / "manifests" / name
        manifest = load_manifest(path)
        manifest["_manifest_path"] = str(path)
        arm = next(iter(manifest["arms"]))
        if args.targets and arm == "splice_reconstruction":
            manifest["arms"][arm]["args"]["concept_transfer_targets"] = str(args.targets.resolve())
        command, _ = command_for(manifest, args.seed, arm, args.attempt_id, output_root=args.output_root)
        if not args.dry_run:
            validate_inputs(command)
        tasks.append((manifest, arm))
    for manifest, arm in tasks:
        run(manifest, args.seed, arm, args.dry_run, existing=args.existing,
            output_root=args.output_root, attempt_id=args.attempt_id)


if __name__ == "__main__":
    main()
