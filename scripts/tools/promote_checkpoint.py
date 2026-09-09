"""Retain a checkpoint explicitly with a label, reason, and metric snapshot."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from splice.artifacts import binary_destination
from splice.run_recording import RunRecorder


def promote(record_path: str | Path, checkpoint: str | Path, label: str, reason: str, metrics: dict) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", label):
        raise ValueError("label must use lowercase letters, numbers, '.', '_' or '-'")
    recorder = RunRecorder.open(record_path)
    source = Path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(source)
    identity = recorder.data["identity"]
    local_target = Path(record_path).parent / f"promoted_{label}{source.suffix}"
    destination = binary_destination(local_target, source.stat().st_size, kind="checkpoints", identity=identity)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    artifact = recorder.register_artifact(destination, kind="ssl_checkpoint", stage="ssl", retention_state="promoted")
    recorder.record_promotion(
        label=label,
        reason=reason,
        metrics=metrics,
        artifact_sha256=artifact["sha256"],
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_record")
    parser.add_argument("checkpoint")
    parser.add_argument("--label", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--metrics", required=True, help="JSON object with the metric snapshot that justifies promotion")
    args = parser.parse_args()
    metrics = json.loads(args.metrics)
    if not isinstance(metrics, dict) or not metrics:
        parser.error("--metrics must be a non-empty JSON object")
    destination = promote(args.run_record, args.checkpoint, args.label, args.reason, metrics)
    print(destination)


if __name__ == "__main__":
    main()
