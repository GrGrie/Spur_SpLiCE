"""Reproduce only the six fixed saved probes required by NEXT_TESTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.tools.run_downstream_evaluator_diagnostic import _find_probe_artifacts, _reproduce_saved_probe


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    validation_failures = []
    unavailable = []
    for entry in config["entries"]:
        root = Path(entry["root"])
        key = f"{entry['name']}/seed{entry['seed']}/{entry['arm']}"
        try:
            result_path, feature_path = _find_probe_artifacts(root / f"seed{entry['seed']}" / entry["arm"] / "training", config["ssl_epoch"])
        except FileNotFoundError as exc:
            record = {"key": key, "status": "unavailable", "reason": str(exc)}
            unavailable.append(key)
            records.append(record)
            continue
        probe_config = {
            **config,
            "dataset": config["dataset"],
            "data_folder": config["data_folder"],
            "train_set_linear_layer": config["train_set_linear_layer"],
            "eval_split": config["eval_split"],
            "batch_size": config["batch_size"],
            "num_classes": config["num_classes"],
        }
        try:
            reproduction = _reproduce_saved_probe(result_path, feature_path, probe_config)
        except Exception as exc:  # retain a precise failure record for cluster debugging
            record = {"key": key, "status": "validation_failed", "error": repr(exc)}
            validation_failures.append(key)
            records.append(record)
            continue
        passed = bool(
            reproduction["matches_existing"]
            and reproduction["alignment"]["passed"]
            and reproduction["convergence"].get("converged", False)
        )
        record = {"key": key, "status": "passed" if passed else "validation_failed", **reproduction}
        records.append(record)
        if not passed:
            validation_failures.append(key)
    report = {
        "artifact": "saved_probe_integrity_v1",
        "protocol": "six_cpu_logistic_probe_fits_only",
        "fit_count": len(records) - len(unavailable),
        "expected_fit_count": len(config["entries"]),
        "records": records,
        "unavailable": unavailable,
        "validation_failures": validation_failures,
        "gate": "continue" if not unavailable and not validation_failures else "stop_before_new_training",
        "passed": not unavailable and not validation_failures and len(records) == len(config["entries"]),
    }
    path = output / "integrity_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"Saved-probe integrity gate failed; see {path}")
    print(f"Saved-probe integrity passed: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
