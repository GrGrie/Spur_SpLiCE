"""Freeze, evaluate and summarize existing Waterbirds control checkpoints (no SSL)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import statistics

from splice.artifacts import PROJECT_ROOT, atomic_write_json, resolve_output_root, scratch_root, sha256_file

STUDIES = {
    "semantic_splice": "next_actions_after_transfer_2026_09_07_graph_ablation",
    "splice_reconstruction": "next_actions_after_transfer_2026_09_07_direct_transfer",
}
RUN_IDS = {
    "semantic_splice": {1: "1hc80m2n", 2: "dc6lr4cp", 3: "zpm7odmj", 4: "yax3mt1q"},
    "splice_reconstruction": {1: "35o8ztla", 2: "ixma3dec", 3: "iywcfbry", 4: "lo48apo0"},
}
CORE = ("simclr", "crp_sampler_only", "raw_clip_sampler_only", "raw_clip_kl", "splice_crp_kl")
PROTOCOL = dict(dataset="waterbirds", model="resnet18_large", head="mlp", batch_size=128,
                num_workers=4, epochs=100, ssl_epoch=500, probe_solver="logistic", probe_l2=.001,
                probe_tolerance=1e-6, probe_max_epochs=200, train_set_linear_layer="ds_train",
                eval_split="test", final_test=True, spurious_probe=False, use_wandb=False)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def code_hash():
    paths = sorted((PROJECT_ROOT / "experiments/spurious_eval").rglob("*.py"))
    paths += [Path(__file__), PROJECT_ROOT / "splice/artifacts.py"]
    return hashlib.sha256("".join(f"{p.relative_to(PROJECT_ROOT)}:{sha256_file(p)}\n" for p in paths).encode()).hexdigest()


def relocated(value):
    if value.startswith("artifact://"):
        return scratch_root() / value.removeprefix("artifact://")
    if value.startswith("project://"):
        return PROJECT_ROOT / value.removeprefix("project://")
    prefix = "/scratch/xar68reb/CoSpRo/"
    return scratch_root() / value[len(prefix):] if value.startswith(prefix) else Path(value)


def discover(root, overrides=None):
    """Select the already reported runs by identity, never by test/validation score."""
    overrides = overrides or {}
    rows = []
    for arm, study in STUDIES.items():
        archived = read(root / "reports" / study / "results.json")["runs"]
        for seed in range(1, 5):
            expected_id = RUN_IDS[arm][seed]
            paths = sorted((root / "seeds" / study / f"seed_{seed:02d}" / arm).glob("*/run.json"))
            candidates = [read(p) for p in paths] + archived
            matching = [r for r in candidates if r.get("wandb", {}).get("id") == expected_id
                        and r.get("status") == "complete"]
            artifacts = [a for r in matching for a in r.get("artifacts", [])
                         if a.get("kind") == "ssl_checkpoint" and a.get("retention_state") == "final"]
            unique = {(a.get("uri") or a["storage_path"], a.get("sha256")) for a in artifacts}
            if len(unique) > 1:
                raise ValueError(f"Ambiguous final checkpoint for {arm}/{seed}: {unique}")
            expected_hash = None
            if unique:
                value, expected_hash = next(iter(unique))
                path = relocated(value)
            else:
                # Seeds 2/4 use the canonical scratch policy. prepare verifies the file and options.
                path = scratch_root() / "checkpoints/Spur_SpLiCE" / study / f"seed_{seed:02d}" / arm / "primary/last.pth"
            key = f"{arm}:{seed}"
            path = Path(overrides.get(key, path)).expanduser().resolve()
            rows.append(dict(arm=arm, seed=seed, wandb_id=expected_id, checkpoint=str(path),
                             expected_sha256=expected_hash, exists=path.is_file()))
    unknown = set(overrides) - {f"{r['arm']}:{r['seed']}" for r in rows}
    if unknown:
        raise ValueError(f"Unknown checkpoint override keys: {unknown}")
    return rows


def validate_checkpoint(path, row):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    options = payload.get("opt", {})
    options = options if isinstance(options, dict) else vars(options)
    expected = {"seed": row["seed"], "model": "resnet18_large", "dataset": "waterbirds",
                "epochs": 500, "temp": .05, "batch_size": 128}
    if payload.get("epoch") != 500:
        raise ValueError(f"Expected epoch 500: {path}")
    for key, value in expected.items():
        if options.get(key) != value:
            raise ValueError(f"Checkpoint {path}: {key}={options.get(key)!r}, expected {value!r}")
    if not any(k.replace("module.", "").startswith("encoder.") for k in payload.get("model", {})):
        raise ValueError(f"Missing encoder weights: {path}")
    if row["arm"] == "splice_reconstruction":
        if options.get("concept_transfer_target_kind") != "reconstruction" or options.get("concept_transfer_alpha_max") != .1:
            raise ValueError("Checkpoint is not the reported reconstruction control")
    elif options.get("splice_weight") != .5:
        raise ValueError("Semantic graph checkpoint must use KL weight 0.5")


def validate_result(payload):
    if payload.get("eval_split") != "test" or payload.get("ssl_epoch") != 500:
        raise ValueError("Expected an epoch-500 test result, not validation")
    if payload.get("train_split") != "ds_train" or payload.get("solver") != "logistic":
        raise ValueError("Unexpected probe protocol")
    convergence = payload.get("convergence", {})
    if convergence.get("converged") is not True or convergence.get("l2") != .001 or convergence.get("tolerance") != 1e-6:
        raise ValueError("Probe must converge with the locked L2 and tolerance")
    metrics = payload["metrics"]
    if metrics["Linear train group counts"] != [56] * 4 or metrics["Linear val group counts"] != [2255, 2255, 642, 642]:
        raise ValueError("Unexpected Waterbirds training/test group counts")
    result = dict(avg=metrics["Average over last 10 linear val acc"],
                  wga=metrics["Average over last 10 linear val worst-group acc"],
                  groups=metrics["Linear val group accuracies"])
    if len(result["groups"]) != 4 or any(not math.isfinite(v) or not 0 <= v <= 100
                                         for v in (result["avg"], result["wga"], *result["groups"])):
        raise ValueError("Invalid accuracy values")
    return result


def prepare(args):
    rows = discover(args.artifact_root, read(args.checkpoints_json) if args.checkpoints_json else None)
    if args.dry_run:
        print(json.dumps(dict(protocol=PROTOCOL, rows=rows), indent=2))
        return
    if (args.output_dir / "lock.json").exists():
        raise FileExistsError("A lock already exists; reuse it or select a new output directory")
    for row in rows:
        if not row["expected_sha256"]:
            raise ValueError(f"Missing checkpoint attestation for {row['wandb_id']}; sync its completed run.json before preparation")
        if not row["exists"]:
            raise FileNotFoundError(f"Missing checkpoint: {row['checkpoint']}. Sync run records or supply --checkpoints-json.")
        digest = sha256_file(row["checkpoint"])
        if row["expected_sha256"] and digest != row["expected_sha256"]:
            raise ValueError(f"Archived checkpoint hash mismatch: {row['checkpoint']}")
        validate_checkpoint(row["checkpoint"], row)
        row["sha256"] = digest
    from experiments.spurious_eval.datasets.waterbirds import WaterbirdsDataset
    metadata = Path(WaterbirdsDataset._find_data_dir(Path(args.data_folder))) / "metadata.csv"
    # Freeze the reused baseline probes too, so aggregation cannot silently switch evidence.
    core = []
    for seed in range(1, 5):
        for arm in CORE:
            path = args.artifact_root / "reports/paper_evidence/final_test/probes" / f"seed_{seed:02d}" / f"{arm}.json"
            validate_result(read(path))
            core.append(dict(seed=seed, arm=arm, path=str(path.resolve()), sha256=sha256_file(path)))
    lock = dict(schema="submission-checkpoint-test-v1", created_at=now(), protocol=PROTOCOL,
                code_sha256=code_hash(), data_folder=str(Path(args.data_folder).resolve()),
                metadata=str(metadata.resolve()), metadata_sha256=sha256_file(metadata), rows=rows, core=core)
    atomic_write_json(args.output_dir / "lock.json", lock)
    print(f"Frozen 8 checkpoints and 20 existing test probes: {args.output_dir / 'lock.json'}")


def execute(args):
    lock_path = args.output_dir / "lock.json"
    lock = read(lock_path)
    if lock["protocol"] != PROTOCOL or lock["code_sha256"] != code_hash():
        raise ValueError("Evaluation code/protocol changed since prepare; create a new plan")
    if sha256_file(lock["metadata"]) != lock["metadata_sha256"]:
        raise ValueError("Dataset metadata changed since prepare")
    from experiments.spurious_eval.linear_probe import main as probe
    for row in lock["rows"]:
        if args.seed is not None and row["seed"] != args.seed:
            continue
        if sha256_file(row["checkpoint"]) != row["sha256"]:
            raise ValueError(f"Checkpoint changed: {row['checkpoint']}")
        destination = args.output_dir / f"seed_{row['seed']:02d}" / row["arm"]
        result = destination / "probe_features_epoch_500_ds_train_test.json"
        receipt = destination / "receipt.json"
        if receipt.exists():
            saved = read(receipt)
            if saved["lock_sha256"] != sha256_file(lock_path) or saved["result_sha256"] != sha256_file(result):
                raise ValueError(f"Existing result/lock mismatch: {destination}")
            validate_result(read(result))
            print(f"Reusing completed evaluation: {destination}")
            continue
        # An incomplete attempt can be rerun; it never changes the checkpoint or lock.
        destination.mkdir(parents=True, exist_ok=True)
        options = argparse.Namespace(**PROTOCOL, seed=row["seed"], ckpt=row["checkpoint"],
                                     artifact_dir=str(destination), data_folder=lock["data_folder"],
                                     device=args.device, study="submission_checkpoint_test",
                                     arm=row["arm"], attempt_id=sha256_file(lock_path)[:16])
        probe(options)
        validate_result(read(result))
        atomic_write_json(receipt, dict(completed_at=now(), lock_sha256=sha256_file(lock_path),
                                       checkpoint_sha256=row["sha256"], result_sha256=sha256_file(result)))


def summarize(rows):
    indexed = {(r["arm"], r["seed"]): r for r in rows}
    expected = {(arm, seed) for arm in (*CORE, *STUDIES) for seed in range(1, 5)}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Need exactly four seeds per arm; incomplete/duplicate matrices cannot be summarized")
    summary = {}
    for arm in (*CORE, *STUDIES):
        rr = [indexed[arm, seed] for seed in range(1, 5)]
        summary[arm] = {metric: dict(mean=statistics.mean(r[metric] for r in rr),
                                    sd=statistics.stdev(r[metric] for r in rr)) for metric in ("avg", "wga")}
    paired = {arm: [dict(seed=seed, **{metric: indexed["splice_crp_kl", seed][metric] - indexed[arm, seed][metric]
                                     for metric in ("avg", "wga")}) for seed in range(1, 5)]
              for arm in (*CORE, *STUDIES) if arm != "splice_crp_kl"}
    return dict(rows=rows, summary=summary, paired_cospro_minus_control=paired)


def collect(args):
    lock_path = args.output_dir / "lock.json"
    lock = read(lock_path)
    rows = []
    for record in lock["core"]:
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("Historical test evidence changed")
        rows.append(dict(seed=record["seed"], arm=record["arm"], **validate_result(read(record["path"]))))
    for record in lock["rows"]:
        directory = args.output_dir / f"seed_{record['seed']:02d}" / record["arm"]
        result = directory / "probe_features_epoch_500_ds_train_test.json"
        receipt = read(directory / "receipt.json")
        if receipt["lock_sha256"] != sha256_file(lock_path) or receipt["result_sha256"] != sha256_file(result):
            raise ValueError("Evaluation result integrity mismatch")
        rows.append(dict(seed=record["seed"], arm=record["arm"], **validate_result(read(result))))
    report = summarize(rows)
    report.update(eval_split="test", ssl_epoch=500, lock_sha256=sha256_file(lock_path))
    atomic_write_json(args.output_dir / "results.json", report)
    lines = ["# Held-out Waterbirds comparison", "", "Mean ± sample SD across four student seeds; percentages.", "",
             "| Method | Avg. | WGA |", "|---|---:|---:|"]
    tex = [r"\begin{tabular}{lrr}", r"\toprule Method & Avg. & WGA \\", r"\midrule"]
    for arm, stats in report["summary"].items():
        values = [f"{stats[m]['mean']:.2f} ± {stats[m]['sd']:.2f}" for m in ("avg", "wga")]
        lines.append(f"| {arm} | {' | '.join(values)} |")
        cells = ["$" + v.replace("\u00b1", r"\pm") + "$" for v in values]
        tex.append(arm.replace("_", r"\_") + " & " + " & ".join(cells) + " " + chr(92) * 2)
    tex += [r"\bottomrule", r"\end{tabular}"]
    lines += ["", "## Paired CoSpRo differences", "", "| Control | Seeds 1, 2, 3, 4: ΔWGA | Mean ΔWGA |", "|---|---|---:|"]
    for arm, rr in report["paired_cospro_minus_control"].items():
        lines.append(f"| {arm} | {', '.join(format(r['wga'], '+.2f') for r in rr)} | {statistics.mean(r['wga'] for r in rr):+.2f} |")
    lines += ["", "## Individual results", "", "| Method | Seed | Avg. | WGA | Groups 00, 01, 10, 11 |", "|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['avg']:.2f} | {r['wga']:.2f} | {', '.join(format(x, '.2f') for x in r['groups'])} |")
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--artifact-root", type=Path, default=resolve_output_root())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-folder", default=os.environ.get("DATA_FOLDER", "/home/xar68reb/Datasets"))
    parser.add_argument("--checkpoints-json", type=Path, help='Optional mapping {"semantic_splice:2": "/path/last.pth", ...}')
    parser.add_argument("--seed", type=int, choices=range(1, 5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="prepare only: show paths without reading checkpoints or data")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.artifact_root / "reports/submission_checkpoint_test"
    if args.dry_run and args.action != "prepare":
        parser.error("--dry-run applies only to prepare")
    {"prepare": prepare, "run": execute, "summarize": collect}[args.action](args)


if __name__ == "__main__":
    main()
