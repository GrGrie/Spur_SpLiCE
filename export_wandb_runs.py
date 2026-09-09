import json
from pathlib import Path

from splice.artifacts import reference
from datetime import datetime, timezone

import wandb


ENTITY = "gsgrechkin-rptu"
PROJECT = "Spur_SpLiCE"
OUTPUT_PATH = reference("wandb_exports", "spur_splice_wandb_runs_current.json")

# Current research generation starts here.
MIN_CREATED_AT = datetime(2026, 9, 5, tzinfo=timezone.utc)

# Old method families that should not be exported anymore.
LEGACY_SPLICE_MODES = {
    "augment",
    "augment_corr_reg",
    "corr_reg",
    "synthesis",
    "synthesis_distill",
}

# Extra name/group markers used only to reject obviously old experiment branches.
LEGACY_NAME_MARKERS = (
    "legacy",
    "deprecated",
    "augment",
    "synthesis",
    "corr_reg",
)


def parse_wandb_datetime(value):
    if not value:
        return None

    text = str(value)

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def make_json_safe(value):
    """Recursively convert W&B values into JSON-safe Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    if isinstance(value, datetime):
        return value.isoformat()

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def safe_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_current_research_run(run):
    """
    Keep runs that belong to the current research generation.

    The decision is based mainly on protocol/config fields.
    Run names/groups are used only to reject obvious legacy branches.
    """
    config = dict(run.config or {})

    # ------------------------------------------------------------------
    # 1. Time gate
    # ------------------------------------------------------------------
    created_at = parse_wandb_datetime(run.created_at)
    if created_at is None or created_at < MIN_CREATED_AT:
        return False

    # ------------------------------------------------------------------
    # 2. Core protocol identity
    # ------------------------------------------------------------------
    if config.get("dataset") != "waterbirds":
        return False

    if config.get("model") != "resnet18_large":
        return False

    if safe_int(config.get("epochs")) != 500:
        return False

    if safe_int(config.get("batch_size")) != 128:
        return False

    if config.get("train_set_linear_layer") != "ds_train":
        return False

    if config.get("linear_eval_split") != "val":
        return False

    if config.get("linear_probe_mode") != "periodic":
        return False

    if safe_int(config.get("linear_probe_freq")) != 25:
        return False

    # New protocol uses the logistic probe. A missing key is rejected
    # because older runs often did not record this setting explicitly.
    if config.get("linear_probe_solver") != "logistic":
        return False

    # ------------------------------------------------------------------
    # 3. Exclude old method families
    # ------------------------------------------------------------------
    splice_mode = str(config.get("splice_mode", "")).strip().lower()

    if splice_mode in LEGACY_SPLICE_MODES:
        return False

    # ------------------------------------------------------------------
    # 4. Modern CLIP / SpLiCE stack
    # ------------------------------------------------------------------
    splice_model = str(config.get("splice_model", "")).strip()

    # Pure SimCLR baselines may still carry this config field, but when it
    # is present it should match the current teacher backbone.
    if splice_model and splice_model != "open_clip:ViT-B-32":
        return False

    splice_vocab = str(config.get("splice_vocab", "")).strip()

    # Current SpLiCE-based runs use OpenImages V7.
    # Pure SimCLR baselines are allowed even if no vocabulary is relevant.
    if splice_mode not in {"", "none"}:
        if splice_vocab != "openimages_v7":
            return False

    # ------------------------------------------------------------------
    # 5. Soft rejection by name/group
    # ------------------------------------------------------------------
    name = str(run.name or "").lower()
    group = str(run.group or "").lower()

    if any(marker in name or marker in group for marker in LEGACY_NAME_MARKERS):
        return False

    return True


def export_runs():
    api = wandb.Api()

    print(f"Loading runs from {ENTITY}/{PROJECT}...")
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    print(f"Found {len(all_runs)} total runs.")

    relevant_runs = [
        run for run in all_runs
        if is_current_research_run(run)
    ]

    relevant_runs.sort(
        key=lambda run: (
            parse_wandb_datetime(run.created_at)
            or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )

    print(f"Keeping {len(relevant_runs)} current/relevant runs.")
    print(f"Filtered out {len(all_runs) - len(relevant_runs)} runs.\n")

    exported_runs = []

    for index, run in enumerate(relevant_runs, start=1):
        print(
            f"[{index}/{len(relevant_runs)}] "
            f"{run.created_at} | {run.group or '-'} | {run.name}"
        )

        try:
            config = make_json_safe(dict(run.config))
        except Exception as exc:
            print(f"  WARNING: config failed: {exc}")
            config = {}

        try:
            summary = make_json_safe(dict(run.summary))
        except Exception as exc:
            print(f"  WARNING: summary failed: {exc}")
            summary = {}

        run_data = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "entity": run.entity,
            "project": run.project,
            "url": run.url,
            "created_at": str(run.created_at),
            "group": run.group,
            "job_type": run.job_type,
            "tags": list(run.tags or []),
            "notes": run.notes,
            "config": config,
            "summary": summary,
        }

        try:
            run_data["metadata"] = make_json_safe(run.metadata)
        except Exception as exc:
            run_data["metadata"] = None
            run_data["metadata_error"] = str(exc)

        exported_runs.append(run_data)

    payload = {
        "entity": ENTITY,
        "project": PROJECT,
        "filter": {
            "min_created_at": MIN_CREATED_AT.isoformat(),
            "dataset": "waterbirds",
            "model": "resnet18_large",
            "epochs": 500,
            "batch_size": 128,
            "train_set_linear_layer": "ds_train",
            "linear_eval_split": "val",
            "linear_probe_mode": "periodic",
            "linear_probe_freq": 25,
            "linear_probe_solver": "logistic",
            "splice_model": "open_clip:ViT-B-32",
            "splice_vocab_for_splice_runs": "openimages_v7",
            "legacy_splice_modes": sorted(LEGACY_SPLICE_MODES),
            "legacy_name_markers": list(LEGACY_NAME_MARKERS),
        },
        "total_wandb_runs": len(all_runs),
        "run_count": len(exported_runs),
        "filtered_out_count": len(all_runs) - len(exported_runs),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "runs": exported_runs,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\n----------------------------------------")
    print("Export complete")
    print("----------------------------------------")
    print(f"Total W&B runs: {len(all_runs)}")
    print(f"Relevant runs:  {len(exported_runs)}")
    print(f"Filtered out:   {len(all_runs) - len(exported_runs)}")
    print(f"File: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    export_runs()
