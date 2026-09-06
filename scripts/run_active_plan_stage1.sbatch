#!/bin/bash
#
# One-command entrypoint for docs/ACTIVE_PLAN.md, stage 1.
# It validates the frozen artifacts, locks both experiment manifests, then
# submits prepare -> GPU arrays -> summaries -> final read-only report using
# Slurm dependencies.  The launcher itself does not allocate a GPU.
#
# Usage:
#   sbatch scripts/run_active_plan_stage1.sbatch
#
# Optional environment variables are intentionally limited to cluster paths:
#   PROJECT_DIR=/home/... CONDA_ENV=... sbatch scripts/run_active_plan_stage1.sbatch
#
#SBATCH --time=00:30:00
#SBATCH --job-name=splice_active_plan_s1
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
export PROJECT_DIR CONDA_ENV
cd "${PROJECT_DIR}"
mkdir -p output outputs/crp_replication_stage1

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

SOURCE_OUTPUT="${PROJECT_DIR}/outputs/crp_controls_logistic_v1_cluster_seeds12"
EXPECTED_CACHE="46b1c1fc552769dfbb9c6543347e1a2b"
EXPECTED_CRP="c50667264bc17a5ef0152a350f0f990c"
EXPECTED_RAW="c81c93e6f316f69249176929c9d3c4a4"

echo "Validating frozen cache and graphs before submitting training jobs."
python - "${SOURCE_OUTPUT}" "${EXPECTED_CACHE}" "${EXPECTED_CRP}" "${EXPECTED_RAW}" <<'PY'
import json
import sys
from pathlib import Path

from splice.graph_io import graph_fingerprint

source = Path(sys.argv[1])
expected = {"cache": sys.argv[2], "crp": sys.argv[3], "raw_clip": sys.argv[4]}
identity_paths = {
    "cache": source / "cache_identity.json",
    "graphs": source / "graph_identity.json",
}
for path in identity_paths.values():
    if not path.is_file():
        raise SystemExit(f"missing frozen identity file: {path}")
cache_identity = json.loads(identity_paths["cache"].read_text())
graph_identity = json.loads(identity_paths["graphs"].read_text())
if cache_identity.get("content_id") != expected["cache"]:
    raise SystemExit("frozen cache identity does not match ACTIVE_PLAN.md")
for name in ("crp", "raw_clip"):
    if graph_identity.get(name) != expected[name]:
        raise SystemExit(f"frozen {name} graph identity does not match ACTIVE_PLAN.md")
cache_path = Path(cache_identity["path"])
graph_paths = {
    "crp": source / "graphs" / "crp_graph.json",
    "raw_clip": source / "graphs" / "raw_clip_graph.json",
}
if not cache_path.is_file():
    raise SystemExit(f"frozen cache is missing: {cache_path}")
if graph_fingerprint(cache_path) != expected["cache"]:
    raise SystemExit("frozen cache content fingerprint changed")
for name, path in graph_paths.items():
    if not path.is_file():
        raise SystemExit(f"frozen graph is missing: {path}")
    if graph_fingerprint(path) != expected[name]:
        raise SystemExit(f"frozen {name} graph content fingerprint changed")
print("Frozen artifacts match the locked fingerprints.")
PY

lock_one() {
    local template="$1"
    local output="$2"
    mkdir -p "$(dirname "${output}")"
    python -u -m scripts.tools.lock_crp_followup \
        --template "${PROJECT_DIR}/scripts/${template}" \
        --source-output "${SOURCE_OUTPUT}" \
        --output "${output}"
    python -u -m scripts.tools.run_crp_controls \
        --config "${output}" \
        --validate-only
}

LOCKED_L05="${PROJECT_DIR}/outputs/crp_lambda05_replication_s34/locked_config.json"
LOCKED_RAW02="${PROJECT_DIR}/outputs/raw_lambda02_replication_s34/locked_config.json"
lock_one "crp_lambda05_replication_s34.conf" "${LOCKED_L05}"
lock_one "raw_lambda02_replication_s34.conf" "${LOCKED_RAW02}"

submit() {
    local job_id
    job_id="$(sbatch --parsable --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONDA_ENV="${CONDA_ENV}" "$@")"
    job_id="${job_id%%;*}"
    echo "submitted ${1}: ${job_id}" >&2
    printf '%s\n' "${job_id}"
}

PREP_L05="$(submit "${PROJECT_DIR}/scripts/prepare_crp_lambda05_replication_s34.sbatch")"
PREP_RAW02="$(submit "${PROJECT_DIR}/scripts/prepare_raw_lambda02_replication_s34.sbatch")"
ARRAY_L05="$(submit --dependency="afterok:${PREP_L05}" "${PROJECT_DIR}/scripts/crp_lambda05_replication_s34_array.sbatch")"
ARRAY_RAW02="$(submit --dependency="afterok:${PREP_RAW02}" "${PROJECT_DIR}/scripts/raw_lambda02_replication_s34_array.sbatch")"
SUMMARY_L05="$(submit --dependency="afterany:${ARRAY_L05}" "${PROJECT_DIR}/scripts/summarize_crp_lambda05_replication_s34.sbatch")"
SUMMARY_RAW02="$(submit --dependency="afterany:${ARRAY_RAW02}" "${PROJECT_DIR}/scripts/summarize_raw_lambda02_replication_s34.sbatch")"
FINAL_SUMMARY="$(submit --dependency="afterany:${SUMMARY_L05}:${SUMMARY_RAW02}" "${PROJECT_DIR}/scripts/summarize_crp_replication_stage1.sbatch")"

MANIFEST="${PROJECT_DIR}/outputs/crp_replication_stage1/slurm_jobs_${SLURM_JOB_ID}.json"
python - "${MANIFEST}" "${PREP_L05}" "${ARRAY_L05}" "${SUMMARY_L05}" "${PREP_RAW02}" "${ARRAY_RAW02}" "${SUMMARY_RAW02}" "${FINAL_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
names = [
    "prepare_lambda05", "array_lambda05", "summary_lambda05",
    "prepare_raw_lambda02", "array_raw_lambda02", "summary_raw_lambda02",
    "final_summary",
]
payload = {name: job_id for name, job_id in zip(names, sys.argv[2:])}
payload["status"] = "submitted_with_dependencies"
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "All stage-1 jobs have been submitted. The final report is produced by job ${FINAL_SUMMARY}."
