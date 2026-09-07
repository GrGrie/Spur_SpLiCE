#!/bin/bash
# One command for NEXT_TESTS_2026-09-07.md.
# Integrity failure prevents all new SSL training through an afterok dependency.
#SBATCH --time=00:30:00
#SBATCH --job-name=splice_next_tests
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=outputs/next_tests_2026-09-07/logs/orchestrator-%j.out
#SBATCH --error=outputs/next_tests_2026-09-07/logs/orchestrator-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
SPLICE_CONDA_ENV="${SPLICE_CONDA_ENV:-grgrie-train}"
export PROJECT_DIR SPLICE_CONDA_ENV
cd "${PROJECT_DIR}"
mkdir -p outputs/next_tests_2026-09-07/logs outputs/next_tests_2026-09-07/integrity outputs/next_tests_2026-09-07/gradient outputs/next_tests_2026-09-07/concept_transfer
source scripts/load_splice_cluster_env.sh

submit() {
    local job_id
    job_id="$(sbatch --parsable --export=ALL,PROJECT_DIR="${PROJECT_DIR}",SPLICE_CONDA_ENV="${SPLICE_CONDA_ENV}" "$@")"
    job_id="${job_id%%;*}"
    echo "submitted $(basename "$1"): ${job_id}" >&2
    printf '%s\n' "${job_id}"
}

INTEGRITY="$(submit scripts/check_saved_probes.sbatch)"
GRAD_PREP="$(submit scripts/prepare_crp_transfer_gradient_diagnostic.sbatch)"
GRAD_ARRAY="$(submit --dependency="afterok:${GRAD_PREP}" scripts/run_crp_transfer_gradient_diagnostic_array.sbatch)"
GRAD_SUMMARY="$(submit --dependency="afterany:${GRAD_ARRAY}" scripts/summarize_crp_transfer_gradient_diagnostic.sbatch)"

TRANSFER_PREP="$(submit --dependency="afterok:${INTEGRITY}" scripts/prepare_concept_transfer_v1.sbatch)"
SMOKE="$(submit --dependency="afterok:${TRANSFER_PREP}" scripts/run_concept_transfer_v1_smoke.sbatch)"
TRANSFER_ARRAY="$(submit --dependency="afterok:${SMOKE}:${GRAD_SUMMARY}" scripts/run_concept_transfer_v1_array.sbatch)"
TRANSFER_SUMMARY="$(submit --dependency="afterany:${TRANSFER_ARRAY}" scripts/summarize_concept_transfer_v1.sbatch)"

MANIFEST="outputs/next_tests_2026-09-07/submitted_${SLURM_JOB_ID}.json"
"${PYTHON_BIN}" - "${MANIFEST}" "${INTEGRITY}" "${GRAD_PREP}" "${GRAD_ARRAY}" "${GRAD_SUMMARY}" "${TRANSFER_PREP}" "${SMOKE}" "${TRANSFER_ARRAY}" "${TRANSFER_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

names = ["integrity", "gradient_prepare", "gradient_array", "gradient_summary", "transfer_prepare", "transfer_smoke", "transfer_array", "transfer_summary"]
payload = {name: job_id for name, job_id in zip(names, sys.argv[2:])}
payload.update({"status": "submitted", "integrity_gate": "afterok", "training_resources": {"mem": "40G", "cpus_per_task": 5}})
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Submitted integrity, gradient diagnostics, and conditional 8-run concept transfer."
echo "Read outputs/next_tests_2026-09-07/submitted_${SLURM_JOB_ID}.json for job IDs."
