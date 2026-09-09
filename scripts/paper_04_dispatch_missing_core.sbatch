#!/bin/bash
# Submit the conditional core-reproduction array after inventory is available.
# This is intentionally a dispatcher: only it knows which array IDs are missing.
# All child Slurm logs still go to the root logs/ directory.
#
# Required environment: PAPER_SUMMARY_JOB, passed by paper_all.sbatch.

#SBATCH --job-name=paper_04_dispatch
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/paper-04-dispatch-%j.out
#SBATCH --error=logs/paper-04-dispatch-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
PAPER_SUMMARY_JOB="${PAPER_SUMMARY_JOB:?PAPER_SUMMARY_JOB must name paper_07_summary}"
cd "${PROJECT_DIR}"
mkdir -p logs outputs/paper_completion_2026-09-08

missing_file="outputs/paper_completion_2026-09-08/missing_core_tasks.txt"
if [[ ! -f "${missing_file}" ]]; then
    echo "ERROR: inventory did not produce ${missing_file}" >&2
    exit 2
fi

missing="$(tr -d '\r\n' < "${missing_file}")"
dependency="afterok:${PAPER_SUMMARY_JOB}"
core_job=""
if [[ -n "${missing}" ]]; then
    core_job="$(sbatch --parsable --array="${missing}%4" scripts/paper_04_missing_core.sbatch)"
    core_job="${core_job%%;*}"
    dependency="${dependency}:${core_job}"
    echo "Submitted missing core reproductions: ${core_job} (tasks ${missing})"
else
    echo "All 20 core models are available; no core reproduction array submitted."
fi

lock_job="$(sbatch --parsable --dependency="${dependency}" scripts/paper_05_lock_test.sbatch)"
lock_job="${lock_job%%;*}"
test_job="$(sbatch --parsable --dependency="afterok:${lock_job}" scripts/paper_06_test.sbatch)"
test_job="${test_job%%;*}"
test_summary_job="$(sbatch --parsable --dependency="afterok:${test_job}" scripts/paper_08_test_summary.sbatch)"
test_summary_job="${test_summary_job%%;*}"

manifest="outputs/paper_completion_2026-09-08/submitted_test_chain_${SLURM_JOB_ID}.json"
printf '{\n  "final_test_lock": "%s",\n  "meeting_summary": "%s",\n  "missing_core": "%s",\n  "status": "submitted",\n  "test_probes": "%s",\n  "test_summary": "%s"\n}\n' \
    "${lock_job}" "${PAPER_SUMMARY_JOB}" "${core_job}" "${test_job}" "${test_summary_job}" > "${manifest}"
cat "${manifest}"
