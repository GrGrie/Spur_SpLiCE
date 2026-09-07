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
python - "${manifest}" "${PAPER_SUMMARY_JOB}" "${core_job}" "${lock_job}" "${test_job}" "${test_summary_job}" <<'PY'
import json
import sys
from pathlib import Path

keys = ["meeting_summary", "missing_core", "final_test_lock", "test_probes", "test_summary"]
payload = dict(zip(keys, sys.argv[2:]))
payload["status"] = "submitted"
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
