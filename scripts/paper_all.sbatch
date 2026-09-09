#!/bin/bash
# One-command CoSpRo paper pipeline. Child jobs use afterok dependencies, so a
# failed prerequisite prevents its dependent work from starting.
#
# Usage on the cluster from the repository root:
#   mkdir -p logs
#   sbatch scripts/paper_all.sbatch

#SBATCH --job-name=paper_all
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/paper-all-%j.out
#SBATCH --error=logs/paper-all-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
cd "${PROJECT_DIR}"
mkdir -p logs outputs/paper_completion_2026-09-08

submit() {
    local dependency="$1"
    local script="$2"
    local job
    if [[ -n "${dependency}" ]]; then
        job="$(sbatch --parsable --dependency="afterok:${dependency}" "${script}")"
    else
        job="$(sbatch --parsable "${script}")"
    fi
    printf '%s' "${job%%;*}"
}

inventory="$(submit "" scripts/paper_00_inventory.sbatch)"
prepare="$(submit "${inventory}" scripts/paper_01_prepare.sbatch)"
direct="$(submit "${prepare}" scripts/paper_02_direct.sbatch)"
graph="$(submit "${prepare}" scripts/paper_03_graph.sbatch)"
summary="$(submit "${direct}:${graph}" scripts/paper_07_summary.sbatch)"
dispatch="$(sbatch --parsable --dependency="afterok:${inventory}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},PAPER_SUMMARY_JOB=${summary}" \
    scripts/paper_04_dispatch_missing_core.sbatch)"
dispatch="${dispatch%%;*}"

manifest="outputs/paper_completion_2026-09-08/submitted_all_${SLURM_JOB_ID}.json"
printf '{\n  "core_test_dispatch": "%s",\n  "direct": "%s",\n  "graph": "%s",\n  "inventory": "%s",\n  "launcher": "paper_all.sbatch",\n  "meeting_summary": "%s",\n  "prepare": "%s",\n  "status": "submitted"\n}\n' \
    "${dispatch}" "${direct}" "${graph}" "${inventory}" "${summary}" "${prepare}" > "${manifest}"
cat "${manifest}"

echo "Submitted paper pipeline. Initial manifest: ${manifest}"
echo "The dispatcher (${dispatch}) will create the conditional core/test chain after inventory."
