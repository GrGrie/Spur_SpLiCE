#!/usr/bin/env bash
# Run the CPU golden training test with the code of several commits on one node and compare them.
#
# A golden snapshot taken on one node can fail on another with unchanged code, because the probe
# accuracies of the 16-image fixture flip with the CPU's floating-point kernels. Commits that agree
# bit for bit on the same node prove the code did not change the numbers; the first commit that
# disagrees names the change that did. Every run uses 2 threads, as the golden smoke job does.
#
# When every commit agrees, the job also rewrites tests/golden/training_runs.linux.json on this node,
# recording its CPU model; commit that file. The golden test compares only on nodes with that CPU.
#
#   sbatch --nodelist=dgx002 scripts/bisect_golden_training.sbatch
#   COMMITS="37d6096 HEAD" sbatch --nodelist=dgx001 scripts/bisect_golden_training.sbatch
#
# Read the verdict at the end of the .out file: "SAME NUMBERS" or "CODE CHANGED NUMBERS".

#SBATCH --job-name=cospro-golden-bisect
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/SLURM/golden-bisect-%j.out
#SBATCH --error=outputs/SLURM/golden-bisect-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

RUN_ID="${SLURM_JOB_ID:-local-$$}"
# Phase 0 (the snapshot's own commit), end of phase 4, end of phase 8, this checkout.
COMMITS="${COMMITS:-37d6096 34c75a0 8398090 HEAD}"
REPORT_DIR="${PROJECT_DIR}/outputs/reports/golden-bisect/${RUN_ID}"
WORK_DIR="${TMPDIR:-/tmp}/golden-bisect-${RUN_ID}"
source scripts/announce_results.sh
announce_results "${REPORT_DIR}/" "verdict: the last line of this .out file"

export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
mkdir -p "${REPORT_DIR}" "${WORK_DIR}"
worktrees=()
cleanup() {
  for tree in "${worktrees[@]}"; do git -C "${PROJECT_DIR}" worktree remove --force "${tree}" 2>/dev/null || true; done
  git -C "${PROJECT_DIR}" worktree prune || true
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT
echo "[golden-bisect] node $(hostname), cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"

reference=""
for commit in ${COMMITS}; do
  sha="$(git rev-parse --short "${commit}")"
  tree="${WORK_DIR}/code-${sha}"
  git worktree add --detach "${tree}" "${sha}" >/dev/null
  worktrees+=("${tree}")
  echo "[golden-bisect] ${commit} (${sha})"
  (cd "${tree}" && PYTHONPATH="${tree}" SPUR_SPLICE_GOLDEN_REPORT="${REPORT_DIR}/${sha}.json" \
    "${PYTHON_BIN}" -m pytest -q tests/test_golden_training.py --basetemp="${WORK_DIR}/pytest-${sha}" -p no:cacheprovider) \
    | tail -n 2 || true
  reference="${REPORT_DIR}/${sha}.json"
done

set +e
"${PYTHON_BIN}" -u -m tools.maintenance.compare_golden_reports "${REPORT_DIR}" --reference "${reference}" \
  --snapshot "${PROJECT_DIR}/tests/golden/training_runs.linux.json"
verdict=$?
set -e
if [[ "${verdict}" == "0" ]]; then
  # The phase 0 code gives these numbers on this node, so they are the node's snapshot.
  SPUR_SPLICE_UPDATE_GOLDEN=1 "${PYTHON_BIN}" -m pytest -q tests/test_golden_training.py \
    --basetemp="${WORK_DIR}/pytest-snapshot" -p no:cacheprovider | tail -n 1
  echo "[golden-bisect] rewrote tests/golden/training_runs.linux.json for this CPU; review it with git diff and commit it"
fi
exit "${verdict}"
