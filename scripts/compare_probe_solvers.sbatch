#!/usr/bin/env bash
# Refit saved probe features with the old SGD probe and the current logistic probe, side by side.
# Needs no checkpoint and no retraining: it reads the probe_features_epoch_*.pt a run already wrote.
#
#   sbatch scripts/compare_probe_solvers.sbatch FEATURES.pt|DIRECTORY [...] [-- compare options]
#   sbatch scripts/compare_probe_solvers.sbatch --find PATH_SUBSTRING [-- compare options]
#
# --find searches the scratch root and outputs/ for probe_features_epoch_500_*.pt whose path contains the
# substring (for example the W&B run name or the Slurm job id of the training run). Without any
# argument the job lists every epoch-500 feature file it can find and exits, so the .out file shows
# what to pass. Options after "--" go to cospro.cli.compare_probe_solvers (default: --seed 1).

#SBATCH --job-name=cospro-probe-solvers
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/SLURM/probe-solvers-%j.out
#SBATCH --error=outputs/SLURM/probe-solvers-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

REPORT="outputs/reports/probe_solver_check/${SLURM_JOB_ID:-local}.json"
source scripts/announce_results.sh
announce_results "${REPORT}" "table: this .out file"

SEARCH_ROOTS=("${SPUR_SPLICE_SCRATCH_ROOT}" "${PROJECT_DIR}/outputs")
features=()
compare_args=()
find_pattern=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --find) find_pattern="$2"; shift 2 ;;
    --find=*) find_pattern="${1#--find=}"; shift ;;
    --) shift; compare_args=("$@"); break ;;
    *) features+=("$1"); shift ;;
  esac
done

if [[ -n "${find_pattern}" ]]; then
  mapfile -d '' found < <(find "${SEARCH_ROOTS[@]}" -type f -name 'probe_features_epoch_500_*.pt' \
    -path "*${find_pattern}*" -print0 2>/dev/null | sort -z)
  features+=("${found[@]}")
fi
if [[ "${#features[@]}" == "0" ]]; then
  echo "No feature files given. Epoch-500 feature files under ${SEARCH_ROOTS[*]}:" >&2
  find "${SEARCH_ROOTS[@]}" -type f -name 'probe_features_epoch_500_*.pt' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' \
    2>/dev/null | sort >&2 || true
  echo "Resubmit with one of these paths, or with --find SUBSTRING." >&2
  exit 2
fi
if [[ "${#compare_args[@]}" == "0" ]]; then
  compare_args=(--seed 1)
fi

"${PYTHON_BIN}" -u -m cospro.cli.compare_probe_solvers "${features[@]}" --output "${REPORT}" "${compare_args[@]}"
