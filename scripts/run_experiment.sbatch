#!/usr/bin/env bash
#SBATCH --job-name=spur-splice
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/SLURM/experiment-%A_%a.out
#SBATCH --error=outputs/SLURM/experiment-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
MANIFEST="${MANIFEST:-experiments/manifests/waterbirds_cospro.json}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
EXISTING_POLICY="${EXISTING_POLICY:-error}"
export DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source scripts/load_splice_cluster_env.sh
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  export SPUR_SPLICE_SCRATCH_ROOT="${SPUR_SPLICE_SCRATCH_ROOT:-${PROJECT_DIR}/tmp}"
fi
runner_args=("${MANIFEST}" --task "${TASK_ID}" --existing "${EXISTING_POLICY}")
if [[ $# -gt 0 && "$1" != --* ]]; then
  runner_args[0]="$1"
  shift
fi
for arg in "$@"; do
  if [[ "$arg" == "--task" || "$arg" == --task=* || "$arg" == "--seed" || "$arg" == --seed=* || "$arg" == "--list" ]]; then
    runner_args=("${runner_args[0]}" --existing "${EXISTING_POLICY}")
    break
  fi
done
if [[ "${LOCKED_TEST:-0}" == "1" ]]; then
  runner_args+=(--locked-test)
fi
if [[ -n "${SPUR_SPLICE_OUTPUT_ROOT:-}" ]]; then
  runner_args+=(--output-root "${SPUR_SPLICE_OUTPUT_ROOT}")
fi
"${PYTHON_BIN}" -u -m experiments.runner "${runner_args[@]}" "$@"
