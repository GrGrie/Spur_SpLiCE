#!/usr/bin/env bash
#SBATCH --job-name=la-ssl
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/SLURM/la-ssl-%j.out
#SBATCH --error=outputs/SLURM/la-ssl-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
cd "${PROJECT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source scripts/load_splice_cluster_env.sh
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  export SPUR_SPLICE_SCRATCH_ROOT="${SPUR_SPLICE_SCRATCH_ROOT:-${PROJECT_DIR}/tmp}"
fi
runner_args=(experiments/manifests/waterbirds_la_ssl.json --arm la_ssl)
selection_given=0
for arg in "$@"; do
  if [[ "$arg" == "--seed" || "$arg" == --seed=* || "$arg" == "--task" || "$arg" == --task=* ]]; then
    selection_given=1
    break
  fi
done
if [[ "${selection_given}" == "0" ]]; then
  runner_args+=(--seed "${SEED:-1}")
fi
"${PYTHON_BIN}" -u -m experiments.runner "${runner_args[@]}" "$@"
