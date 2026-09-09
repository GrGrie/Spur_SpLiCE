#!/usr/bin/env bash
#SBATCH --job-name=spur-splice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/experiment-%A_%a.out
#SBATCH --error=logs/experiment-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
MANIFEST="${MANIFEST:-experiments/manifests/waterbirds_crp.json}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
EXISTING_POLICY="${EXISTING_POLICY:-error}"
export DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
cd "${PROJECT_DIR}"
mkdir -p logs
source scripts/load_splice_cluster_env.sh
runner_args=("${MANIFEST}" --task "${TASK_ID}" --existing "${EXISTING_POLICY}")
if [[ -n "${SPUR_SPLICE_ARTIFACT_ROOT:-}" ]]; then
  runner_args+=(--artifact-root "${SPUR_SPLICE_ARTIFACT_ROOT}")
fi
"${PYTHON_BIN}" -u -m experiments.runner "${runner_args[@]}"
