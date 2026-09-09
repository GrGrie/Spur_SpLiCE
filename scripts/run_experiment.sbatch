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
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR must be exported by submit_experiment.sh}"
MANIFEST="${MANIFEST:-experiments/manifests/waterbirds_crp.json}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m experiments.runner "${MANIFEST}" --task "${TASK_ID}"
