#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=splice_transfer
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --array=0-7%8
#SBATCH --output=outputs/next_tests_2026-09-07/logs/transfer-%A_%a.out
#SBATCH --error=outputs/next_tests_2026-09-07/logs/transfer-%A_%a.err
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"; SPLICE_CONDA_ENV="${SPLICE_CONDA_ENV:-grgrie-train}"; export PROJECT_DIR SPLICE_CONDA_ENV
cd "${PROJECT_DIR}"; mkdir -p outputs/next_tests_2026-09-07/logs
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.run_concept_transfer --config scripts/concept_transfer_v1.conf --stage run --array-task-id "${SLURM_ARRAY_TASK_ID}"
