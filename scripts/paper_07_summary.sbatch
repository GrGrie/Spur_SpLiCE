#!/bin/bash
#SBATCH --job-name=paper_07_summary
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/paper-07_summary-%A_%a.out
#SBATCH --error=logs/paper-07_summary-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
cd "${PROJECT_DIR}"
mkdir -p logs outputs/paper_completion_2026-09-08
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
export WANDB_MODE="${WANDB_MODE:-online}"
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.paper_completion summary

