#!/bin/bash
#SBATCH --job-name=groups_01_audit
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/groups-01_audit-%A_%a.out
#SBATCH --error=logs/groups-01_audit-%A_%a.err
#SBATCH --array=0-1%2

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
cd "${PROJECT_DIR}"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
export WANDB_MODE="${WANDB_MODE:-online}"
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.run_concept_group_search audit --task "${SLURM_ARRAY_TASK_ID}"
