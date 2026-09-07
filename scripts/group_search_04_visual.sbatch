#!/bin/bash
#SBATCH --job-name=groups_04_visual
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/groups-04_visual-%A_%a.out
#SBATCH --error=logs/groups-04_visual-%A_%a.err
#SBATCH --array=0-2%3

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
cd "${PROJECT_DIR}"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
export WANDB_MODE="${WANDB_MODE:-online}"
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.visualize_concept_group_search --task "${SLURM_ARRAY_TASK_ID}" --data-folder "${DATA_FOLDER:-/home/xar68reb/Datasets}"
