#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=splice_raw02
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --array=0-1%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/load_splice_cluster_env.sh"
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5

"${PYTHON_BIN}" -u -m scripts.tools.run_crp_controls \
    --config "${PROJECT_DIR}/outputs/raw_lambda02_replication_s34/locked_config.json" \
    --array-task-id "${SLURM_ARRAY_TASK_ID}"
