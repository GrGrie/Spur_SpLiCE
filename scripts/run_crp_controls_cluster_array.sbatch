#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CRP_array
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --array=0-7%8
#SBATCH --output=./output/crp_controls_cluster/array-%A_%a.out
#SBATCH --error=./output/crp_controls_cluster/array-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${PROJECT_DIR}/scripts/run_crp_controls_cluster.conf"

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-grgrie-train}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u -m scripts.tools.run_crp_controls \
    --config "${CONFIG_FILE}" \
    --array-task-id "${SLURM_ARRAY_TASK_ID}"
