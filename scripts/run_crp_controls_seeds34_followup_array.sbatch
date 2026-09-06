#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_s34_controls
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --array=0-9%10
#SBATCH --output=./output/crp_followup/%x-%A_%a.out
#SBATCH --error=./output/crp_followup/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="/home/xar68reb/Spur_SpLiCE/outputs/crp_followup_controls_v1_seeds34/locked_config.json"

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-grgrie-train}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u -m scripts.tools.run_crp_controls \
    --config "${CONFIG_FILE}" \
    --array-task-id "${SLURM_ARRAY_TASK_ID}"
