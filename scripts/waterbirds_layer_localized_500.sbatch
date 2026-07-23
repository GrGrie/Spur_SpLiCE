#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=Waterbirds_Localized500
#SBATCH --array=0-1
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/%x_%A_%a.out
#SBATCH --error=./output/%x_%A_%a.err

set -euo pipefail

# task 0: matched SimCLR baseline; task 1: automatic layer-localized SpLiCE.
# Override DATASET, DATA_FOLDER, SEED, PROJECT_DIR, or CONDA_ENV through --export.
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"
DATASET="${DATASET:-waterbirds}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-5}"
WANDB_NAME="${WANDB_NAME:-Spur_SpLiCE}"
WANDB_ENTITY="${WANDB_ENTITY:-gsgrechkin-rptu}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) VARIANT="baseline" ;;
    1) VARIANT="localized" ;;
    *)
        echo "Expected SLURM_ARRAY_TASK_ID 0 or 1, got ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

mkdir -p "${PROJECT_DIR}/output"
cd "${PROJECT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python scripts/run_layer_localized_500.py \
    --variant "${VARIANT}" \
    --dataset "${DATASET}" \
    --data_folder "${DATA_FOLDER}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --use_wandb \
    --wandb_name "${WANDB_NAME}" \
    --entity "${WANDB_ENTITY}"
