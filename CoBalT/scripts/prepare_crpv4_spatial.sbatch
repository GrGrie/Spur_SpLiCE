#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=CRPv4_spatial
#SBATCH --array=0-3%2
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/slurm-%x_%A_%a.out
#SBATCH --error=./output/slurm-%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/scripts/prepare_crpv4_spatial.conf}"
source "${PROJECT_DIR}/scripts/load_config.sh"
load_splice_config "${CONFIG_FILE}"

record="${CRPV4_VARIANTS[${SLURM_ARRAY_TASK_ID}]:-}"
[[ -n "${record}" ]] || { echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
IFS='|' read -r VARIANT FEATURE_SOURCE USE_SLOTS <<< "${record}"
[[ -f "${PROJECT_DIR}/${CRP_CACHE_PATH}" ]] || {
    echo "CRP cache not found: ${PROJECT_DIR}/${CRP_CACHE_PATH}" >&2
    exit 2
}

OUTPUT_DIR="output/CRPv4/${DATASET}/seed${SEED}/${VARIANT}"
CHECKPOINT="${OUTPUT_DIR}/slots.pt"
BALANCE="${OUTPUT_DIR}/balance.pt"
mkdir -p "${PROJECT_DIR}/output/CRPv4" "${PROJECT_DIR}/${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ "${USE_SLOTS}" == "true" ]]; then
    TRAIN_ARGS=(
        -u -m CoBalT.train_spatial
        --dataset "${DATASET}"
        --data-root "${DATA_FOLDER}"
        --cache "${CRP_CACHE_PATH}"
        --output "${CHECKPOINT}"
        --seed "${SEED}"
        --epochs "${CRPV4_EPOCHS}"
        --batch-size "${CRPV4_BATCH_SIZE}"
        --workers "${NUM_WORKERS}"
        --image-size "${CRPV4_IMAGE_SIZE}"
        --crop-min "${CRPV4_CROP_MIN}"
        --num-slots "${CRPV4_NUM_SLOTS}"
        --feature-source "${FEATURE_SOURCE}"
        --learning-rate "${CRPV4_LEARNING_RATE}"
        --weight-decay "${CRPV4_WEIGHT_DECAY}"
        --student-temperature "${CRPV4_STUDENT_TEMPERATURE}"
        --teacher-temperature "${CRPV4_TEACHER_TEMPERATURE}"
        --teacher-momentum "${CRPV4_TEACHER_MOMENTUM}"
        --center-momentum "${CRPV4_CENTER_MOMENTUM}"
        --semantic-weight "${CRPV4_SEMANTIC_WEIGHT}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-entity "${WANDB_ENTITY}"
        --wandb-run-name "CRPv4_${DATASET}_${VARIANT}_s${SEED}"
        --wandb-group "CRPv4_${DATASET}_spatial_s${SEED}"
        --wandb-tags "crpv4,spatial,${VARIANT},label_free,dataset_${DATASET},seed_${SEED}"
    )
    if [[ "${CRPV4_AMP}" == "true" ]]; then TRAIN_ARGS+=(--amp); else TRAIN_ARGS+=(--no-amp); fi
    if [[ -n "${CRPV4_MAX_STEPS_PER_EPOCH}" ]]; then
        TRAIN_ARGS+=(--max-steps-per-epoch "${CRPV4_MAX_STEPS_PER_EPOCH}")
    fi
    python "${TRAIN_ARGS[@]}"
fi

EXTRACT_ARGS=(
    -u -m CoBalT.extract_spatial_balance
    --dataset "${DATASET}"
    --data-root "${DATA_FOLDER}"
    --cache "${CRP_CACHE_PATH}"
    --output "${BALANCE}"
    --variant "${VARIANT}"
    --batch-size "${CRPV4_BATCH_SIZE}"
    --workers "${NUM_WORKERS}"
    --image-size "${CRPV4_IMAGE_SIZE}"
    --concepts-per-region "${CRPV4_CONCEPTS_PER_REGION}"
)
if [[ "${USE_SLOTS}" == "true" ]]; then EXTRACT_ARGS+=(--checkpoint "${CHECKPOINT}"); fi
python "${EXTRACT_ARGS[@]}"

echo "[INFO] CRPv4 spatial evidence ready: ${PROJECT_DIR}/${BALANCE}"
conda deactivate
