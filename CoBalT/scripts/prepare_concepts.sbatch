#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=CoBalT_concepts
#SBATCH --array=0-0
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

# Label-free CoBalT Stage 1 only: discovery followed by fixed concept inference.
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/scripts/prepare_concepts.conf}"
source "${PROJECT_DIR}/scripts/load_config.sh"
load_splice_config "${CONFIG_FILE}"

CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
DATASET="${DATASET:-waterbirds}" # waterbirds | celeba
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
NUM_WORKERS="${NUM_WORKERS:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-Spur_SpLiCE}"
WANDB_ENTITY="${WANDB_ENTITY:-gsgrechkin-rptu}"
WANDB_GROUP="CoBalT_${DATASET}_concept_check"
WANDB_TAGS="cobalt_concept_check,label_free_discovery,machine_cluster,dataset_${DATASET},seed_${SEED}"
OUTPUT_DIR="output/CoBalT/${DATASET}/seed${SEED}"
DISCOVERY_CHECKPOINT="${OUTPUT_DIR}/discovery.pt"
CONCEPT_ARTIFACT="${OUTPUT_DIR}/concepts.pt"

[[ -d "${PROJECT_DIR}" ]] || { echo "Missing project: ${PROJECT_DIR}" >&2; exit 2; }
mkdir -p "${PROJECT_DIR}/output/CoBalT" "${PROJECT_DIR}/${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "=== CoBalT label-free concept discovery ==="
case "${COBALT_MODEL}" in
    resnet18_large)
        COBALT_BACKBONE="resnet18"
        COBALT_PRETRAINED="false"
        COBALT_ALLOW_NONPAPER_BACKBONE="true"
        ;;
    resnet50_pretrained)
        COBALT_BACKBONE="resnet50"
        COBALT_PRETRAINED="true"
        COBALT_ALLOW_NONPAPER_BACKBONE="false"
        ;;
    *)
        echo "ERROR: COBALT_MODEL must be resnet18_large or resnet50_pretrained; got ${COBALT_MODEL}" >&2
        exit 2
        ;;
esac

DISCOVERY_ARGS=(
    -u -m CoBalT.train_discovery
    --dataset "${DATASET}"
    --data-root "${DATA_FOLDER}"
    --output "${DISCOVERY_CHECKPOINT}"
    --seed "${SEED}"
    --epochs "${COBALT_EPOCHS}"
    --batch-size "${COBALT_BATCH_SIZE}"
    --workers "${NUM_WORKERS}"
    --image-size "${COBALT_IMAGE_SIZE}"
    --crop-min "${COBALT_CROP_MIN}"
    --num-slots "${COBALT_NUM_SLOTS}"
    --codebook-size "${COBALT_CODEBOOK_SIZE}"
    --slot-dim "${COBALT_SLOT_DIM}"
    --hidden-dim "${COBALT_HIDDEN_DIM}"
    --learning-rate "${COBALT_LEARNING_RATE}"
    --weight-decay "${COBALT_WEIGHT_DECAY}"
    --student-temperature "${COBALT_STUDENT_TEMPERATURE}"
    --teacher-temperature "${COBALT_TEACHER_TEMPERATURE}"
    --contrastive-temperature "${COBALT_CONTRASTIVE_TEMPERATURE}"
    --teacher-momentum "${COBALT_TEACHER_MOMENTUM}"
    --codebook-momentum "${COBALT_CODEBOOK_MOMENTUM}"
    --center-momentum "${COBALT_CENTER_MOMENTUM}"
    --backbone "${COBALT_BACKBONE}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-run-name "CoBalT_${DATASET}_s${SEED}_concept_check"
    --wandb-group "${WANDB_GROUP}"
    --wandb-tags "${WANDB_TAGS}"
)
if [[ "${COBALT_PRETRAINED}" == "true" ]]; then
    DISCOVERY_ARGS+=(--pretrained)
else
    DISCOVERY_ARGS+=(--no-pretrained)
fi
if [[ "${COBALT_ALLOW_NONPAPER_BACKBONE}" == "true" ]]; then
    DISCOVERY_ARGS+=(--allow-nonpaper-backbone)
else
    DISCOVERY_ARGS+=(--no-allow-nonpaper-backbone)
fi
if [[ "${COBALT_AMP}" == "true" ]]; then
    DISCOVERY_ARGS+=(--amp)
else
    DISCOVERY_ARGS+=(--no-amp)
fi
if [[ -n "${COBALT_MAX_STEPS_PER_EPOCH}" ]]; then
    DISCOVERY_ARGS+=(--max-steps-per-epoch "${COBALT_MAX_STEPS_PER_EPOCH}")
fi

python "${DISCOVERY_ARGS[@]}"

echo "=== Extracting fixed CoBalT concept memberships ==="
python -u -m CoBalT.extract_concepts \
    --checkpoint "${DISCOVERY_CHECKPOINT}" \
    --data-root "${DATA_FOLDER}" \
    --output "${CONCEPT_ARTIFACT}" \
    --workers "${NUM_WORKERS}"

echo "[INFO] CoBalT concepts ready: ${PROJECT_DIR}/${CONCEPT_ARTIFACT}"
conda deactivate
