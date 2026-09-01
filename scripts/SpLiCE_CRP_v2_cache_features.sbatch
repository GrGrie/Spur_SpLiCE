#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=SpLiCE_CRP_v2_cache
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/%x_%j.out
#SBATCH --error=./output/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"
DATASET="${DATASET:-waterbirds}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_DINO="${USE_DINO:-true}"

if [[ "${USE_DINO}" != "true" && "${USE_DINO}" != "false" ]]; then
    echo "USE_DINO must be true or false, got: ${USE_DINO}" >&2
    exit 2
fi
if [[ -z "${OUTPUT_PATH}" ]]; then
    if [[ "${USE_DINO}" == "true" ]]; then
        OUTPUT_PATH="outputs/crp/${DATASET}_train_features.pt"
    else
        OUTPUT_PATH="outputs/crp/${DATASET}_train_features_no_dino.pt"
    fi
fi

# Frozen feature settings. This job is a one-time preparation step and is not
# submitted by the graph/training sweep pipeline.
SPLICE_MODEL="open_clip:ViT-B-32"
SPLICE_PRETRAINED="laion2b_s34b_b79k"
SPLICE_VOCAB="laion"
SPLICE_VOCAB_SIZE="10000"
SPLICE_L1_PENALTY="0.25"
DINO_MODEL="vit_small_patch16_dinov3.lvd1689m"
FORCE="false"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${OUTPUT_PATH}")" ./output

if [[ -f "${OUTPUT_PATH}" && "${FORCE}" != "true" ]]; then
    echo "Frozen feature file already exists: ${OUTPUT_PATH}"
    exit 0
fi

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== SpLiCE-CRP v2 frozen feature cache ==="
echo "dataset=${DATASET} data=${DATA_FOLDER} output=${OUTPUT_PATH} use_dino=${USE_DINO} dino=${DINO_MODEL}"
echo "splice_model=${SPLICE_MODEL} pretrained=${SPLICE_PRETRAINED} vocab=${SPLICE_VOCAB}:${SPLICE_VOCAB_SIZE} l1=${SPLICE_L1_PENALTY}"

ARGS=(
    --dataset "${DATASET}"
    --data-folder "${DATA_FOLDER}"
    --output "${OUTPUT_PATH}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --splice-model "${SPLICE_MODEL}"
    --splice-pretrained "${SPLICE_PRETRAINED}"
    --splice-vocab "${SPLICE_VOCAB}"
    --splice-vocab-size "${SPLICE_VOCAB_SIZE}"
    --splice-l1-penalty "${SPLICE_L1_PENALTY}"
    --dino-model "${DINO_MODEL}"
    --use-dino "${USE_DINO}"
)

python -u -m scripts.tools.cache_crp_features "${ARGS[@]}"

echo "Finished frozen feature cache; output=${OUTPUT_PATH}; $(date --iso-8601=seconds)"
conda deactivate
