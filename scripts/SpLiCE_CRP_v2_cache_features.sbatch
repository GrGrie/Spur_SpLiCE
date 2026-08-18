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
OUTPUT_PATH="${OUTPUT_PATH:-outputs/crp/${DATASET}_train_features.pt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DINO_MODEL="${DINO_MODEL:-vit_small_patch16_dinov3.lvd1689m}"
FORCE="${FORCE:-false}"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${OUTPUT_PATH}")" ./output

if [[ -f "${OUTPUT_PATH}" && "${FORCE}" != "true" ]]; then
    echo "Frozen feature cache already exists: ${OUTPUT_PATH}"
    echo "Set FORCE=true or choose another OUTPUT_PATH to rebuild it."
    exit 0
fi

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== SpLiCE-CRP v2 frozen feature cache ==="
echo "dataset=${DATASET} data=${DATA_FOLDER} output=${OUTPUT_PATH} dino=${DINO_MODEL}"

python -u -m scripts.tools.cache_crp_features \
    --dataset "${DATASET}" \
    --data-folder "${DATA_FOLDER}" \
    --output "${OUTPUT_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --dino-model "${DINO_MODEL}"

echo "Finished frozen feature cache; output=${OUTPUT_PATH}; $(date --iso-8601=seconds)"
conda deactivate
