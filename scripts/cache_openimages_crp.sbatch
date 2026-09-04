#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=SpurSpLiCE_OI_cache
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/CRPv2/slurm-%x_%j.out
#SBATCH --error=./output/CRPv2/slurm-%x_%j.err

set -euo pipefail

# One-time frozen cache preparation for the Open Images V7 vocabulary. This is
# a diagnostic/preparation job; it does not train the student network.
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/scripts/cache_openimages_crp.conf}"
source "${PROJECT_DIR}/scripts/load_config.sh"
load_splice_config "${CONFIG_FILE}"

CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
DATASET="${DATASET:-waterbirds}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/crp/${DATASET}_train_features_oi_v7.pt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${OUTPUT_PATH}")" ./output

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "=== Open Images V7 SpLiCE/CRP frozen feature cache ==="
echo "dataset=${DATASET} data=${DATA_FOLDER} output=${OUTPUT_PATH}"
echo "vocabulary=openimages_v7:all labels; time_limit=02:00:00"

python -u -m scripts.tools.cache_crp_features \
    --dataset "${DATASET}" \
    --data-folder "${DATA_FOLDER}" \
    --output "${OUTPUT_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --splice-model "open_clip:ViT-B-32" \
    --splice-pretrained "laion2b_s34b_b79k" \
    --splice-vocab "openimages_v7" \
    --splice-vocab-size "-1" \
    --splice-l1-penalty "0.25"

echo "Finished Open Images V7 frozen cache; output=${OUTPUT_PATH}; $(date --iso-8601=seconds)"
conda deactivate
