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
python -u -m CoBalT.train_discovery \
    --dataset "${DATASET}" \
    --data-root "${DATA_FOLDER}" \
    --output "${DISCOVERY_CHECKPOINT}" \
    --seed "${SEED}" \
    --workers "${NUM_WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-run-name "CoBalT_${DATASET}_s${SEED}_concept_check" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-tags "${WANDB_TAGS}"

echo "=== Extracting fixed CoBalT concept memberships ==="
python -u -m CoBalT.extract_concepts \
    --checkpoint "${DISCOVERY_CHECKPOINT}" \
    --data-root "${DATA_FOLDER}" \
    --output "${CONCEPT_ARTIFACT}" \
    --workers "${NUM_WORKERS}"

echo "[INFO] CoBalT concepts ready: ${PROJECT_DIR}/${CONCEPT_ARTIFACT}"
conda deactivate
