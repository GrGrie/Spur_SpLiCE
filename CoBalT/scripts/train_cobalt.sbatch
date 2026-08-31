#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=CoBalT_waterbirds
#SBATCH --array=0-2
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/slurm-%x_%A_%a.out
#SBATCH --error=./output/slurm-%x_%A_%a.err

set -euo pipefail

# Paper run configuration. Change DATASET to celeba for the second supported benchmark.
PROJECT_DIR="/home/xar68reb/Spur_SpLiCE"
CONDA_ENV="grgrie-train"
DATA_FOLDER="/home/xar68reb/Datasets"
DATASET="waterbirds"             # waterbirds | celeba
SEED="${SLURM_ARRAY_TASK_ID:-0}"
NUM_WORKERS="5"
WANDB_PROJECT="Spur_SpLiCE"
WANDB_ENTITY="gsgrechkin-rptu"
WANDB_GROUP="CoBalT_${DATASET}_paper_reproduction"
WANDB_TAGS="baseline_cobalt,paper_reproduction,machine_cluster,dataset_${DATASET},seed_${SEED}"
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

echo "=== CoBalT paper reproduction: concept discovery ==="
python -u -m CoBalT.train_discovery \
    --dataset "${DATASET}" \
    --data-root "${DATA_FOLDER}" \
    --output "${DISCOVERY_CHECKPOINT}" \
    --seed "${SEED}" \
    --workers "${NUM_WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-run-name "CoBalT_${DATASET}_s${SEED}_discovery" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-tags "${WANDB_TAGS},stage_discovery"

echo "=== CoBalT paper reproduction: fixed concept inference ==="
python -u -m CoBalT.extract_concepts \
    --checkpoint "${DISCOVERY_CHECKPOINT}" \
    --data-root "${DATA_FOLDER}" \
    --output "${CONCEPT_ARTIFACT}" \
    --workers "${NUM_WORKERS}"

echo "=== CoBalT paper reproduction: robust classifier ==="
python -u -m CoBalT.train_classifier \
    --dataset "${DATASET}" \
    --data-root "${DATA_FOLDER}" \
    --concepts "${CONCEPT_ARTIFACT}" \
    --output-dir "${OUTPUT_DIR}/classifier" \
    --seed "${SEED}" \
    --workers "${NUM_WORKERS}" \
    --selectors "avg,ig,hg" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-run-name "CoBalT_${DATASET}_s${SEED}_classifier" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-tags "${WANDB_TAGS},stage_classifier"

conda deactivate
