#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=SpLiCE_CRP_v2_audit
#SBATCH --array=0-0
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=./output/%x_%A_%a.out
#SBATCH --error=./output/%x_%A_%a.err

set -euo pipefail

# Label-free frozen audit only: this job does not read target/group annotations
# and does not start SSL training. Override --array=0-2 to repeat null calibration
# with three paired seeds.

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATASET="${DATASET:-waterbirds}"
CACHE_PATH="${CACHE_PATH:-outputs/crp/${DATASET}_train_features.pt}"
OUT_DIR="${OUT_DIR:-outputs/crp/${DATASET}}"
SEED_BASE="${SEED_BASE:-0}"
CRP_CONFIG_JSON="${CRP_CONFIG_JSON:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((SEED_BASE + TASK_ID))
OUTPUT_PATH="${OUTPUT_PATH:-${OUT_DIR}/teacher_graph_seed${SEED}.pt}"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}" ./output

if [[ ! -f "${CACHE_PATH}" ]]; then
    echo "CRP feature cache not found: ${CACHE_PATH}" >&2
    echo "Set CACHE_PATH to a version-1 cache described in README.md." >&2
    exit 2
fi

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

ARGS=(--cache "${CACHE_PATH}" --output "${OUTPUT_PATH}" --seed "${SEED}")
if [[ -n "${CRP_CONFIG_JSON}" ]]; then
    ARGS+=(--config "${CRP_CONFIG_JSON}")
fi

echo "=== SpLiCE-CRP v2 frozen audit ==="
echo "dataset=${DATASET} seed=${SEED} cache=${CACHE_PATH}"
echo "output=${OUTPUT_PATH} config=${CRP_CONFIG_JSON:-defaults}"

python -u -m splice.crp "${ARGS[@]}"

echo "Finished CRP v2 frozen audit; output=${OUTPUT_PATH}; $(date --iso-8601=seconds)"
conda deactivate
