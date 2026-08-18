#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=SpLiCE_CRP_v2_baselines
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=./output/%x_%j.out
#SBATCH --error=./output/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
CACHE_PATH="${CACHE_PATH:-outputs/crp/waterbirds_train_features.pt}"
OUT_DIR="${OUT_DIR:-outputs/crp/waterbirds_baselines}"
TOP_K="${TOP_K:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}" ./output
if [[ ! -f "${CACHE_PATH}" ]]; then
    echo "CRP feature cache not found: ${CACHE_PATH}" >&2
    exit 2
fi

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "=== CRP v2 matched baseline graphs ==="
echo "cache=${CACHE_PATH} output=${OUT_DIR} top_k=${TOP_K}"
python -u scripts/tools/build_crp_baseline_graphs.py \
  --cache "${CACHE_PATH}" \
  --output-dir "${OUT_DIR}" \
  --top-k "${TOP_K}" \
  --chunk-size "${CHUNK_SIZE}"

echo "Finished baseline graph build; $(date --iso-8601=seconds)"
conda deactivate
