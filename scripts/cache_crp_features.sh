#!/usr/bin/env bash
#SBATCH --job-name=crp-cache
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/SLURM/crp-cache-%j.out
#SBATCH --error=outputs/SLURM/crp-cache-%j.err

set -euo pipefail
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [OUTPUT.pt] (set DATA_FOLDER for the dataset root)" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source scripts/load_splice_cluster_env.sh
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
CACHE_ROOT="${SPUR_SPLICE_SCRATCH_ROOT:-${SPUR_SPLICE_ARTIFACT_ROOT:-${PROJECT_DIR}/tmp}}"
CACHE_PATH="${1:-${CACHE_ROOT}/features/Spur_SpLiCE/waterbirds_crp/crp_features.pt}"
mkdir -p "$(dirname "${CACHE_PATH}")"

"${PYTHON_BIN}" -u -m scripts.tools.cache_crp_features \
  --dataset waterbirds \
  --data-folder "${DATA_FOLDER}" \
  --output "${CACHE_PATH}" \
  --splice-model open_clip:ViT-B-32 \
  --splice-pretrained laion2b_s34b_b79k \
  --splice-vocab openimages_v7 \
  --splice-vocab-size -1 \
  --splice-l1-penalty 0.25
