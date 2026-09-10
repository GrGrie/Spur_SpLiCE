#!/usr/bin/env bash
#SBATCH --job-name=SpLiCE-Dataset-Cache
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/SLURM/%x-%j.out
#SBATCH --error=outputs/SLURM/%x-%j.err

set -euo pipefail
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [OUTPUT_ROOT] (set DATASET and DATA_FOLDER as needed)" >&2
  exit 2
fi

if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

cd "${PROJECT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source "${PROJECT_DIR}/scripts/load_splice_cluster_env.sh"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

DATASET="${DATASET:-waterbirds}"
DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
DEFAULT_OUTPUT_ROOT="${SPUR_SPLICE_SCRATCH_ROOT:-${SPUR_SPLICE_ARTIFACT_ROOT:-${PROJECT_DIR}/tmp}}/features/Spur_SpLiCE"
OUTPUT_ROOT="${1:-${DEFAULT_OUTPUT_ROOT}}"

"${PYTHON_BIN}" -u -m scripts.tools.cache_splice_dataset \
  --dataset "${DATASET}" \
  --data-folder "${DATA_FOLDER}" \
  --output-root "${OUTPUT_ROOT}" \
  --splice-model open_clip:ViT-B-32 \
  --splice-pretrained laion2b_s34b_b79k \
  --splice-vocab openimages_v7 \
  --splice-vocab-size -1 \
  --splice-l1-penalty 0.25
