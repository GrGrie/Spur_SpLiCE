#!/usr/bin/env bash
#SBATCH --job-name=Cache-Dataset
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/SLURM/Cache-Dataset-%j.out
#SBATCH --error=outputs/SLURM/Cache-Dataset-%j.err

set -euo pipefail
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [OUTPUT.pt] (set DATA_FOLDER for the dataset root)" >&2
  exit 2
fi

# Slurm executes a submitted script from a temporary spool copy, so
# BASH_SOURCE[0] is not the path inside the checkout.  SLURM_SUBMIT_DIR is
# the directory from which sbatch was invoked and remains the repository
# root for the documented invocation.  Keep the BASH_SOURCE fallback for
# direct local execution outside Slurm.
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
