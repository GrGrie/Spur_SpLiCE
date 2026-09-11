#!/usr/bin/env bash
#SBATCH --job-name=SpLiCE-Dataset-Cache
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/SLURM/%x-%j.out
#SBATCH --error=outputs/SLURM/%x-%j.err

set -euo pipefail
if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  echo "Usage: $0 DATASET [OUTPUT_ROOT]"
  echo "Datasets: waterbirds, celebA (celeba), spur_cifar10"
  echo "Set DATA_FOLDER to override the dataset location."
  echo "Caches are stored under OUTPUT_ROOT/<dataset>/splice_dataset_cache/<configuration>/."
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DATASET [OUTPUT_ROOT] (set DATA_FOLDER as needed)" >&2
  exit 2
fi
DATASET="$1"

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
DEFAULT_OUTPUT_ROOT="${SPUR_SPLICE_SCRATCH_ROOT:-${SPUR_SPLICE_ARTIFACT_ROOT:-${PROJECT_DIR}/tmp}}/features/Spur_SpLiCE"
# The Python cache builder adds the canonical dataset name and configuration.
OUTPUT_ROOT="${2:-${DEFAULT_OUTPUT_ROOT}}"

"${PYTHON_BIN}" -u -m scripts.tools.cache_splice_dataset \
  --dataset "${DATASET}" \
  --data-folder "${DATA_FOLDER}" \
  --output-root "${OUTPUT_ROOT}" \
  --splice-model open_clip:ViT-B-32 \
  --splice-pretrained laion2b_s34b_b79k \
  --splice-vocab openimages_v7 \
  --splice-vocab-size -1 \
  --splice-l1-penalty 0.25
