#!/usr/bin/env bash
#SBATCH --job-name=spur-splice-train
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/SLURM/train-%j.out
#SBATCH --error=outputs/SLURM/train-%j.err

set -euo pipefail

has_dataset=0
has_seed=0
for arg in "$@"; do
  case "${arg}" in
    --help|-h) has_dataset=1; has_seed=1 ;;
    --dataset|--dataset=*) has_dataset=1 ;;
    --seed|--seed=*) has_seed=1 ;;
  esac
done
if [[ "${has_dataset}" != "1" || "${has_seed}" != "1" ]]; then
  echo "Usage: sbatch scripts/run_training.sbatch --dataset DATASET --seed N [spur_splice.py options]" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source scripts/load_splice_cluster_env.sh
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  export SPUR_SPLICE_SCRATCH_ROOT="${SPUR_SPLICE_SCRATCH_ROOT:-${PROJECT_DIR}/tmp}"
fi

# Use a distinct output and artifact identity for each submission. Explicit
# options in "$@" take precedence, including --checkpoint_dir and --resume.
RUN_ID="${SLURM_JOB_ID:-local-$$}"
CHECKPOINT_ROOT="${SPUR_SPLICE_SCRATCH_ROOT}/manual/${RUN_ID}"
"${PYTHON_BIN}" -u spur_splice.py \
  --data_folder "${DATA_FOLDER:-/home/xar68reb/Datasets}" \
  --attempt_id "${RUN_ID}" \
  --checkpoint_dir "${CHECKPOINT_ROOT}" \
  --run_record "${CHECKPOINT_ROOT}/run.json" \
  "$@"
