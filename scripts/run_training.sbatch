#!/usr/bin/env bash
#SBATCH --job-name=cospro-train-celeba_s4
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --output=outputs/SLURM/train-%j.out
#SBATCH --error=outputs/SLURM/train-%j.err

set -euo pipefail

has_dataset=0
has_seed=0
has_graph=0
has_splice_mode=0
dataset_value=""
expect_dataset=0
for arg in "$@"; do
  if [[ "${expect_dataset}" == "1" ]]; then
    dataset_value="${arg}"
    expect_dataset=0
  fi
  case "${arg}" in
    --help|-h) has_dataset=1; has_seed=1 ;;
    --dataset) has_dataset=1; expect_dataset=1 ;;
    --dataset=*) has_dataset=1; dataset_value="${arg#--dataset=}" ;;
    --seed|--seed=*) has_seed=1 ;;
    --cospro_teacher_graph|--cospro_teacher_graph=*|--crp_teacher_graph|--crp_teacher_graph=*) has_graph=1 ;;
    --splice_mode|--splice_mode=*) has_splice_mode=1 ;;
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
source scripts/announce_results.sh
announce_results "${CHECKPOINT_ROOT}/run.json" "training folder: ${CHECKPOINT_ROOT}/<storage name>/ (or the --checkpoint_dir you passed)" "W&B: the run link printed by wandb below"
automatic_graph_args=()
if [[ "${has_graph}" == "1" && "${has_splice_mode}" != "1" ]]; then
  automatic_graph_args+=(
    --batch_size "${COSPRO_BATCH_SIZE:-128}"
    --num_workers "${COSPRO_NUM_WORKERS:-4}"
    --temp "${COSPRO_SIMCLR_TEMPERATURE:-0.05}"
    --splice_mode cospro_relational
    --splice_weight "${COSPRO_SPLICE_WEIGHT:-0.5}"
    --cospro_temperature "${COSPRO_TEMPERATURE:-0.25}"
  )
elif [[ "${has_graph}" != "1" && "${has_splice_mode}" != "1" ]]; then
  canonical_dataset="$(printf '%s' "${dataset_value}" | tr '[:upper:]' '[:lower:]' | tr -d '_-')"
  case "${canonical_dataset}" in
    celeba) canonical_dataset="celeba" ;;
    waterbirds) canonical_dataset="waterbirds" ;;
    spurcifar10) canonical_dataset="spur_cifar10" ;;
  esac
  GRAPH_ROOT="${SPUR_SPLICE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs}/shared/${canonical_dataset}/graphs"
  graph_candidates=()
  if [[ -d "${GRAPH_ROOT}" ]]; then
    mapfile -d '' graph_candidates < <(find "${GRAPH_ROOT}" -type f -name teacher_graph.json -print0 | sort -z)
  fi
  if [[ "${#graph_candidates[@]}" == "1" ]]; then
    automatic_graph_args+=(
      --batch_size "${COSPRO_BATCH_SIZE:-128}"
      --num_workers "${COSPRO_NUM_WORKERS:-4}"
      --temp "${COSPRO_SIMCLR_TEMPERATURE:-0.05}"
      --splice_mode cospro_relational
      --splice_weight "${COSPRO_SPLICE_WEIGHT:-0.5}"
      --cospro_temperature "${COSPRO_TEMPERATURE:-0.25}"
      --cospro_teacher_graph "${graph_candidates[0]}"
    )
    echo "[INFO] Using CoSpRo teacher graph: ${graph_candidates[0]}" >&2
  elif [[ "${#graph_candidates[@]}" -gt "1" ]]; then
    echo "ERROR: found multiple teacher graphs for ${dataset_value} under ${GRAPH_ROOT}." >&2
    echo "Pass --cospro_teacher_graph PATH to select the intended graph." >&2
    exit 2
  else
    echo "[WARNING] No teacher graph found under ${GRAPH_ROOT}; running standard SimCLR." >&2
    echo "Pass --cospro_teacher_graph PATH if the graph is stored elsewhere." >&2
  fi
fi
"${PYTHON_BIN}" -u spur_splice.py \
  ${DATA_FOLDER:+--data_folder "${DATA_FOLDER}"} \
  --attempt_id "${RUN_ID}" \
  --checkpoint_dir "${CHECKPOINT_ROOT}" \
  --run_record "${CHECKPOINT_ROOT}/run.json" \
  "${automatic_graph_args[@]}" \
  "$@"
