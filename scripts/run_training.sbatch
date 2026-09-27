#!/usr/bin/env bash
#SBATCH --job-name=cospro-train
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
seed_value=""
study_value=""
arm_value=""
expecting=""
for arg in "$@"; do
  case "${expecting}" in
    dataset) dataset_value="${arg}" ;;
    seed) seed_value="${arg}" ;;
    study) study_value="${arg}" ;;
    arm) arm_value="${arg}" ;;
  esac
  expecting=""
  case "${arg}" in
    --help|-h) has_dataset=1; has_seed=1 ;;
    --dataset) has_dataset=1; expecting=dataset ;;
    --dataset=*) has_dataset=1; dataset_value="${arg#--dataset=}" ;;
    --seed) has_seed=1; expecting=seed ;;
    --seed=*) has_seed=1; seed_value="${arg#--seed=}" ;;
    --study) expecting=study ;;
    --study=*) study_value="${arg#--study=}" ;;
    --arm) expecting=arm ;;
    --arm=*) arm_value="${arg#--arm=}" ;;
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

# Every submission is one attempt of a study arm, recorded where Git carries it:
# outputs/seeds/<study>/seed_<NN>/<arm>/<job id>/. --study (default manual) and --arm (default run)
# name it; checkpoints and feature tensors still go to scratch. Explicit options in "$@" take
# precedence, including --checkpoint_dir and --resume.
RUN_ID="${SLURM_JOB_ID:-local-$$}"
STUDY="${study_value:-manual}"
ARM="${arm_value:-run}"
if [[ "${seed_value}" =~ ^[0-9]+$ ]]; then
  printf -v SEED_DIRECTORY 'seed_%02d' "$((10#${seed_value}))"
else
  SEED_DIRECTORY="seed_${seed_value}"
fi
OUTPUT_ROOT="${SPUR_SPLICE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
RECORD_ROOT="${OUTPUT_ROOT}/seeds/${STUDY}/${SEED_DIRECTORY}/${ARM}/${RUN_ID}"
source scripts/announce_results.sh
announce_results "${RECORD_ROOT}/run.json" "training folder: ${RECORD_ROOT}/training/<storage name>/ (binaries on scratch)" "study summary: ${OUTPUT_ROOT}/reports/${STUDY}/summary.md" "results book: ${OUTPUT_ROOT}/reports/results/README.md" "W&B: the run link printed by wandb below"
# With a teacher graph and no explicit --splice_mode, train the CoSpRo student: the cospro_student
# preset (cospro/config/presets.py) sets its values and later options in "$@" override them.
automatic_graph_args=()
if [[ "${has_graph}" == "1" && "${has_splice_mode}" != "1" ]]; then
  automatic_graph_args+=(--preset cospro_student)
elif [[ "${has_graph}" != "1" && "${has_splice_mode}" != "1" ]]; then
  # Shared results are stored under the canonical dataset name, so the directory that matches the
  # requested spelling names the dataset. Python keeps the canonical names; this shell does not.
  SHARED_ROOT="${SPUR_SPLICE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs}/shared"
  requested_dataset="$(printf '%s' "${dataset_value}" | tr '[:upper:]' '[:lower:]' | tr -d '_-')"
  GRAPH_ROOT="${SHARED_ROOT}/${dataset_value}/graphs"
  for shared_dataset in "${SHARED_ROOT}"/*/; do
    [[ -d "${shared_dataset}" ]] || continue
    shared_name="$(basename "${shared_dataset}")"
    if [[ "$(printf '%s' "${shared_name}" | tr '[:upper:]' '[:lower:]' | tr -d '_-')" == "${requested_dataset}" ]]; then
      GRAPH_ROOT="${shared_dataset}graphs"
      break
    fi
  done
  graph_candidates=()
  if [[ -d "${GRAPH_ROOT}" ]]; then
    mapfile -d '' graph_candidates < <(find "${GRAPH_ROOT}" -type f -name teacher_graph.json -print0 | sort -z)
  fi
  if [[ "${#graph_candidates[@]}" == "1" ]]; then
    automatic_graph_args+=(--preset cospro_student --cospro_teacher_graph "${graph_candidates[0]}")
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
  --study "${STUDY}" \
  --arm "${ARM}" \
  --attempt_id "${RUN_ID}" \
  --artifact_dir "${RECORD_ROOT}/training" \
  --checkpoint_dir "${RECORD_ROOT}/training" \
  --run_record "${RECORD_ROOT}/run.json" \
  "${automatic_graph_args[@]}" \
  "$@"
# Refresh the study summary and the results book with this run; neither can fail the finished training.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  "${PYTHON_BIN}" -u -m cospro.cli.summarize_study --study "${STUDY}" || echo "[WARNING] Study summary failed." >&2
  "${PYTHON_BIN}" -u -m cospro.cli.build_results_book || echo "[WARNING] Results book failed." >&2
fi
