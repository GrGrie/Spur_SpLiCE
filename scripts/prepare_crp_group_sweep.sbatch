#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CRP_groups
#SBATCH --array=0-11%2
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/CRPv2/slurm-%x_%A_%a.out
#SBATCH --error=./output/CRPv2/slurm-%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/scripts/prepare_crp_group_sweep.conf}"
source "${PROJECT_DIR}/scripts/load_config.sh"
load_splice_config "${CONFIG_FILE}"

variant_record="${CRP_SWEEP_VARIANTS[${SLURM_ARRAY_TASK_ID}]:-}"
[[ -n "${variant_record}" ]] || { echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
IFS='|' read -r RELATIONAL_GRAPH_VARIANT CRP_MIN_GROUP_SIZE \
    CRP_TEXT_SIMILARITY_THRESHOLD CRP_COACTIVATION_THRESHOLD \
    CRP_MAX_SELECTED_GROUPS <<< "${variant_record}"

if [[ -n "${GRAPH_VARIANT_PREFIX}" ]]; then
    RELATIONAL_GRAPH_VARIANT="${GRAPH_VARIANT_PREFIX}_${RELATIONAL_GRAPH_VARIANT}"
fi

CONFIG_ALREADY_LOADED="true"
source "${PROJECT_DIR}/scripts/train_crp.sbatch"

# A NO_GO exit is expected for rejected audit candidates and should not hide
# the summary in Slurm output.
python -u -m scripts.tools.summarize_crp_audit "${RELATIONAL_TEACHER_GRAPH}" || true

REPORT_PATH="$(dirname "${RELATIONAL_TEACHER_GRAPH}")/graph_audit.html"
python -u -m scripts.tools.render_concept_ablation_examples \
    --cache "${CRP_CACHE_PATH}" \
    --graph "${RELATIONAL_TEACHER_GRAPH}" \
    --data-folder "${DATA_FOLDER}" \
    --scope selected \
    --max-interventions 12 \
    --edges-per-group 1 \
    --output "${REPORT_PATH}"

echo "[INFO] Graph audit report: ${PROJECT_DIR}/${REPORT_PATH}"
