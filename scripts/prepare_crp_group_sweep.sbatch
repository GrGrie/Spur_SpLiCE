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
SEED="0"
PREPARE_ONLY="true"
CRP_FORCE_REBUILD="${CRP_FORCE_REBUILD:-false}"
CRP_NULL_TRIALS="32"
GRAPH_VARIANT_PREFIX="${GRAPH_VARIANT_PREFIX:-}"

# This is a frozen, label-free grouping audit. Task 0 preserves the current
# singleton-heavy graph; tasks 1-11 vary group size, text/coactivation thresholds,
# and the cap on null-passing groups before any downstream metric is inspected.
case "${SLURM_ARRAY_TASK_ID}" in
    0)
        RELATIONAL_GRAPH_VARIANT="g1_t080_c035_all"
        CRP_MIN_GROUP_SIZE="1"; CRP_TEXT_SIMILARITY_THRESHOLD="0.80"
        CRP_COACTIVATION_THRESHOLD="0.35"; CRP_MAX_SELECTED_GROUPS="0"
        ;;
    1)
        RELATIONAL_GRAPH_VARIANT="g2_t075_c025_k12"
        CRP_MIN_GROUP_SIZE="2"; CRP_TEXT_SIMILARITY_THRESHOLD="0.75"
        CRP_COACTIVATION_THRESHOLD="0.25"; CRP_MAX_SELECTED_GROUPS="12"
        ;;
    2)
        RELATIONAL_GRAPH_VARIANT="g2_t070_c020_k12"
        CRP_MIN_GROUP_SIZE="2"; CRP_TEXT_SIMILARITY_THRESHOLD="0.70"
        CRP_COACTIVATION_THRESHOLD="0.20"; CRP_MAX_SELECTED_GROUPS="12"
        ;;
    3)
        RELATIONAL_GRAPH_VARIANT="g3_t070_c020_k8"
        CRP_MIN_GROUP_SIZE="3"; CRP_TEXT_SIMILARITY_THRESHOLD="0.70"
        CRP_COACTIVATION_THRESHOLD="0.20"; CRP_MAX_SELECTED_GROUPS="8"
        ;;
    4)
        RELATIONAL_GRAPH_VARIANT="g2_t065_c025_k12"
        CRP_MIN_GROUP_SIZE="2"; CRP_TEXT_SIMILARITY_THRESHOLD="0.65"
        CRP_COACTIVATION_THRESHOLD="0.25"; CRP_MAX_SELECTED_GROUPS="12"
        ;;
    5)
        RELATIONAL_GRAPH_VARIANT="g3_t065_c015_k8"
        CRP_MIN_GROUP_SIZE="3"; CRP_TEXT_SIMILARITY_THRESHOLD="0.65"
        CRP_COACTIVATION_THRESHOLD="0.15"; CRP_MAX_SELECTED_GROUPS="8"
        ;;
    6)
        RELATIONAL_GRAPH_VARIANT="g1_t075_c020_k16"
        CRP_MIN_GROUP_SIZE="1"; CRP_TEXT_SIMILARITY_THRESHOLD="0.75"
        CRP_COACTIVATION_THRESHOLD="0.20"; CRP_MAX_SELECTED_GROUPS="16"
        ;;
    7)
        RELATIONAL_GRAPH_VARIANT="g1_t070_c015_k16"
        CRP_MIN_GROUP_SIZE="1"; CRP_TEXT_SIMILARITY_THRESHOLD="0.70"
        CRP_COACTIVATION_THRESHOLD="0.15"; CRP_MAX_SELECTED_GROUPS="16"
        ;;
    8)
        RELATIONAL_GRAPH_VARIANT="g2_t080_c030_k16"
        CRP_MIN_GROUP_SIZE="2"; CRP_TEXT_SIMILARITY_THRESHOLD="0.80"
        CRP_COACTIVATION_THRESHOLD="0.30"; CRP_MAX_SELECTED_GROUPS="16"
        ;;
    9)
        RELATIONAL_GRAPH_VARIANT="g2_t070_c015_k16"
        CRP_MIN_GROUP_SIZE="2"; CRP_TEXT_SIMILARITY_THRESHOLD="0.70"
        CRP_COACTIVATION_THRESHOLD="0.15"; CRP_MAX_SELECTED_GROUPS="16"
        ;;
    10)
        RELATIONAL_GRAPH_VARIANT="g3_t060_c015_k12"
        CRP_MIN_GROUP_SIZE="3"; CRP_TEXT_SIMILARITY_THRESHOLD="0.60"
        CRP_COACTIVATION_THRESHOLD="0.15"; CRP_MAX_SELECTED_GROUPS="12"
        ;;
    11)
        RELATIONAL_GRAPH_VARIANT="g4_t060_c010_k8"
        CRP_MIN_GROUP_SIZE="4"; CRP_TEXT_SIMILARITY_THRESHOLD="0.60"
        CRP_COACTIVATION_THRESHOLD="0.10"; CRP_MAX_SELECTED_GROUPS="8"
        ;;
    *) echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

if [[ -n "${GRAPH_VARIANT_PREFIX}" ]]; then
    RELATIONAL_GRAPH_VARIANT="${GRAPH_VARIANT_PREFIX}_${RELATIONAL_GRAPH_VARIANT}"
fi

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
