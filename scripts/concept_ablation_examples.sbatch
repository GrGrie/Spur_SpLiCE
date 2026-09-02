#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_examples
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/CRPv2/slurm-%x_%j.out
#SBATCH --error=./output/CRPv2/slurm-%x_%j.err

set -euo pipefail

# CRP by default. Submit a CQT version with:
#   sbatch --export=ALL,REPORT_METHOD=cqt scripts/concept_ablation_examples.sbatch
REPORT_METHOD="${REPORT_METHOD:-crp}" # crp | cqt
REPORT_SCOPE="${REPORT_SCOPE:-selected}" # selected | all
REPORT_MAX_INTERVENTIONS="${REPORT_MAX_INTERVENTIONS:-12}" # 0 = show every eligible item.
REPORT_EDGES_PER_GROUP="${REPORT_EDGES_PER_GROUP:-1}" # Typical retained edges shown per group.
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
PREPARE_ONLY="true"

export PROJECT_DIR SEED PREPARE_ONLY
case "${REPORT_METHOD}" in
    crp)
        MODE="crp_relational"
        export MODE
        source "${PROJECT_DIR}/scripts/train_crp.sbatch"
        ;;
    cqt)
        MODE="cqt_relational"
        export MODE
        source "${PROJECT_DIR}/scripts/train_cqt.sbatch"
        ;;
    *)
        echo "ERROR: REPORT_METHOD must be crp or cqt; got ${REPORT_METHOD}" >&2
        exit 2
        ;;
esac

REPORT_DIR="$(dirname "${RELATIONAL_TEACHER_GRAPH}")"
REPORT_PATH="${REPORT_DIR}/graph_audit.html"
mkdir -p "${REPORT_DIR}"
python -u -m scripts.tools.render_concept_ablation_examples \
    --cache "${CRP_CACHE_PATH}" \
    --graph "${RELATIONAL_TEACHER_GRAPH}" \
    --data-folder "${DATA_FOLDER}" \
    --scope "${REPORT_SCOPE}" \
    --max-interventions "${REPORT_MAX_INTERVENTIONS}" \
    --edges-per-group "${REPORT_EDGES_PER_GROUP}" \
    --output "${REPORT_PATH}"

echo "[INFO] Copy or open this single report file: ${PROJECT_DIR}/${REPORT_PATH}"
conda deactivate
