#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=SpurSpLiCE_SSL_graph_screen
#SBATCH --array=0-2%2
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/CRPv3/slurm-%x_%A_%a.out
#SBATCH --error=./output/CRPv3/slurm-%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
source "${PROJECT_DIR}/scripts/load_config.sh"
CONFIG_FILE="${PROJECT_DIR}/scripts/train_ssl_graph_shortlist.conf"
load_splice_config "${CONFIG_FILE}"

variant_record="${SSL_GRAPH_SHORTLIST_VARIANTS[${SLURM_ARRAY_TASK_ID}]:-}"
[[ -n "${variant_record}" ]] || { echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
IFS='|' read -r PILOT_LABEL MODE RELATIONAL_GRAPH_VARIANT CRP_MIN_GROUP_SIZE \
    CRP_TEXT_SIMILARITY_THRESHOLD CRP_COACTIVATION_THRESHOLD \
    CRP_MAX_SELECTED_GROUPS SPLICE_WEIGHT <<< "${variant_record}"

if [[ "${MODE}" == "none" ]]; then
    USE_DINO="false"
    COBALT="false"
fi

WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS},pilot_${PILOT_LABEL},array_task_${SLURM_ARRAY_TASK_ID}"
CONFIG_ALREADY_LOADED="true"
source "${PROJECT_DIR}/scripts/train_crp.sbatch"
