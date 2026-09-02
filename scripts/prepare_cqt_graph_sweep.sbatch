#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CQT_graphs
#SBATCH --array=0-5%2
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
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/scripts/prepare_cqt_graph_sweep.conf}"
source "${PROJECT_DIR}/scripts/load_config.sh"
load_splice_config "${CONFIG_FILE}"

variant_record="${CQT_SWEEP_VARIANTS[${SLURM_ARRAY_TASK_ID}]:-}"
[[ -n "${variant_record}" ]] || { echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
IFS='|' read -r RELATIONAL_GRAPH_VARIANT CQT_MIN_QUOTIENT_EFFICACY \
    CQT_DINO_DAMAGE_QUANTILE CQT_MAX_FACTORS CQT_MIN_GROUP_SIZE \
    CQT_TEXT_SIMILARITY_THRESHOLD CQT_COACTIVATION_THRESHOLD <<< "${variant_record}"

CONFIG_ALREADY_LOADED="true"
source "${PROJECT_DIR}/scripts/train_cqt.sbatch"
