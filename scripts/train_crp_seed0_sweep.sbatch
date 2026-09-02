#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CRP_s0_sweep
#SBATCH --array=0-7%2
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
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/scripts/train_crp_seed0_sweep.conf}"
load_splice_config "${CONFIG_FILE}"

variant_record="${CRP_TRAIN_SWEEP_VARIANTS[${SLURM_ARRAY_TASK_ID}]:-}"
[[ -n "${variant_record}" ]] || { echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
IFS='|' read -r SPLICE_WEIGHT CRP_TEMPERATURE CRP_GRAPH_POSITIVES <<< "${variant_record}"

WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS},sweep_task_${SLURM_ARRAY_TASK_ID}"
CONFIG_ALREADY_LOADED="true"
source "${PROJECT_DIR}/scripts/train_crp.sbatch"
