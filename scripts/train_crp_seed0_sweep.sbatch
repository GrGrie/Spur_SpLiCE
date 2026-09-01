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
#SBATCH --output=./output/CRPv2/slurm-%x_%A_%a.out
#SBATCH --error=./output/CRPv2/slurm-%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
SEED="0"
CRP_AUTO_BUILD_GRAPH="false"
RELATIONAL_GRAPH_VARIANT="${CRP_SWEEP_GRAPH_VARIANT:-g2_t070_c020_k12}"
CRP_START_EPOCH="10"
CRP_WARMUP_EPOCHS="10"
CRP_DECAY_START_EPOCH="0"
CRP_DECAY_END_EPOCH="0"
CRP_GRAPH_POSITIVES="false"
CRP_TEMPERATURE="0.25"

# First isolate the effective KL scale under absolute graph confidence. Tasks
# 5-6 test temperature; task 7 re-enables graph positives at the central weight.
case "${SLURM_ARRAY_TASK_ID}" in
    0) SPLICE_WEIGHT="0" ;;
    1) SPLICE_WEIGHT="0.25" ;;
    2) SPLICE_WEIGHT="0.5" ;;
    3) SPLICE_WEIGHT="1.0" ;;
    4) SPLICE_WEIGHT="2.0" ;;
    5) SPLICE_WEIGHT="1.0"; CRP_TEMPERATURE="0.10" ;;
    6) SPLICE_WEIGHT="1.0"; CRP_TEMPERATURE="0.50" ;;
    7) SPLICE_WEIGHT="1.0"; CRP_GRAPH_POSITIVES="true" ;;
    *) echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

WANDB_EXTRA_TAGS="machine_cluster,sweep_crp_seed0,sweep_task_${SLURM_ARRAY_TASK_ID}"
source "${PROJECT_DIR}/scripts/train_crp.sbatch"

