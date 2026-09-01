#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CQT_s0_sweep
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
CRP_START_EPOCH="10"
CRP_WARMUP_EPOCHS="10"
CRP_DECAY_START_EPOCH="0"
CRP_DECAY_END_EPOCH="0"
CRP_TEMPERATURE="0.25"
CQT_MIN_QUOTIENT_EFFICACY="0.04"
CQT_DINO_DAMAGE_QUANTILE="0.60"
CQT_MAX_FACTORS="8"
RELATIONAL_GRAPH_VARIANT="current"

# Tasks 0-5 use exactly one frozen graph to separate sampler, graph-positive,
# and KL effects. Tasks 6-7 compare wider frozen graphs with graph positives
# only. Task 4 is intentionally high-risk; submit 0-3 first if GPU time is tight.
case "${SLURM_ARRAY_TASK_ID}" in
    0) SPLICE_WEIGHT="0"; CRP_GRAPH_POSITIVES="false" ;;
    1) SPLICE_WEIGHT="0"; CRP_GRAPH_POSITIVES="true" ;;
    2) SPLICE_WEIGHT="10"; CRP_GRAPH_POSITIVES="false" ;;
    3) SPLICE_WEIGHT="100"; CRP_GRAPH_POSITIVES="false" ;;
    4) SPLICE_WEIGHT="1000"; CRP_GRAPH_POSITIVES="false" ;;
    5) SPLICE_WEIGHT="100"; CRP_GRAPH_POSITIVES="true" ;;
    6)
        SPLICE_WEIGHT="0"; CRP_GRAPH_POSITIVES="true"
        CQT_MIN_QUOTIENT_EFFICACY="0.01"; CQT_DINO_DAMAGE_QUANTILE="0.60"
        CQT_MAX_FACTORS="16"; RELATIONAL_GRAPH_VARIANT="q001_d060_wide"
        ;;
    7)
        SPLICE_WEIGHT="0"; CRP_GRAPH_POSITIVES="true"
        CQT_MIN_QUOTIENT_EFFICACY="0.01"; CQT_DINO_DAMAGE_QUANTILE="0.60"
        CQT_MAX_FACTORS="16"; RELATIONAL_GRAPH_VARIANT="q001_d060_grouped"
        ;;
    *) echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

WANDB_EXTRA_TAGS="machine_cluster,sweep_cqt_seed0,sweep_task_${SLURM_ARRAY_TASK_ID}"
source "${PROJECT_DIR}/scripts/train_cqt.sbatch"

