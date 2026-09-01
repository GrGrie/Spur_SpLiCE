#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_KL_only
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

# CRP by default. Submit CQT with:
#   sbatch --export=ALL,KL_METHOD=cqt scripts/train_kl_only.sbatch
KL_METHOD="${KL_METHOD:-crp}" # crp | cqt
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"

# This ablation trains the backbone only with the confidence-weighted relational KL.
SIMCLR_WEIGHT="0"
SPLICE_WEIGHT="${SPLICE_WEIGHT:-1.0}"
CRP_START_EPOCH="0"
CRP_WARMUP_EPOCHS="0"
CRP_DECAY_START_EPOCH="0"
CRP_DECAY_END_EPOCH="0"
CRP_GRAPH_POSITIVES="false"
WANDB_EXTRA_TAGS="machine_cluster,ablation_kl_only,simclr_weight_0"

export PROJECT_DIR SIMCLR_WEIGHT SPLICE_WEIGHT CRP_START_EPOCH CRP_WARMUP_EPOCHS
export CRP_DECAY_START_EPOCH CRP_DECAY_END_EPOCH CRP_GRAPH_POSITIVES WANDB_EXTRA_TAGS

case "${KL_METHOD}" in
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
        echo "ERROR: KL_METHOD must be crp or cqt; got ${KL_METHOD}" >&2
        exit 2
        ;;
esac
