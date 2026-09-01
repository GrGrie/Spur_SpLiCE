#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CQT
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

# =============================================================================
# CQT CONFIGURATION — normally edit only this block.
# Submit with: sbatch scripts/train_cqt.sbatch
# =============================================================================
MODE="cqt_relational"
DATASET="waterbirds"             # waterbirds | celeba | spur_cifar10
MODEL="resnet18_large"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}" # sbatch --array=0-2 for three seeds.

# Main SSL/CQT sweep parameters.
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
SPLICE_WEIGHT="${SPLICE_WEIGHT:-0.01}"
CRP_TEMPERATURE="${CRP_TEMPERATURE:-0.25}" # Student relational temperature, shared trainer name.
CRP_START_EPOCH="${CRP_START_EPOCH:-10}"
CRP_WARMUP_EPOCHS="${CRP_WARMUP_EPOCHS:-10}"
CRP_DECAY_START_EPOCH="${CRP_DECAY_START_EPOCH:-200}"
CRP_DECAY_END_EPOCH="${CRP_DECAY_END_EPOCH:-350}"
CRP_GRAPH_POSITIVES="${CRP_GRAPH_POSITIVES:-true}"
USE_DINO="${USE_DINO:-true}"
COBALT="${COBALT:-false}"
COBALT_CONCEPTS_PATH="${COBALT_CONCEPTS_PATH:-}"

# CQT factor proposal and quotient-transport hyperparameters.
CQT_MIN_CONCEPT_FREQUENCY="0.01"   # Try 0.005, 0.01, 0.02.
CQT_MAX_CONCEPT_FREQUENCY="0.95"   # Try 0.90, 0.95, 0.98.
CQT_TEXT_SIMILARITY_THRESHOLD="0.8" # Try 0.75, 0.8, 0.85.
CQT_COACTIVATION_THRESHOLD="0.35"  # Try 0.25, 0.35, 0.5.
CQT_MIN_GROUP_SIZE="1"
CQT_MAX_CANDIDATE_GROUPS="128"     # Try 64, 128, 256.
CQT_MAX_FACTORS="8"                # Try 4, 8, 16.
CQT_MIN_STATE_SAMPLES="16"         # Try 8, 16, 32.
CQT_MIN_EXCLUSIVITY="0.8"          # Try 0.7, 0.8, 0.9.
CQT_MIN_CONTEXT_SIMILARITY="0.2"   # Try 0.1, 0.2, 0.3.
CQT_MIN_STATE_BALANCED_ACCURACY="0.7"
CQT_MIN_QUOTIENT_EFFICACY="0.04"
CQT_TRANSPORT_CANDIDATES="32"      # Try 16, 32, 64.
CQT_TRANSPORT_MASS="0.25"          # Try 0.1, 0.25, 0.5.
CQT_MIN_TRANSPORT_PAIRS="8"        # Try 4, 8, 16.
CQT_MIN_WORD_SIMILARITY="0.2"      # Try 0.1, 0.2, 0.3.
CQT_DINO_NEIGHBORS="10"            # Try 5, 10, 20.
CQT_DINO_DAMAGE_QUANTILE="0.6"
CQT_MIN_INTERVENTION_GAIN="1e-4"   # Try 0, 1e-4, 1e-3.
CQT_MIN_COVERAGE="0.01"            # Try 0.01, 0.05, 0.1.
CQT_NULL_TRIALS="32"
CQT_NULL_QUANTILE="0.95"
CQT_GRAPH_TOP_K="10"               # Try 5, 10, 20.
CQT_INDEGREE_FACTOR="3.0"          # Try 2, 3, 5.
CQT_SIMILARITY_CHUNK_SIZE="512"
CQT_ORACLE_AUDIT="true"            # Post-hoc Waterbirds audit; never builds the graph.
WATERBIRDS_METADATA_CSV="./datasets/waterbirds/metadata.csv"

# Graph/cache preparation.
CRP_AUTO_BUILD_GRAPH="true"
CRP_FORCE_REBUILD="false"
CRP_CACHE_BATCH_SIZE="64"
CRP_CACHE_NUM_WORKERS="4"
CRP_DINO_MODEL="vit_small_patch16_dinov3.lvd1689m"

# Cluster checkout containing the shared training implementation.
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"

# The renamed CRP script contains the shared, W&B-enabled relational pipeline.
# Variables above remain set when it is sourced, so it executes CQT only.
source "${PROJECT_DIR}/scripts/train_crp.sbatch"
