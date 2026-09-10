#!/usr/bin/env bash
#SBATCH --job-name=celeba-cospro-pipeline
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/SLURM/celeba-cospro-%j.out
#SBATCH --error=outputs/SLURM/celeba-cospro-%j.err

set -euo pipefail

# Full CoSpRo pipeline configuration. Every value can also be overridden
# without editing this file, for example: EPOCHS=10 USE_WANDB=0 bash $0 --dry-run

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source "${PROJECT_DIR}/scripts/load_splice_cluster_env.sh"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
  export SPUR_SPLICE_SCRATCH_ROOT="${SPUR_SPLICE_SCRATCH_ROOT:-${PROJECT_DIR}/tmp}"
fi

# Dataset and storage
DATASET="${DATASET:-celeba}"
DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
FEATURE_ROOT="${FEATURE_ROOT:-${SPUR_SPLICE_SCRATCH_ROOT:-${SPUR_SPLICE_ARTIFACT_ROOT:-${PROJECT_DIR}/tmp}}/features/Spur_SpLiCE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SPUR_SPLICE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs}}"
REBUILD_PREPROCESSING="${REBUILD_PREPROCESSING:-0}"

# Cache construction
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-64}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
CACHE_DEVICE="${CACHE_DEVICE:-cuda}"
SPLICE_MODEL="${SPLICE_MODEL:-open_clip:ViT-B-32}"
SPLICE_PRETRAINED="${SPLICE_PRETRAINED:-laion2b_s34b_b79k}"
SPLICE_VOCAB="${SPLICE_VOCAB:-openimages_v7}"
SPLICE_VOCAB_SIZE="${SPLICE_VOCAB_SIZE:--1}"
SPLICE_L1_PENALTY="${SPLICE_L1_PENALTY:-0.25}"

# Concept grouping
MIN_CONCEPT_FREQUENCY="${MIN_CONCEPT_FREQUENCY:-0.01}"
MAX_CONCEPT_FREQUENCY="${MAX_CONCEPT_FREQUENCY:-0.95}"
TEXT_SIMILARITY_THRESHOLD="${TEXT_SIMILARITY_THRESHOLD:-0.82}"
COACTIVATION_THRESHOLD="${COACTIVATION_THRESHOLD:-0.35}"
MIN_GROUP_SIZE="${MIN_GROUP_SIZE:-1}"
SIMILARITY_CHUNK_SIZE="${SIMILARITY_CHUNK_SIZE:-512}"

# Teacher-graph audit
MAX_SELECTED_GROUPS="${MAX_SELECTED_GROUPS:-0}"
PROJECTED_NEIGHBORS="${PROJECTED_NEIGHBORS:-20}"
ACTIVATION_DIFFERENCE_QUANTILE="${ACTIVATION_DIFFERENCE_QUANTILE:-0.75}"
MIN_INTERVENTION_GAIN="${MIN_INTERVENTION_GAIN:-0.0001}"
MIN_COVERAGE="${MIN_COVERAGE:-0.01}"
GRAPH_TOP_K="${GRAPH_TOP_K:-3}"
MAX_INDEGREE="${MAX_INDEGREE:-10}"
INDEGREE_FACTOR="${INDEGREE_FACTOR:-3.0}"
NULL_TRIALS="${NULL_TRIALS:-16}"
NULL_QUANTILE="${NULL_QUANTILE:-0.95}"
AUDIT_SEED="${AUDIT_SEED:-0}"
ORTHOGONAL_TOLERANCE="${ORTHOGONAL_TOLERANCE:-0.000001}"
USE_RESIDUAL_SPLICE_GATE="${USE_RESIDUAL_SPLICE_GATE:-1}"
RESIDUAL_SPLICE_SIMILARITY_THRESHOLD="${RESIDUAL_SPLICE_SIMILARITY_THRESHOLD:-0.25}"

# Student SSL training
STUDY="${STUDY:-}" # empty becomes <canonical-dataset>_cospro_pipeline
SEED="${SEED:-1}"
STUDENT_EXISTING="${STUDENT_EXISTING:-error}" # error, reuse, resume, or new-attempt
ATTEMPT_ID="${ATTEMPT_ID:-}"
STUDENT_DEVICE="${STUDENT_DEVICE:-cuda}"
MODEL="${MODEL:-resnet18_large}"
HEAD="${HEAD:-mlp}"
FEAT_DIM="${FEAT_DIM:-128}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
LR_DECAY_EPOCHS="${LR_DECAY_EPOCHS:-auto}"
LR_DECAY_RATE="${LR_DECAY_RATE:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MOMENTUM="${MOMENTUM:-0.9}"
OPTIMIZER="${OPTIMIZER:-SGD}"
TEMPERATURE="${TEMPERATURE:-0.05}"
SIMCLR_WEIGHT="${SIMCLR_WEIGHT:-1.0}"
SPLICE_WEIGHT="${SPLICE_WEIGHT:-0.5}"
COSPRO_TEMPERATURE="${COSPRO_TEMPERATURE:-0.25}"
COSPRO_START_EPOCH="${COSPRO_START_EPOCH:-10}"
COSPRO_WARMUP_EPOCHS="${COSPRO_WARMUP_EPOCHS:-10}"
COSPRO_DECAY_START_EPOCH="${COSPRO_DECAY_START_EPOCH:-0}"
COSPRO_DECAY_END_EPOCH="${COSPRO_DECAY_END_EPOCH:-0}"
SSL_CROP_MIN="${SSL_CROP_MIN:-0.2}"
RANK_EVAL_FREQ="${RANK_EVAL_FREQ:-100}"
PRINT_FREQ="${PRINT_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-50}"
CHECKPOINT_KEEP_COUNT="${CHECKPOINT_KEEP_COUNT:-2}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-1}"
DELETE_CHECKPOINTS_AFTER_TRAINING="${DELETE_CHECKPOINTS_AFTER_TRAINING:-1}"
AMP="${AMP:-1}"
CHANNELS_LAST="${CHANNELS_LAST:-1}"
CUDNN_ENABLED="${CUDNN_ENABLED:-1}"
COSINE="${COSINE:-0}"

# Linear evaluation and experiment tracking
LINEAR_TRAIN_SPLIT="${LINEAR_TRAIN_SPLIT:-ds_train}"
LINEAR_EVAL_SPLIT="${LINEAR_EVAL_SPLIT:-val}"
LINEAR_PROBE_MODE="${LINEAR_PROBE_MODE:-periodic}"
LINEAR_PROBE_FREQ="${LINEAR_PROBE_FREQ:-25}"
LINEAR_PROBE_SOLVER="${LINEAR_PROBE_SOLVER:-logistic}"
LINEAR_PROBE_EPOCHS="${LINEAR_PROBE_EPOCHS:-100}"
LINEAR_PROBE_L2="${LINEAR_PROBE_L2:-0.001}"
LINEAR_PROBE_TOLERANCE="${LINEAR_PROBE_TOLERANCE:-0.000001}"
LINEAR_PROBE_MAX_EPOCHS="${LINEAR_PROBE_MAX_EPOCHS:-200}"
LINEAR_SPURIOUS_PROBE="${LINEAR_SPURIOUS_PROBE:-1}"
USE_WANDB="${USE_WANDB:-0}"
COLLECT_RESULTS="${COLLECT_RESULTS:-1}"
WANDB_NAME="${WANDB_NAME:-CoSpRo}"
WANDB_GROUP="${WANDB_GROUP:-}" # empty follows STUDY
WANDB_TAGS="${WANDB_TAGS:-${DATASET},cospro,full-pipeline}"
WANDB_ENTITY="${WANDB_ENTITY:-gsgrechkin-rptu}"

pipeline_args=(
  --dataset "${DATASET}"
  --data-folder "${DATA_FOLDER}"
  --feature-root "${FEATURE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --python "${PYTHON_BIN}"
  --cache-batch-size "${CACHE_BATCH_SIZE}"
  --cache-num-workers "${CACHE_NUM_WORKERS}"
  --cache-device "${CACHE_DEVICE}"
  --splice-model "${SPLICE_MODEL}"
  --splice-pretrained "${SPLICE_PRETRAINED}"
  --splice-vocab "${SPLICE_VOCAB}"
  --splice-vocab-size "${SPLICE_VOCAB_SIZE}"
  --splice-l1-penalty "${SPLICE_L1_PENALTY}"
  --min-concept-frequency "${MIN_CONCEPT_FREQUENCY}"
  --max-concept-frequency "${MAX_CONCEPT_FREQUENCY}"
  --text-similarity-threshold "${TEXT_SIMILARITY_THRESHOLD}"
  --coactivation-threshold "${COACTIVATION_THRESHOLD}"
  --min-group-size "${MIN_GROUP_SIZE}"
  --similarity-chunk-size "${SIMILARITY_CHUNK_SIZE}"
  --max-selected-groups "${MAX_SELECTED_GROUPS}"
  --projected-neighbors "${PROJECTED_NEIGHBORS}"
  --activation-difference-quantile "${ACTIVATION_DIFFERENCE_QUANTILE}"
  --min-intervention-gain "${MIN_INTERVENTION_GAIN}"
  --min-coverage "${MIN_COVERAGE}"
  --graph-top-k "${GRAPH_TOP_K}"
  --max-indegree "${MAX_INDEGREE}"
  --indegree-factor "${INDEGREE_FACTOR}"
  --null-trials "${NULL_TRIALS}"
  --null-quantile "${NULL_QUANTILE}"
  --audit-seed "${AUDIT_SEED}"
  --orthogonal-tolerance "${ORTHOGONAL_TOLERANCE}"
  --residual-splice-similarity-threshold "${RESIDUAL_SPLICE_SIMILARITY_THRESHOLD}"
  --study "${STUDY}"
  --seed "${SEED}"
  --student-existing "${STUDENT_EXISTING}"
  --student-device "${STUDENT_DEVICE}"
  --model "${MODEL}"
  --head "${HEAD}"
  --feat-dim "${FEAT_DIM}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --learning-rate "${LEARNING_RATE}"
  --lr-decay-epochs "${LR_DECAY_EPOCHS}"
  --lr-decay-rate "${LR_DECAY_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --momentum "${MOMENTUM}"
  --optimizer "${OPTIMIZER}"
  --temp "${TEMPERATURE}"
  --simclr-weight "${SIMCLR_WEIGHT}"
  --splice-weight "${SPLICE_WEIGHT}"
  --cospro-temperature "${COSPRO_TEMPERATURE}"
  --cospro-start-epoch "${COSPRO_START_EPOCH}"
  --cospro-warmup-epochs "${COSPRO_WARMUP_EPOCHS}"
  --cospro-decay-start-epoch "${COSPRO_DECAY_START_EPOCH}"
  --cospro-decay-end-epoch "${COSPRO_DECAY_END_EPOCH}"
  --ssl-crop-min "${SSL_CROP_MIN}"
  --rank-eval-freq "${RANK_EVAL_FREQ}"
  --print-freq "${PRINT_FREQ}"
  --save-freq "${SAVE_FREQ}"
  --checkpoint-keep-count "${CHECKPOINT_KEEP_COUNT}"
  --linear-train-split "${LINEAR_TRAIN_SPLIT}"
  --linear-eval-split "${LINEAR_EVAL_SPLIT}"
  --linear-probe-mode "${LINEAR_PROBE_MODE}"
  --linear-probe-freq "${LINEAR_PROBE_FREQ}"
  --linear-probe-solver "${LINEAR_PROBE_SOLVER}"
  --linear-probe-epochs "${LINEAR_PROBE_EPOCHS}"
  --linear-probe-l2 "${LINEAR_PROBE_L2}"
  --linear-probe-tolerance "${LINEAR_PROBE_TOLERANCE}"
  --linear-probe-max-epochs "${LINEAR_PROBE_MAX_EPOCHS}"
  --wandb-name "${WANDB_NAME}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-tags "${WANDB_TAGS}"
  --entity "${WANDB_ENTITY}"
)

[[ "${REBUILD_PREPROCESSING}" == "1" ]] && pipeline_args+=(--rebuild-preprocessing)
[[ "${USE_RESIDUAL_SPLICE_GATE}" == "1" ]] || pipeline_args+=(--no-use-residual-splice-gate)
[[ "${KEEP_CHECKPOINTS}" == "1" ]] || pipeline_args+=(--no-keep-checkpoints)
[[ "${DELETE_CHECKPOINTS_AFTER_TRAINING}" == "1" ]] || pipeline_args+=(--no-delete-checkpoints-after-training)
[[ "${AMP}" == "1" ]] || pipeline_args+=(--no-amp)
[[ "${CHANNELS_LAST}" == "1" ]] || pipeline_args+=(--no-channels-last)
[[ "${CUDNN_ENABLED}" == "1" ]] || pipeline_args+=(--no-cudnn-enabled)
[[ "${COSINE}" == "1" ]] && pipeline_args+=(--cosine)
[[ "${LINEAR_SPURIOUS_PROBE}" == "1" ]] || pipeline_args+=(--no-linear-spurious-probe)
[[ "${USE_WANDB}" == "1" ]] && pipeline_args+=(--use-wandb)
[[ "${COLLECT_RESULTS}" == "1" ]] || pipeline_args+=(--no-collect-results)
[[ -n "${ATTEMPT_ID}" ]] && pipeline_args+=(--attempt-id "${ATTEMPT_ID}")

"${PYTHON_BIN}" -u -m scripts.tools.run_cospro_pipeline "${pipeline_args[@]}" "$@"
