#!/usr/bin/env bash
#SBATCH --job-name=cospro-pipeline
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=3-12:00:00
#SBATCH --output=outputs/SLURM/cospro-pipeline-%j.out
#SBATCH --error=outputs/SLURM/cospro-pipeline-%j.err

# Full CoSpRo pipeline: SpLiCE cache -> concept groups -> teacher graph -> student -> results.
#
# Defaults live in one place, the Python CLI:
#   python -m cospro.cli.run_cospro_pipeline --help
# Override any value with an environment variable from the tables below or with a trailing
# option, for example:
#   EPOCHS=10 USE_WANDB=0 bash scripts/run_cospro_pipeline.sh --dry-run
#   sbatch scripts/run_cospro_pipeline.sh --dataset waterbirds --text-similarity-threshold 0.8
# Boolean variables take 1 or 0.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source "${PROJECT_DIR}/scripts/load_splice_cluster_env.sh"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  export SPUR_SPLICE_SCRATCH_ROOT="${SPUR_SPLICE_SCRATCH_ROOT:-${PROJECT_DIR}/tmp}"
fi

# Environment variable -> pipeline option.
VALUE_OPTIONS=(
  DATASET:--dataset DATA_FOLDER:--data-folder FEATURE_ROOT:--feature-root OUTPUT_ROOT:--output-root
  CACHE_BATCH_SIZE:--cache-batch-size CACHE_NUM_WORKERS:--cache-num-workers CACHE_DEVICE:--cache-device
  SPLICE_MODEL:--splice-model SPLICE_PRETRAINED:--splice-pretrained SPLICE_VOCAB:--splice-vocab
  SPLICE_VOCAB_SIZE:--splice-vocab-size SPLICE_VOCAB_FILE:--splice-vocab-file
  SPLICE_VOCAB_ORDER:--splice-vocab-order SPLICE_L1_PENALTY:--splice-l1-penalty
  MIN_CONCEPT_FREQUENCY:--min-concept-frequency MAX_CONCEPT_FREQUENCY:--max-concept-frequency
  TEXT_SIMILARITY_THRESHOLD:--text-similarity-threshold COACTIVATION_THRESHOLD:--coactivation-threshold
  MIN_GROUP_SIZE:--min-group-size SIMILARITY_CHUNK_SIZE:--similarity-chunk-size
  MAX_SELECTED_GROUPS:--max-selected-groups PROJECTED_NEIGHBORS:--projected-neighbors
  ACTIVATION_DIFFERENCE_QUANTILE:--activation-difference-quantile MIN_INTERVENTION_GAIN:--min-intervention-gain
  MIN_COVERAGE:--min-coverage GRAPH_TOP_K:--graph-top-k MAX_INDEGREE:--max-indegree
  INDEGREE_FACTOR:--indegree-factor NULL_TRIALS:--null-trials NULL_QUANTILE:--null-quantile
  AUDIT_SEED:--audit-seed ORTHOGONAL_TOLERANCE:--orthogonal-tolerance GRAPH_DEVICE:--graph-device
  NEIGHBOR_BACKEND:--neighbor-backend ANN_THRESHOLD:--ann-threshold ANN_TABLES:--ann-tables
  ANN_BUCKET_SIZE:--ann-bucket-size RESIDUAL_SPLICE_SIMILARITY_THRESHOLD:--residual-splice-similarity-threshold
  STUDY:--study SEED:--seed STUDENT_EXISTING:--student-existing ATTEMPT_ID:--attempt-id
  STUDENT_DEVICE:--student-device MODEL:--model HEAD:--head FEAT_DIM:--feat-dim EPOCHS:--epochs
  BATCH_SIZE:--batch-size NUM_WORKERS:--num-workers LEARNING_RATE:--learning-rate
  LR_DECAY_EPOCHS:--lr-decay-epochs LR_DECAY_RATE:--lr-decay-rate WEIGHT_DECAY:--weight-decay
  MOMENTUM:--momentum OPTIMIZER:--optimizer TEMPERATURE:--temp SIMCLR_WEIGHT:--simclr-weight
  SPLICE_WEIGHT:--splice-weight COSPRO_TEMPERATURE:--cospro-temperature
  COSPRO_START_EPOCH:--cospro-start-epoch COSPRO_WARMUP_EPOCHS:--cospro-warmup-epochs
  COSPRO_DECAY_START_EPOCH:--cospro-decay-start-epoch COSPRO_DECAY_END_EPOCH:--cospro-decay-end-epoch
  SSL_CROP_MIN:--ssl-crop-min RANK_EVAL_FREQ:--rank-eval-freq PRINT_FREQ:--print-freq
  SAVE_FREQ:--save-freq CHECKPOINT_KEEP_COUNT:--checkpoint-keep-count
  LINEAR_TRAIN_SPLIT:--linear-train-split LINEAR_EVAL_SPLIT:--linear-eval-split
  LINEAR_PROBE_MODE:--linear-probe-mode LINEAR_PROBE_FREQ:--linear-probe-freq
  LINEAR_PROBE_SOLVER:--linear-probe-solver LINEAR_PROBE_EPOCHS:--linear-probe-epochs
  LINEAR_PROBE_L2:--linear-probe-l2 LINEAR_PROBE_TOLERANCE:--linear-probe-tolerance
  LINEAR_PROBE_MAX_EPOCHS:--linear-probe-max-epochs WANDB_NAME:--wandb-name WANDB_GROUP:--wandb-group
  WANDB_TAGS:--wandb-tags WANDB_ENTITY:--entity
)
# Environment variable -> boolean option (1 adds --option, 0 adds --no-option).
BOOLEAN_OPTIONS=(
  USE_RESIDUAL_SPLICE_GATE:--use-residual-splice-gate KEEP_CHECKPOINTS:--keep-checkpoints
  DELETE_CHECKPOINTS_AFTER_TRAINING:--delete-checkpoints-after-training AMP:--amp
  CHANNELS_LAST:--channels-last CUDNN_ENABLED:--cudnn-enabled COSINE:--cosine
  LINEAR_SPURIOUS_PROBE:--linear-spurious-probe USE_WANDB:--use-wandb COLLECT_RESULTS:--collect-results
)

source scripts/announce_results.sh
PIPELINE_DATASET="${DATASET:-celeba}"
PIPELINE_STUDY="${STUDY:-${PIPELINE_DATASET}_cospro_pipeline}"
PIPELINE_OUTPUTS="${OUTPUT_ROOT:-${SPUR_SPLICE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs}}"
announce_results "${PIPELINE_OUTPUTS}/reports/${PIPELINE_STUDY}/results.json" "student run: ${PIPELINE_OUTPUTS}/seeds/${PIPELINE_STUDY}/seed_<NN>/cospro/<attempt>/run.json" "concept groups and teacher graph: ${PIPELINE_OUTPUTS}/shared/${PIPELINE_DATASET}/graphs/concept_groups/" "SpLiCE cache: ${FEATURE_ROOT:-${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE}/${PIPELINE_DATASET}/splice_dataset_cache/" "a --dataset or --study option overrides the names above"
pipeline_args=(--python "${PYTHON_BIN}")
for entry in "${VALUE_OPTIONS[@]}"; do
  variable="${entry%%:*}"
  if [[ -n "${!variable:-}" ]]; then
    pipeline_args+=("${entry#*:}" "${!variable}")
  fi
done
for entry in "${BOOLEAN_OPTIONS[@]}"; do
  variable="${entry%%:*}"
  option="${entry#*:}"
  case "${!variable:-}" in
    1) pipeline_args+=("${option}") ;;
    0) pipeline_args+=("--no-${option#--}") ;;
    "") ;;
    *) echo "ERROR: ${variable} must be 1 or 0, got '${!variable}'." >&2; exit 2 ;;
  esac
done
if [[ "${REBUILD_PREPROCESSING:-0}" == "1" ]]; then
  pipeline_args+=(--rebuild-preprocessing)
fi

"${PYTHON_BIN}" -u -m cospro.cli.run_cospro_pipeline "${pipeline_args[@]}" "$@"
