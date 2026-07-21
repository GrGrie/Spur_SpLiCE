#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --job-name=Waterbirds_SpLiCE_controls
#SBATCH --array=0-4
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=slurm-%x_%A_%a.out
#SBATCH --error=slurm-%x_%A_%a.err

set -euo pipefail

# Causal routing controls at a fixed q=.75 augmentation budget:
#   0 baseline; 1 semantic; 2 shuffled-score; 3 matched-random; 4 augment-all.
# Automatic discovery is restricted to five direction-consistent, deduplicated
# concepts. Tasks 1--3 route exactly the same number of training examples.

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"
DATASET="${DATASET:-waterbirds}"
MODEL="${MODEL:-resnet18_large}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TEMP="${TEMP:-0.05}"
PRINT_FREQ="${PRINT_FREQ:-25}"
RANK_EVAL_FREQ="${RANK_EVAL_FREQ:-0}"

SPLICE_AUTO_TOP_K="${SPLICE_AUTO_TOP_K:-5}"
SPLICE_SCORE_QUANTILE="${SPLICE_SCORE_QUANTILE:-0.75}"
SPLICE_SCORE_REDUCTION="${SPLICE_SCORE_REDUCTION:-max}"
SPLICE_BATCH_SIZE="${SPLICE_BATCH_SIZE:-128}"
SPLICE_NUM_WORKERS="${SPLICE_NUM_WORKERS:-1}"
SPLICE_L1_PENALTY="${SPLICE_L1_PENALTY:-0.25}"
SPLICE_MODEL="${SPLICE_MODEL:-open_clip:ViT-B-32}"
SPLICE_PRETRAINED="${SPLICE_PRETRAINED:-laion2b_s34b_b79k}"
SPLICE_VOCAB="${SPLICE_VOCAB:-laion}"
SPLICE_VOCAB_SIZE="${SPLICE_VOCAB_SIZE:-10000}"
SPLICE_SCORE_CACHE_DIR="${SPLICE_SCORE_CACHE_DIR:-outputs/splice_score_cache}"
SPLICE_REQUIRE_CONSISTENT_DIRECTION="${SPLICE_REQUIRE_CONSISTENT_DIRECTION:-true}"
SPLICE_DEDUPLICATE_CONCEPTS="${SPLICE_DEDUPLICATE_CONCEPTS:-true}"

LINEAR_TRAIN_SPLIT="${LINEAR_TRAIN_SPLIT:-ds_train}"
LINEAR_EVAL_SPLIT="${LINEAR_EVAL_SPLIT:-val}"
LINEAR_PROBE_FREQ="${LINEAR_PROBE_FREQ:-25}"
LINEAR_PROBE_EPOCHS="${LINEAR_PROBE_EPOCHS:-100}"
LINEAR_SPURIOUS_PROBE="${LINEAR_SPURIOUS_PROBE:-false}"
WANDB_NAME="${WANDB_NAME:-Spur_SpLiCE}"
WANDB_ENTITY="${WANDB_ENTITY:-gsgrechkin-rptu}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) RUN_LABEL="baseline"; SPLICE_MODE="none"; ROUTING_MODE="semantic" ;;
    1) RUN_LABEL="top5_semantic_q75"; SPLICE_MODE="augment"; ROUTING_MODE="semantic" ;;
    2) RUN_LABEL="top5_shuffled_q75"; SPLICE_MODE="augment"; ROUTING_MODE="shuffled" ;;
    3) RUN_LABEL="top5_random_q75"; SPLICE_MODE="augment"; ROUTING_MODE="random" ;;
    4) RUN_LABEL="top5_all"; SPLICE_MODE="augment"; ROUTING_MODE="all" ;;
    *) echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
WANDB_GROUP="${WANDB_GROUP:-${DATASET}_routing_controls_top5_seed${SEED}}"
WANDB_TAGS="${WANDB_TAGS:-dataset_${DATASET},seed_${SEED},family_routing_controls,route_${ROUTING_MODE},auto_top5,direction_consistent,deduplicated}"
AUTO_OUT_DIR="${SPLICE_AUTO_OUT_DIR:-outputs/${DATASET}_routing_controls_top${SPLICE_AUTO_TOP_K}_dir${SPLICE_REQUIRE_CONSISTENT_DIRECTION}_dedup${SPLICE_DEDUPLICATE_CONCEPTS}}"
DISCOVERY_JSON="${AUTO_OUT_DIR}/${DATASET}_splice_concepts.json"
CONCEPTS_FILE="${AUTO_OUT_DIR}/${DATASET}_splice_concepts.concepts.txt"
DISCOVERY_LOCK="${AUTO_OUT_DIR}/discovery.lock"
mkdir -p "${AUTO_OUT_DIR}" "${SPLICE_SCORE_CACHE_DIR}"

if ! command -v flock >/dev/null 2>&1; then
    echo "The cluster node does not provide flock; array-safe discovery cannot continue." >&2
    exit 3
fi

exec 9>"${DISCOVERY_LOCK}"
flock 9
if [[ ! -s "${CONCEPTS_FILE}" ]]; then
    python tools/discover_splice_spurious_concepts.py \
        --dataset "${DATASET}" \
        --data_folder "${DATA_FOLDER}" \
        --split train \
        --out_path "${DISCOVERY_JSON}" \
        --top_k "${SPLICE_AUTO_TOP_K}" \
        --per_image_top_k 0 \
        --batch_size "${SPLICE_BATCH_SIZE}" \
        --num_workers "${SPLICE_NUM_WORKERS}" \
        --device cuda \
        --disable_cudnn \
        --splice_model "${SPLICE_MODEL}" \
        --splice_pretrained "${SPLICE_PRETRAINED}" \
        --splice_vocab "${SPLICE_VOCAB}" \
        --splice_vocab_size "${SPLICE_VOCAB_SIZE}" \
        --splice_l1_penalty "${SPLICE_L1_PENALTY}" \
        --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}" \
        --require_consistent_spurious_direction "${SPLICE_REQUIRE_CONSISTENT_DIRECTION}" \
        --deduplicate_concepts "${SPLICE_DEDUPLICATE_CONCEPTS}"
fi
flock -u 9

if [[ ! -s "${CONCEPTS_FILE}" ]]; then
    echo "Automatic discovery did not produce ${CONCEPTS_FILE}" >&2
    exit 4
fi
SPLICE_CONCEPTS="$(tr -d '\r\n' < "${CONCEPTS_FILE}")"

COMMON_ARGS=(
    --use_wandb
    --wandb_name "${WANDB_NAME}"
    --entity "${WANDB_ENTITY}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_tags "${WANDB_TAGS}"
    --dataset "${DATASET}"
    --data_folder "${DATA_FOLDER}"
    --model "${MODEL}"
    --num_workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --temp "${TEMP}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --print_freq "${PRINT_FREQ}"
    --rank_eval_freq "${RANK_EVAL_FREQ}"
    --train_set_linear_layer "${LINEAR_TRAIN_SPLIT}"
    --linear_eval_split "${LINEAR_EVAL_SPLIT}"
    --linear_probe_mode periodic
    --linear_probe_freq "${LINEAR_PROBE_FREQ}"
    --linear_probe_epochs "${LINEAR_PROBE_EPOCHS}"
    --linear_spurious_probe "${LINEAR_SPURIOUS_PROBE}"
)

if [[ "${SPLICE_MODE}" == "none" ]]; then
    python spur_splice.py "${COMMON_ARGS[@]}" --splice_mode none
else
    python spur_splice.py "${COMMON_ARGS[@]}" \
        --splice_mode augment \
        --splice_concepts "${SPLICE_CONCEPTS}" \
        --splice_auto_top_k "${SPLICE_AUTO_TOP_K}" \
        --splice_auto_out_dir "${AUTO_OUT_DIR}" \
        --splice_auto_require_consistent_direction "${SPLICE_REQUIRE_CONSISTENT_DIRECTION}" \
        --splice_auto_deduplicate_concepts "${SPLICE_DEDUPLICATE_CONCEPTS}" \
        --splice_score_reduction "${SPLICE_SCORE_REDUCTION}" \
        --splice_score_threshold auto \
        --splice_score_quantile "${SPLICE_SCORE_QUANTILE}" \
        --splice_routing_mode "${ROUTING_MODE}" \
        --splice_batch_size "${SPLICE_BATCH_SIZE}" \
        --splice_num_workers "${SPLICE_NUM_WORKERS}" \
        --splice_l1_penalty "${SPLICE_L1_PENALTY}" \
        --splice_model "${SPLICE_MODEL}" \
        --splice_pretrained "${SPLICE_PRETRAINED}" \
        --splice_vocab "${SPLICE_VOCAB}" \
        --splice_vocab_size "${SPLICE_VOCAB_SIZE}" \
        --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}" \
        --splice_strong_crop true \
        --splice_strong_color_jitter true \
        --splice_strong_grayscale_p true \
        --splice_strong_blur_p true
fi

echo "Finished ${RUN_LABEL}; seed=${SEED}; concepts=${SPLICE_CONCEPTS}; $(date --iso-8601=seconds)"
conda deactivate
