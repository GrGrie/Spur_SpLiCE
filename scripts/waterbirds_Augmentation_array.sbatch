#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=Waterbirds_SpurSplice_aug
#SBATCH --array=0-4
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/%x_%A_%a.out
#SBATCH --error=./output/%x_%A_%a.err

set -euo pipefail

# Submit with: sbatch scripts/waterbirds_Augmentation_array.sbatch
# Override one top-level value without editing the file, for example:
# sbatch --export=ALL,SEED=0 scripts/waterbirds_Augmentation_array.sbatch
# SpurCIFAR10 uses the same array with a dataset-specific model:
# sbatch --job-name=SpurCIFAR10_SpLiCE_aug --export=ALL,DATASET=spur_cifar10,SEED=0 scripts/waterbirds_Augmentation_array.sbatch

# ------------------------- paths and environment -------------------------
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"

# -------------------------- training protocol ----------------------------
DATASET="${DATASET:-waterbirds}"
case "${DATASET}" in
    waterbirds|celeba|CelebA|celebA)
        MODEL="${MODEL:-resnet18_large}"
        WANDB_DATASET_LABEL="$([[ "${DATASET}" == "waterbirds" ]] && echo Waterbirds || echo CelebA)"
        ;;
    spur_cifar10)
        MODEL="${MODEL:-resnet18}"
        WANDB_DATASET_LABEL="SpurCIFAR10"
        ;;
    *)
        echo "Unsupported DATASET=${DATASET}" >&2
        exit 2
        ;;
esac
SEED="${SEED:-2}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TEMP="${TEMP:-0.05}"
PRINT_FREQ="${PRINT_FREQ:-25}"
RANK_EVAL_FREQ="${RANK_EVAL_FREQ:-100}"

# ----------------------------- SpLiCE ------------------------------------
SPLICE_MODE="augment"
SPLICE_AUTO_TOP_K=10
SPLICE_SCORE_REDUCTION="max"
SPLICE_SCORE_QUANTILE=0.75
SPLICE_BATCH_SIZE=128
SPLICE_NUM_WORKERS=1
SPLICE_L1_PENALTY=0.25
SPLICE_MODEL="open_clip:ViT-B-32"
SPLICE_PRETRAINED="laion2b_s34b_b79k"
SPLICE_VOCAB="laion"
SPLICE_VOCAB_SIZE=10000
SPLICE_SCORE_CACHE_DIR="${SPLICE_SCORE_CACHE_DIR:-outputs/splice_score_cache}"
SPLICE_STRONG_LINE_RECOLOR="${SPLICE_STRONG_LINE_RECOLOR:-false}"

# --------------------------- linear probe -------------------------------
# Development array: periodic curves are validation-only.
LINEAR_TRAIN_SPLIT="${LINEAR_TRAIN_SPLIT:-ds_train}"
LINEAR_EVAL_SPLIT="${LINEAR_EVAL_SPLIT:-val}"
LINEAR_PROBE_FREQ="${LINEAR_PROBE_FREQ:-25}"
LINEAR_PROBE_EPOCHS="${LINEAR_PROBE_EPOCHS:-100}"
LINEAR_SPURIOUS_PROBE="${LINEAR_SPURIOUS_PROBE:-true}"

# ------------------------------- W&B -------------------------------------
WANDB_NAME="${WANDB_NAME:-Spur_SpLiCE}"
WANDB_ENTITY="${WANDB_ENTITY:-gsgrechkin-rptu}"

# Each array task changes only the targeted augmentation delta.
case "${SLURM_ARRAY_TASK_ID}" in
    0)
        RUN_LABEL="All"
        SPLICE_STRONG_CROP=true
        SPLICE_STRONG_COLOR_JITTER=true
        SPLICE_STRONG_GRAYSCALE_P=true
        SPLICE_STRONG_BLUR_P=true
        ;;
    1)
        RUN_LABEL="Crop"
        SPLICE_STRONG_CROP=true
        SPLICE_STRONG_COLOR_JITTER=false
        SPLICE_STRONG_GRAYSCALE_P=false
        SPLICE_STRONG_BLUR_P=false
        ;;
    2)
        RUN_LABEL="ColorJitter"
        SPLICE_STRONG_CROP=false
        SPLICE_STRONG_COLOR_JITTER=true
        SPLICE_STRONG_GRAYSCALE_P=false
        SPLICE_STRONG_BLUR_P=false
        ;;
    3)
        RUN_LABEL="Grayscale"
        SPLICE_STRONG_CROP=false
        SPLICE_STRONG_COLOR_JITTER=false
        SPLICE_STRONG_GRAYSCALE_P=true
        SPLICE_STRONG_BLUR_P=false
        ;;
    4)
        RUN_LABEL="Blur"
        SPLICE_STRONG_CROP=false
        SPLICE_STRONG_COLOR_JITTER=false
        SPLICE_STRONG_GRAYSCALE_P=false
        SPLICE_STRONG_BLUR_P=true
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

cd "${PROJECT_DIR}"

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# All array tasks need the same automatically discovered concepts. A lock
# makes exactly one task perform discovery; the others wait and reuse its
# output and selected-vector cache. The Slurm job id gives every submission a
# fresh discovery directory, avoiding stale concept lists from older code.
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
WANDB_GROUP="${WANDB_GROUP:-${DATASET}_augmentation_${ARRAY_JOB_ID}_seed${SEED}}"
WANDB_TAGS="${WANDB_TAGS:-dataset_${DATASET},seed_${SEED},family_augmentation,task_${RUN_LABEL}}"
AUTO_OUT_DIR="outputs/slurm_${ARRAY_JOB_ID}_${DATASET}_auto"
DISCOVERY_JSON="${AUTO_OUT_DIR}/${DATASET}_splice_concepts.json"
CONCEPTS_FILE="${AUTO_OUT_DIR}/${DATASET}_splice_concepts.concepts.txt"
DISCOVERY_LOCK="${AUTO_OUT_DIR}/discovery.lock"

mkdir -p "${AUTO_OUT_DIR}" "${SPLICE_SCORE_CACHE_DIR}"

if ! command -v flock >/dev/null 2>&1; then
    echo "The cluster node does not provide 'flock'; automatic array-safe discovery cannot continue." >&2
    exit 3
fi

exec 9>"${DISCOVERY_LOCK}"
flock 9
if [[ ! -s "${CONCEPTS_FILE}" ]]; then
    echo "[SpLiCE] Discovering the shared automatic top-${SPLICE_AUTO_TOP_K} concept set..."
    python scripts/tools/discover_splice_spurious_concepts.py \
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
        --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
fi
flock -u 9

if [[ ! -s "${CONCEPTS_FILE}" ]]; then
    echo "Automatic concept discovery did not produce ${CONCEPTS_FILE}" >&2
    exit 4
fi
SPLICE_CONCEPTS="$(tr -d '\r\n' < "${CONCEPTS_FILE}")"

echo "Job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Configuration: ${RUN_LABEL}; seed=${SEED}; eval=${LINEAR_EVAL_SPLIT}"
echo "Automatic concepts: ${SPLICE_CONCEPTS}"
echo "Started: $(date --iso-8601=seconds)"

python spur_splice.py \
    --use_wandb \
    --wandb_name "${WANDB_NAME}" \
    --wandb_run_name "${WANDB_DATASET_LABEL}_S${SEED}_${RUN_LABEL}" \
    --entity "${WANDB_ENTITY}" \
    --wandb_group "${WANDB_GROUP}" \
    --wandb_tags "${WANDB_TAGS}" \
    --dataset "${DATASET}" \
    --data_folder "${DATA_FOLDER}" \
    --model "${MODEL}" \
    --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --temp "${TEMP}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --print_freq "${PRINT_FREQ}" \
    --rank_eval_freq "${RANK_EVAL_FREQ}" \
    --splice_mode "${SPLICE_MODE}" \
    --splice_concepts "${SPLICE_CONCEPTS}" \
    --splice_auto_top_k "${SPLICE_AUTO_TOP_K}" \
    --splice_auto_out_dir "${AUTO_OUT_DIR}" \
    --splice_score_reduction "${SPLICE_SCORE_REDUCTION}" \
    --splice_score_threshold auto \
    --splice_score_quantile "${SPLICE_SCORE_QUANTILE}" \
    --splice_batch_size "${SPLICE_BATCH_SIZE}" \
    --splice_num_workers "${SPLICE_NUM_WORKERS}" \
    --splice_l1_penalty "${SPLICE_L1_PENALTY}" \
    --splice_model "${SPLICE_MODEL}" \
    --splice_pretrained "${SPLICE_PRETRAINED}" \
    --splice_vocab "${SPLICE_VOCAB}" \
    --splice_vocab_size "${SPLICE_VOCAB_SIZE}" \
    --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}" \
    --splice_strong_crop "${SPLICE_STRONG_CROP}" \
    --splice_strong_color_jitter "${SPLICE_STRONG_COLOR_JITTER}" \
    --splice_strong_grayscale_p "${SPLICE_STRONG_GRAYSCALE_P}" \
    --splice_strong_blur_p "${SPLICE_STRONG_BLUR_P}" \
    --splice_strong_line_recolor "${SPLICE_STRONG_LINE_RECOLOR}" \
    --train_set_linear_layer "${LINEAR_TRAIN_SPLIT}" \
    --linear_eval_split "${LINEAR_EVAL_SPLIT}" \
    --linear_probe_mode periodic \
    --linear_probe_freq "${LINEAR_PROBE_FREQ}" \
    --linear_probe_epochs "${LINEAR_PROBE_EPOCHS}" \
    --linear_spurious_probe "${LINEAR_SPURIOUS_PROBE}"

echo "Finished ${RUN_LABEL} successfully: $(date --iso-8601=seconds)"
conda deactivate
