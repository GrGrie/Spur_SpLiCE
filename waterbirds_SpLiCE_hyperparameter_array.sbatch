#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --job-name=Waterbirds_SpLiCE_sweep
#SBATCH --array=0-8
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

# Validation-only model-selection sweep:
#   task 0:    SimCLR baseline
#   tasks 1-4: automatic SpLiCE augmentation quantiles
#   tasks 5-8: conditional vector-correlation weights
# Submit with: sbatch waterbirds_SpLiCE_hyperparameter_array.sbatch

# ------------------------- paths and environment -------------------------
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"

# -------------------------- training protocol ----------------------------
DATASET="waterbirds"
MODEL="resnet18_large"
SEED="${SEED:-2}"
EPOCHS=1000
BATCH_SIZE=256
NUM_WORKERS=5
LEARNING_RATE=0.01
WEIGHT_DECAY="1e-4"
TEMP=0.05
PRINT_FREQ=25

# Rank evaluation does not affect training and is disabled for this sweep to
# save time. Re-enable it in the selected final runs if those graphs are used.
RANK_EVAL_FREQ=0

# ----------------------------- SpLiCE ------------------------------------
SPLICE_AUTO_TOP_K=10
SPLICE_SCORE_REDUCTION="max"
SPLICE_BATCH_SIZE=128
SPLICE_NUM_WORKERS=1
SPLICE_L1_PENALTY=0.25
SPLICE_MODEL="open_clip:ViT-B-32"
SPLICE_PRETRAINED="laion2b_s34b_b79k"
SPLICE_VOCAB="laion"
SPLICE_VOCAB_SIZE=10000
SPLICE_SCORE_CACHE_DIR="outputs/splice_score_cache"

# --------------------------- linear probe -------------------------------
LINEAR_TRAIN_SPLIT="ds_train"
LINEAR_EVAL_SPLIT="val"
LINEAR_PROBE_FREQ=25
LINEAR_PROBE_EPOCHS=100

# The leakage probe approximately doubles probe time and is not needed to
# choose q/lambda. Enable it again for the selected three-seed final runs.
LINEAR_SPURIOUS_PROBE=false

# ------------------------------- W&B -------------------------------------
WANDB_NAME="Spur_SpLiCE"
WANDB_ENTITY="gsgrechkin-rptu"

# Quantiles are preferable to old absolute thresholds because they preserve a
# stable routed fraction when the automatically selected concepts change.
case "${SLURM_ARRAY_TASK_ID}" in
    0)
        RUN_LABEL="baseline"
        SPLICE_MODE="none"
        ;;
    1)
        RUN_LABEL="augment_q50"
        SPLICE_MODE="augment"
        SPLICE_SCORE_QUANTILE=0.50
        ;;
    2)
        RUN_LABEL="augment_q75"
        SPLICE_MODE="augment"
        SPLICE_SCORE_QUANTILE=0.75
        ;;
    3)
        RUN_LABEL="augment_q90"
        SPLICE_MODE="augment"
        SPLICE_SCORE_QUANTILE=0.90
        ;;
    4)
        RUN_LABEL="augment_q95"
        SPLICE_MODE="augment"
        SPLICE_SCORE_QUANTILE=0.95
        ;;
    5)
        RUN_LABEL="corr_w0.001"
        SPLICE_MODE="corr_reg"
        SPLICE_WEIGHT=0.001
        ;;
    6)
        RUN_LABEL="corr_w0.01"
        SPLICE_MODE="corr_reg"
        SPLICE_WEIGHT=0.01
        ;;
    7)
        RUN_LABEL="corr_w0.1"
        SPLICE_MODE="corr_reg"
        SPLICE_WEIGHT=0.1
        ;;
    8)
        RUN_LABEL="corr_w1.0"
        SPLICE_MODE="corr_reg"
        SPLICE_WEIGHT=1.0
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

# Discover one shared top-10 for the complete array. Other tasks wait on the
# lock and reuse the selected-vector cache instead of repeating SpLiCE.
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
AUTO_OUT_DIR="outputs/slurm_${ARRAY_JOB_ID}_${DATASET}_sweep_auto"
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
        --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
fi
flock -u 9

if [[ ! -s "${CONCEPTS_FILE}" ]]; then
    echo "Automatic concept discovery did not produce ${CONCEPTS_FILE}" >&2
    exit 4
fi
SPLICE_CONCEPTS="$(tr -d '\r\n' < "${CONCEPTS_FILE}")"

COMMON_ARGS=(
    --use_wandb
    --wandb_name "${WANDB_NAME}"
    --entity "${WANDB_ENTITY}"
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

SPLICE_COMMON_ARGS=(
    --splice_concepts "${SPLICE_CONCEPTS}"
    --splice_auto_top_k "${SPLICE_AUTO_TOP_K}"
    --splice_auto_out_dir "${AUTO_OUT_DIR}"
    --splice_score_reduction "${SPLICE_SCORE_REDUCTION}"
    --splice_batch_size "${SPLICE_BATCH_SIZE}"
    --splice_num_workers "${SPLICE_NUM_WORKERS}"
    --splice_l1_penalty "${SPLICE_L1_PENALTY}"
    --splice_model "${SPLICE_MODEL}"
    --splice_pretrained "${SPLICE_PRETRAINED}"
    --splice_vocab "${SPLICE_VOCAB}"
    --splice_vocab_size "${SPLICE_VOCAB_SIZE}"
    --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
)

EXPERIMENT_ARGS=(--splice_mode "${SPLICE_MODE}")
if [[ "${SPLICE_MODE}" == "augment" ]]; then
    EXPERIMENT_ARGS+=(
        --splice_score_threshold auto
        --splice_score_quantile "${SPLICE_SCORE_QUANTILE}"
        --splice_strong_crop true
        --splice_strong_color_jitter true
        --splice_strong_grayscale_p true
        --splice_strong_blur_p true
    )
elif [[ "${SPLICE_MODE}" == "corr_reg" ]]; then
    EXPERIMENT_ARGS+=(
        --splice_weight "${SPLICE_WEIGHT}"
        --splice_conditional_on_target true
    )
fi

echo "Job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Configuration: ${RUN_LABEL}; seed=${SEED}; eval=${LINEAR_EVAL_SPLIT}"
echo "Automatic concepts: ${SPLICE_CONCEPTS}"
echo "Started: $(date --iso-8601=seconds)"

if [[ "${SPLICE_MODE}" == "none" ]]; then
    python spur_splice.py "${COMMON_ARGS[@]}" "${EXPERIMENT_ARGS[@]}"
else
    python spur_splice.py "${COMMON_ARGS[@]}" "${SPLICE_COMMON_ARGS[@]}" "${EXPERIMENT_ARGS[@]}"
fi

echo "Finished ${RUN_LABEL} successfully: $(date --iso-8601=seconds)"
conda deactivate
