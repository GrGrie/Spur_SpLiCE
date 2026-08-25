#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --job-name=SpurSpLiCE_train
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./slurm-%x_%j.out
#SBATCH --error=./slurm-%x_%j.err

set -euo pipefail

# =============================================================================
# EDIT ONLY THIS CONFIGURATION BLOCK
# Submit with: sbatch scripts/train.sbatch
#
# MODE selects the experiment:
#   none               plain SimCLR baseline
#   crp_relational     label-free CRP teacher-graph distillation
#   augment            concept-routed strong augmentation
#   corr_reg           SpLiCE/feature correlation regularization
#   augment_corr_reg   routed augmentation + correlation regularization
# =============================================================================

# =============================================================================
# CURRENT METHOD: CRPv2
# Ordered approximately by expected impact on WG/average accuracy.
# Start a sweep from the top; parameters lower in the file usually need fewer changes.
# =============================================================================
MODE="crp_relational"
SPLICE_WEIGHT="0.05"              # First CRP sweep axis: try 0.025 and 0.05.
CRP_TEMPERATURE="0.1"             # Second CRP sweep axis: try 0.1 and 0.2.
LEARNING_RATE="0.01"
EPOCHS="500"
LR_DECAY_EPOCHS="auto"            # At 500 epochs: 350,400,450.
LR_DECAY_RATE="0.1"
CRP_START_EPOCH="10"
CRP_WARMUP_EPOCHS="10"
BATCH_SIZE="256"
SEED="0"
CRP_TEACHER_GRAPH="outputs/crp/waterbirds/teacher_graph_seed0.pt"

# =============================================================================
# LEGACY METHODS: augment / corr_reg / augment_corr_reg
# These settings are ignored by CRPv2. SPLICE_WEIGHT above is also the corr-reg
# weight, so it remains near the top rather than being duplicated here.
# =============================================================================
SPLICE_SCORE_QUANTILE="0.75"      # Main augment sweep axis: 0.50/0.75/0.90/0.95.
SPLICE_ROUTING_MODE="semantic"    # semantic | shuffled | random | all
SPLICE_CONDITIONAL_ON_TARGET="true"
SPLICE_CONCEPTS="auto"            # "auto" or comma-separated names/indices.
SPLICE_AUTO_TOP_K="5"
SPLICE_SCORE_REDUCTION="max"      # mean | max
SPLICE_L1_PENALTY="0.25"

# Strong augmentation components; use "false" to disable one.
SPLICE_STRONG_CROP="0.08"
SPLICE_STRONG_COLOR_JITTER="0.8,0.8,0.8,0.2"
SPLICE_STRONG_COLOR_JITTER_P="0.9"
SPLICE_STRONG_GRAYSCALE_P="0.3"
SPLICE_STRONG_BLUR_P="0.5"
SPLICE_STRONG_BLUR_KERNEL_SIZE="" # Empty = automatic kernel size.
SPLICE_STRONG_BLUR_SIGMA="0.1,2.0"
SPLICE_STRONG_LINE_RECOLOR="false"
SPLICE_SCORE_THRESHOLD="auto"

# Automatic concept discovery (normally leave fixed while sweeping training).
SPLICE_AUTO_RANKING_METHOD="intervention_utility"
SPLICE_AUTO_SPLIT="train"
SPLICE_AUTO_PROBE_C="1.0"
SPLICE_AUTO_PROBE_MAX_ITER="5000"
SPLICE_AUTO_PROBE_CV_FOLDS="5"
SPLICE_AUTO_UTILITY_MAX_SAMPLES="20000"
SPLICE_AUTO_UTILITY_CANDIDATE_POOL="100"
SPLICE_AUTO_UTILITY_MIN_REPAIR="1"
SPLICE_AUTO_UTILITY_MIN_MARGINAL="0.0"
SPLICE_AUTO_MIN_MEAN_WEIGHT="0.0"
SPLICE_AUTO_LABEL_PENALTY="1.0"
SPLICE_AUTO_INSTABILITY_PENALTY="1.0"
SPLICE_AUTO_USE_ABS_SCORE="false"
SPLICE_AUTO_REQUIRE_CONSISTENT_DIRECTION="true"
SPLICE_AUTO_DEDUPLICATE_CONCEPTS="true"

# =============================================================================
# SHARED LOWER-PRIORITY TRAINING SETTINGS
# Change these only after the high-impact CRP/legacy parameters above.
# =============================================================================
OPTIMIZER="SGD"                   # SGD | AdamW | SAM
WEIGHT_DECAY="1e-4"
MOMENTUM="0.9"
SIMCLR_TEMPERATURE="0.05"
SSL_CROP_MIN="0.2"
COSINE="false"
LR_WARMUP="false"
MODEL="resnet18_large"            # spur_cifar10 should use resnet18, not *_large.
HEAD="mlp"
FEAT_DIM="128"
DATASET="waterbirds"              # waterbirds | celeba | spur_cifar10
TRIAL="0"

# SAM-only settings (ignored by SGD/AdamW)
SAM_BASE_OPTIMIZER="SGD"
SAM_RHO="0.05"
SAM_NO_GRAD_NORM="false"
ONLY_SAM_STEP_SIZE="false"

# Frozen SpLiCE/OpenCLIP backend (legacy methods only)
SPLICE_MODEL="open_clip:ViT-B-32"
SPLICE_PRETRAINED="laion2b_s34b_b79k"
SPLICE_VOCAB="laion"
SPLICE_VOCAB_SIZE="10000"
SPLICE_BATCH_SIZE="128"
SPLICE_NUM_WORKERS="1"

# Evaluation and checkpoints
LINEAR_TRAIN_SPLIT="ds_train"
LINEAR_EVAL_SPLIT="val"
LINEAR_PROBE_MODE="periodic"      # periodic | final | none
LINEAR_PROBE_FREQ="25"
LINEAR_PROBE_EPOCHS="100"
LINEAR_LEARNING_RATE="1.0"
LINEAR_LR_DECAY_EPOCHS="auto"
LINEAR_LR_DECAY_RATE="0.2"
LINEAR_WEIGHT_DECAY="0.0"
LINEAR_SPURIOUS_PROBE="true"
FINAL_TEST="false"                # true requires LINEAR_PROBE_MODE=final.
SAVE_FREQ="25"
KEEP_CHECKPOINTS="true"
PRINT_FREQ="25"
RANK_EVAL_FREQ="0"

# Cluster paths and runtime (rarely changed)
PROJECT_DIR="/home/xar68reb/Spur_SpLiCE"
CONDA_ENV="grgrie-train"
DATA_FOLDER="./datasets"
CHECKPOINT_DIR="outputs/training"
RESUME=""                         # Empty = start from scratch; otherwise a .pth path.
SPLICE_AUTO_OUT_DIR="outputs/auto_concepts/${DATASET}_seed${SEED}"
SPLICE_SCORE_CACHE_DIR="outputs/splice_score_cache"
NUM_WORKERS="5"
AMP="true"
CHANNELS_LAST="true"
CUDNN_ENABLED="true"
CUDNN_BENCHMARK="false"           # Must remain false for reproducibility.

# Weights & Biases
USE_WANDB="true"
WANDB_PROJECT="Spur_SpLiCE"
WANDB_ENTITY="gsgrechkin-rptu"
WANDB_EXTRA_TAGS="machine_cluster"

# =============================================================================
# IMPLEMENTATION BELOW. Normal experiment changes should not require edits here.
# =============================================================================

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -d "${PROJECT_DIR}" ]] || fail "PROJECT_DIR does not exist: ${PROJECT_DIR}"

case "${MODE}" in
    none|crp_relational|corr_reg|augment|augment_corr_reg) ;;
    *) fail "Unknown MODE=${MODE}" ;;
esac

if [[ "${DATASET}" == "spur_cifar10" && "${MODEL}" == *_large ]]; then
    fail "spur_cifar10 requires resnet18 or resnet50, not ${MODEL}."
fi
if [[ "${FINAL_TEST}" == "true" && "${LINEAR_PROBE_MODE}" != "final" ]]; then
    fail "FINAL_TEST=true requires LINEAR_PROBE_MODE=final."
fi
if [[ "${MODE}" == "crp_relational" && ! -f "${PROJECT_DIR}/${CRP_TEACHER_GRAPH}" && ! -f "${CRP_TEACHER_GRAPH}" ]]; then
    fail "CRP teacher graph does not exist: ${CRP_TEACHER_GRAPH}"
fi

cd "${PROJECT_DIR}"
mkdir -p ./output "${CHECKPOINT_DIR}" "${SPLICE_AUTO_OUT_DIR}" "${SPLICE_SCORE_CACHE_DIR}"

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

case "${MODE}" in
    none) RUN_LABEL="Baseline" ;;
    crp_relational) RUN_LABEL="CRP_w${SPLICE_WEIGHT}_t${CRP_TEMPERATURE}_s${CRP_START_EPOCH}_wu${CRP_WARMUP_EPOCHS}" ;;
    corr_reg) RUN_LABEL="Corr_w${SPLICE_WEIGHT}" ;;
    augment) RUN_LABEL="Aug_q${SPLICE_SCORE_QUANTILE}_${SPLICE_ROUTING_MODE}" ;;
    augment_corr_reg) RUN_LABEL="AugCorr_q${SPLICE_SCORE_QUANTILE}_w${SPLICE_WEIGHT}" ;;
esac

WANDB_RUN_NAME="${DATASET}_S${SEED}_${RUN_LABEL}_e${EPOCHS}"
WANDB_GROUP="${DATASET}_${MODE}_e${EPOCHS}_seed${SEED}"
WANDB_TAGS="dataset_${DATASET},seed_${SEED},mode_${MODE},protocol_e${EPOCHS},${WANDB_EXTRA_TAGS}"

ARGS=(
    spur_splice.py
    --dataset "${DATASET}"
    --data_folder "${DATA_FOLDER}"
    --model "${MODEL}"
    --seed "${SEED}"
    --trial "${TRIAL}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --optimizer "${OPTIMIZER}"
    --learning_rate "${LEARNING_RATE}"
    --lr_decay_epochs "${LR_DECAY_EPOCHS}"
    --lr_decay_rate "${LR_DECAY_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --momentum "${MOMENTUM}"
    --temp "${SIMCLR_TEMPERATURE}"
    --ssl_crop_min "${SSL_CROP_MIN}"
    --head "${HEAD}"
    --feat_dim "${FEAT_DIM}"
    --amp "${AMP}"
    --channels_last "${CHANNELS_LAST}"
    --cudnn_enabled "${CUDNN_ENABLED}"
    --cudnn_benchmark "${CUDNN_BENCHMARK}"
    --print_freq "${PRINT_FREQ}"
    --rank_eval_freq "${RANK_EVAL_FREQ}"
    --train_set_linear_layer "${LINEAR_TRAIN_SPLIT}"
    --linear_eval_split "${LINEAR_EVAL_SPLIT}"
    --linear_probe_mode "${LINEAR_PROBE_MODE}"
    --linear_probe_freq "${LINEAR_PROBE_FREQ}"
    --linear_probe_epochs "${LINEAR_PROBE_EPOCHS}"
    --linear_learning_rate "${LINEAR_LEARNING_RATE}"
    --linear_lr_decay_epochs "${LINEAR_LR_DECAY_EPOCHS}"
    --linear_lr_decay_rate "${LINEAR_LR_DECAY_RATE}"
    --linear_weight_decay "${LINEAR_WEIGHT_DECAY}"
    --linear_spurious_probe "${LINEAR_SPURIOUS_PROBE}"
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --save_freq "${SAVE_FREQ}"
    --splice_mode "${MODE}"
    --wandb_name "${WANDB_PROJECT}"
    --wandb_run_name "${WANDB_RUN_NAME}"
    --entity "${WANDB_ENTITY}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_tags "${WANDB_TAGS}"
)

[[ "${COSINE}" == "true" ]] && ARGS+=(--cosine)
[[ "${LR_WARMUP}" == "true" ]] && ARGS+=(--warm)
[[ "${FINAL_TEST}" == "true" ]] && ARGS+=(--final_test)
[[ "${KEEP_CHECKPOINTS}" == "true" ]] && ARGS+=(--keep_checkpoints)
[[ "${USE_WANDB}" == "true" ]] && ARGS+=(--use_wandb)
[[ -n "${RESUME}" ]] && ARGS+=(--resume "${RESUME}")

if [[ "${OPTIMIZER}" == "SAM" ]]; then
    ARGS+=(--sam_base_optimizer "${SAM_BASE_OPTIMIZER}" --rho "${SAM_RHO}")
    [[ "${SAM_NO_GRAD_NORM}" == "true" ]] && ARGS+=(--sam_no_grad_norm)
    [[ "${ONLY_SAM_STEP_SIZE}" == "true" ]] && ARGS+=(--only_sam_step_size)
fi

if [[ "${MODE}" == "crp_relational" ]]; then
    ARGS+=(
        --splice_weight "${SPLICE_WEIGHT}"
        --crp_teacher_graph "${CRP_TEACHER_GRAPH}"
        --crp_temperature "${CRP_TEMPERATURE}"
        --crp_start_epoch "${CRP_START_EPOCH}"
        --crp_warmup_epochs "${CRP_WARMUP_EPOCHS}"
    )
elif [[ "${MODE}" != "none" ]]; then
    ARGS+=(
        --splice_concepts "${SPLICE_CONCEPTS}"
        --splice_score_reduction "${SPLICE_SCORE_REDUCTION}"
        --splice_l1_penalty "${SPLICE_L1_PENALTY}"
        --splice_auto_top_k "${SPLICE_AUTO_TOP_K}"
        --splice_auto_ranking_method "${SPLICE_AUTO_RANKING_METHOD}"
        --splice_auto_split "${SPLICE_AUTO_SPLIT}"
        --splice_auto_out_dir "${SPLICE_AUTO_OUT_DIR}"
        --splice_auto_probe_c "${SPLICE_AUTO_PROBE_C}"
        --splice_auto_probe_max_iter "${SPLICE_AUTO_PROBE_MAX_ITER}"
        --splice_auto_probe_cv_folds "${SPLICE_AUTO_PROBE_CV_FOLDS}"
        --splice_auto_utility_max_samples "${SPLICE_AUTO_UTILITY_MAX_SAMPLES}"
        --splice_auto_utility_candidate_pool "${SPLICE_AUTO_UTILITY_CANDIDATE_POOL}"
        --splice_auto_utility_min_repair "${SPLICE_AUTO_UTILITY_MIN_REPAIR}"
        --splice_auto_utility_min_marginal "${SPLICE_AUTO_UTILITY_MIN_MARGINAL}"
        --splice_auto_min_mean_weight "${SPLICE_AUTO_MIN_MEAN_WEIGHT}"
        --splice_auto_label_penalty "${SPLICE_AUTO_LABEL_PENALTY}"
        --splice_auto_instability_penalty "${SPLICE_AUTO_INSTABILITY_PENALTY}"
        --splice_auto_require_consistent_direction "${SPLICE_AUTO_REQUIRE_CONSISTENT_DIRECTION}"
        --splice_auto_deduplicate_concepts "${SPLICE_AUTO_DEDUPLICATE_CONCEPTS}"
        --splice_model "${SPLICE_MODEL}"
        --splice_pretrained "${SPLICE_PRETRAINED}"
        --splice_vocab "${SPLICE_VOCAB}"
        --splice_vocab_size "${SPLICE_VOCAB_SIZE}"
        --splice_batch_size "${SPLICE_BATCH_SIZE}"
        --splice_num_workers "${SPLICE_NUM_WORKERS}"
        --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
    )
    [[ "${SPLICE_AUTO_USE_ABS_SCORE}" == "true" ]] && ARGS+=(--splice_auto_use_abs_score)
fi

if [[ "${MODE}" == "corr_reg" ]]; then
    ARGS+=(
        --splice_weight "${SPLICE_WEIGHT}"
        --splice_conditional_on_target "${SPLICE_CONDITIONAL_ON_TARGET}"
    )
elif [[ "${MODE}" == "augment" || "${MODE}" == "augment_corr_reg" ]]; then
    ARGS+=(
        --splice_score_threshold "${SPLICE_SCORE_THRESHOLD}"
        --splice_score_quantile "${SPLICE_SCORE_QUANTILE}"
        --splice_routing_mode "${SPLICE_ROUTING_MODE}"
        --splice_strong_crop "${SPLICE_STRONG_CROP}"
        --splice_strong_color_jitter "${SPLICE_STRONG_COLOR_JITTER}"
        --splice_strong_color_jitter_p "${SPLICE_STRONG_COLOR_JITTER_P}"
        --splice_strong_grayscale_p "${SPLICE_STRONG_GRAYSCALE_P}"
        --splice_strong_blur_p "${SPLICE_STRONG_BLUR_P}"
        --splice_strong_blur_sigma "${SPLICE_STRONG_BLUR_SIGMA}"
        --splice_strong_line_recolor "${SPLICE_STRONG_LINE_RECOLOR}"
    )
    [[ -n "${SPLICE_STRONG_BLUR_KERNEL_SIZE}" ]] && ARGS+=(--splice_strong_blur_kernel_size "${SPLICE_STRONG_BLUR_KERNEL_SIZE}")
    if [[ "${MODE}" == "augment_corr_reg" ]]; then
        ARGS+=(
            --splice_weight "${SPLICE_WEIGHT}"
            --splice_conditional_on_target "${SPLICE_CONDITIONAL_ON_TARGET}"
        )
    fi
fi

echo "=== Spur SpLiCE unified training ==="
echo "mode=${MODE} dataset=${DATASET} seed=${SEED} epochs=${EPOCHS}"
echo "run=${WANDB_RUN_NAME} checkpoints=${CHECKPOINT_DIR}"
echo "started=$(date --iso-8601=seconds) host=$(hostname)"

python -u "${ARGS[@]}"

echo "finished=$(date --iso-8601=seconds)"
conda deactivate
