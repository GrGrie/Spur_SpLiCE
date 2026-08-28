#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --job-name=SpurSpLiCE_train
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

# CRP frozen-artifact preparation.  When enabled, this same Slurm job first
# builds the dataset-ordered feature cache and then runs the label-free audit.
# Set CRP_FORCE_REBUILD=true after changing any audit setting below.
CRP_AUTO_BUILD_GRAPH="true"
CRP_FORCE_REBUILD="false"
CRP_CACHE_BATCH_SIZE="64"
CRP_CACHE_NUM_WORKERS="4"
CRP_DINO_MODEL="vit_small_patch16_dinov3.lvd1689m"

# CRP frozen-audit hyperparameters.
CRP_MIN_CONCEPT_FREQUENCY="0.01"
CRP_MAX_CONCEPT_FREQUENCY="0.95"
CRP_TEXT_SIMILARITY_THRESHOLD="0.8"
CRP_COACTIVATION_THRESHOLD="0.35"
CRP_MIN_GROUP_SIZE="1"
CRP_PROJECTED_NEIGHBORS="20"
CRP_DINO_NEIGHBORS="50"
CRP_ACTIVATION_DIFFERENCE_QUANTILE="0.75"
CRP_MIN_INTERVENTION_GAIN="1e-4"
CRP_MIN_COVERAGE="0.01"
CRP_GRAPH_TOP_K="10"
CRP_INDEGREE_FACTOR="3.0"
CRP_NULL_TRIALS="16"
CRP_NULL_QUANTILE="0.95"
CRP_SIMILARITY_CHUNK_SIZE="512"
CRP_ORTHOGONAL_TOLERANCE="1e-6"

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
CHECKPOINT_KEEP_COUNT="4"          # Keep the current checkpoint plus 3 previous ones.
DELETE_CHECKPOINTS_AFTER_TRAINING="true" # W&B remains the durable experiment record.
PRINT_FREQ="25"
RANK_EVAL_FREQ="0"

# Cluster paths and runtime (rarely changed)
PROJECT_DIR="/home/xar68reb/Spur_SpLiCE"
CONDA_ENV="grgrie-train"
DATA_FOLDER="/home/xar68reb/Datasets"
CHECKPOINT_DIR="outputs/training"
RESUME=""                         # Empty = start from scratch; otherwise a .pth path.
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

# The root training CLI accepts a few historical CelebA spellings, while the
# CRP cache CLI intentionally uses the canonical lowercase spelling.
case "${DATASET}" in
    CelebA|celebA) DATASET="celeba" ;;
esac
# Recompute dataset-derived paths after canonicalization.
SPLICE_AUTO_OUT_DIR="outputs/auto_concepts/${DATASET}_seed${SEED}"
CRP_CACHE_PATH="outputs/crp/${DATASET}_train_features.pt"
CRP_GRAPH_DIR="outputs/crp/${DATASET}"
CRP_TEACHER_GRAPH="${CRP_GRAPH_DIR}/teacher_graph_seed${SEED}.pt"

if [[ "${DATASET}" == "spur_cifar10" && "${MODEL}" == *_large ]]; then
    fail "spur_cifar10 requires resnet18 or resnet50, not ${MODEL}."
fi
if [[ "${FINAL_TEST}" == "true" && "${LINEAR_PROBE_MODE}" != "final" ]]; then
    fail "FINAL_TEST=true requires LINEAR_PROBE_MODE=final."
fi
cd "${PROJECT_DIR}"
mkdir -p ./output "${CHECKPOINT_DIR}" "${SPLICE_AUTO_OUT_DIR}" "${SPLICE_SCORE_CACHE_DIR}"

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

validate_crp_feature_cache() {
    python -c 'import sys, torch; from splice.crp import validate_feature_cache; validate_feature_cache(torch.load(sys.argv[1], map_location="cpu", weights_only=True))' \
        "${CRP_CACHE_PATH}" >/dev/null 2>&1
}

prepare_crp_teacher_graph() {
    [[ "${MODE}" == "crp_relational" ]] || return 0

    if [[ "${CRP_AUTO_BUILD_GRAPH}" != "true" ]]; then
        echo "[CRP] Automatic cache/graph preparation disabled."
        return 0
    fi

    mkdir -p "${CRP_GRAPH_DIR}" "$(dirname "${CRP_CACHE_PATH}")"

    if [[ "${CRP_FORCE_REBUILD}" != "true" && -f "${CRP_TEACHER_GRAPH}" ]]; then
        echo "[CRP] Reusing existing teacher graph: ${CRP_TEACHER_GRAPH}"
        echo "[CRP] Set CRP_FORCE_REBUILD=true after changing CRP audit settings."
        return 0
    fi

    cache_is_valid="false"
    if [[ "${CRP_FORCE_REBUILD}" != "true" && -f "${CRP_CACHE_PATH}" ]]; then
        if validate_crp_feature_cache; then
            cache_is_valid="true"
        else
            echo "[CRP] Existing feature cache is unreadable or invalid; rebuilding: ${CRP_CACHE_PATH}"
        fi
    fi

    if [[ "${CRP_FORCE_REBUILD}" == "true" || "${cache_is_valid}" != "true" ]]; then
        echo "=== Building CRP frozen feature cache ==="
        python -u -m scripts.tools.cache_crp_features \
            --dataset "${DATASET}" \
            --data-folder "${DATA_FOLDER}" \
            --output "${CRP_CACHE_PATH}" \
            --batch-size "${CRP_CACHE_BATCH_SIZE}" \
            --num-workers "${CRP_CACHE_NUM_WORKERS}" \
            --splice-model "${SPLICE_MODEL}" \
            --splice-pretrained "${SPLICE_PRETRAINED}" \
            --splice-vocab "${SPLICE_VOCAB}" \
            --splice-vocab-size "${SPLICE_VOCAB_SIZE}" \
            --splice-l1-penalty "${SPLICE_L1_PENALTY}" \
            --dino-model "${CRP_DINO_MODEL}"
    else
        echo "[CRP] Reusing existing feature cache: ${CRP_CACHE_PATH}"
    fi

    if ! validate_crp_feature_cache; then
        fail "CRP feature cache is invalid after preparation: ${CRP_CACHE_PATH}"
    fi

    CRP_CONFIG_JSON="{\"min_concept_frequency\":${CRP_MIN_CONCEPT_FREQUENCY},\"max_concept_frequency\":${CRP_MAX_CONCEPT_FREQUENCY},\"text_similarity_threshold\":${CRP_TEXT_SIMILARITY_THRESHOLD},\"coactivation_threshold\":${CRP_COACTIVATION_THRESHOLD},\"min_group_size\":${CRP_MIN_GROUP_SIZE},\"projected_neighbors\":${CRP_PROJECTED_NEIGHBORS},\"dino_neighbors\":${CRP_DINO_NEIGHBORS},\"activation_difference_quantile\":${CRP_ACTIVATION_DIFFERENCE_QUANTILE},\"min_intervention_gain\":${CRP_MIN_INTERVENTION_GAIN},\"min_coverage\":${CRP_MIN_COVERAGE},\"graph_top_k\":${CRP_GRAPH_TOP_K},\"indegree_factor\":${CRP_INDEGREE_FACTOR},\"null_trials\":${CRP_NULL_TRIALS},\"null_quantile\":${CRP_NULL_QUANTILE},\"similarity_chunk_size\":${CRP_SIMILARITY_CHUNK_SIZE},\"orthogonal_tolerance\":${CRP_ORTHOGONAL_TOLERANCE},\"seed\":${SEED}}"

    echo "=== Building CRP teacher graph ==="
    echo "[CRP] cache=${CRP_CACHE_PATH} graph=${CRP_TEACHER_GRAPH}"
    echo "[CRP] config=${CRP_CONFIG_JSON}"
    python -u -m splice.crp \
        --cache "${CRP_CACHE_PATH}" \
        --output "${CRP_TEACHER_GRAPH}" \
        --seed "${SEED}" \
        --config "${CRP_CONFIG_JSON}"
}

prepare_crp_teacher_graph

if [[ "${MODE}" == "crp_relational" && ! -f "${CRP_TEACHER_GRAPH}" ]]; then
    fail "CRP teacher graph does not exist after preparation: ${CRP_TEACHER_GRAPH}"
fi

case "${MODE}" in
    none) RUN_LABEL="Baseline_lr_${LEARNING_RATE}" ;;
    crp_relational) RUN_LABEL="CRP_splWei_${SPLICE_WEIGHT}_crpTemp_${CRP_TEMPERATURE}_lr_${LEARNING_RATE}" ;;
    corr_reg) RUN_LABEL="Corr_splWei_${SPLICE_WEIGHT}_lr_${LEARNING_RATE}" ;;
    augment) RUN_LABEL="Aug_q_${SPLICE_SCORE_QUANTILE}_route_${SPLICE_ROUTING_MODE}_lr_${LEARNING_RATE}" ;;
    augment_corr_reg) RUN_LABEL="AugCorr_q_${SPLICE_SCORE_QUANTILE}_splWei_${SPLICE_WEIGHT}_route_${SPLICE_ROUTING_MODE}_lr_${LEARNING_RATE}" ;;
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
    --checkpoint_keep_count "${CHECKPOINT_KEEP_COUNT}"
    --delete_checkpoints_after_training "${DELETE_CHECKPOINTS_AFTER_TRAINING}"
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
