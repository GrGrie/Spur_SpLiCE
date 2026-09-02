#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CRP
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
# CRP CONFIGURATION — normally edit only this block.
# Submit with: sbatch scripts/train_crp.sbatch
# =============================================================================
MODE="${MODE:-crp_relational}"    # Fixed by this entry point; CQT uses train_cqt.sbatch.
DATASET="${DATASET:-waterbirds}" # waterbirds | celeba | spur_cifar10
MODEL="${MODEL:-resnet18_large}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}" # sbatch --array=0-2 for three seeds.

# Main SSL/CRP sweep parameters.
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
SPLICE_WEIGHT="${SPLICE_WEIGHT:-0.01}"
CRP_TEMPERATURE="${CRP_TEMPERATURE:-0.25}"
CRP_START_EPOCH="${CRP_START_EPOCH:-10}"
CRP_WARMUP_EPOCHS="${CRP_WARMUP_EPOCHS:-10}"
CRP_DECAY_START_EPOCH="${CRP_DECAY_START_EPOCH:-200}"
CRP_DECAY_END_EPOCH="${CRP_DECAY_END_EPOCH:-350}"
CRP_GRAPH_POSITIVES="${CRP_GRAPH_POSITIVES:-true}"
SIMCLR_WEIGHT="${SIMCLR_WEIGHT:-1.0}"
USE_DINO="${USE_DINO:-true}"
COBALT="${COBALT:-false}"
COBALT_CONCEPTS_PATH="${COBALT_CONCEPTS_PATH:-}"

# CRP frozen-audit hyperparameters.
CRP_MIN_CONCEPT_FREQUENCY="${CRP_MIN_CONCEPT_FREQUENCY:-0.01}"
CRP_MAX_CONCEPT_FREQUENCY="${CRP_MAX_CONCEPT_FREQUENCY:-0.95}"
CRP_TEXT_SIMILARITY_THRESHOLD="${CRP_TEXT_SIMILARITY_THRESHOLD:-0.8}"
CRP_COACTIVATION_THRESHOLD="${CRP_COACTIVATION_THRESHOLD:-0.35}"
CRP_MIN_GROUP_SIZE="${CRP_MIN_GROUP_SIZE:-1}"
CRP_MAX_SELECTED_GROUPS="${CRP_MAX_SELECTED_GROUPS:-0}" # 0 = retain every null-passing group.
CRP_PROJECTED_NEIGHBORS="${CRP_PROJECTED_NEIGHBORS:-20}"
CRP_DINO_NEIGHBORS="${CRP_DINO_NEIGHBORS:-50}"
CRP_ACTIVATION_DIFFERENCE_QUANTILE="${CRP_ACTIVATION_DIFFERENCE_QUANTILE:-0.85}"
CRP_MIN_INTERVENTION_GAIN="${CRP_MIN_INTERVENTION_GAIN:-5e-4}"
CRP_MIN_COVERAGE="${CRP_MIN_COVERAGE:-0.01}"
CRP_GRAPH_TOP_K="${CRP_GRAPH_TOP_K:-5}"
CRP_INDEGREE_FACTOR="${CRP_INDEGREE_FACTOR:-3.0}"
CRP_NULL_TRIALS="${CRP_NULL_TRIALS:-32}"
CRP_NULL_QUANTILE="${CRP_NULL_QUANTILE:-0.95}"
CRP_SIMILARITY_CHUNK_SIZE="${CRP_SIMILARITY_CHUNK_SIZE:-512}"
CRP_ORTHOGONAL_TOLERANCE="${CRP_ORTHOGONAL_TOLERANCE:-1e-6}"
RELATIONAL_GRAPH_VARIANT="${RELATIONAL_GRAPH_VARIANT:-}"

# Graph/cache preparation.
CRP_AUTO_BUILD_GRAPH="${CRP_AUTO_BUILD_GRAPH:-true}"
CRP_FORCE_REBUILD="${CRP_FORCE_REBUILD:-false}"
CRP_CACHE_BATCH_SIZE="${CRP_CACHE_BATCH_SIZE:-64}"
CRP_CACHE_NUM_WORKERS="${CRP_CACHE_NUM_WORKERS:-4}"
CRP_DINO_MODEL="${CRP_DINO_MODEL:-vit_small_patch16_dinov3.lvd1689m}"

# =============================================================================
# SHARED LOWER-PRIORITY TRAINING SETTINGS
# =============================================================================
OPTIMIZER="SGD"                   # SGD | AdamW | SAM
WEIGHT_DECAY="1e-4"
MOMENTUM="0.9"
SIMCLR_TEMPERATURE="0.05"
SSL_CROP_MIN="0.2"
COSINE="false"
LR_WARMUP="false"
HEAD="mlp"
FEAT_DIM="128"
TRIAL="0"
LR_DECAY_EPOCHS="auto"
LR_DECAY_RATE="0.1"

# SAM-only settings (ignored by SGD/AdamW)
SAM_BASE_OPTIMIZER="SGD"
SAM_RHO="0.05"
SAM_NO_GRAD_NORM="false"
ONLY_SAM_STEP_SIZE="false"

# Frozen SpLiCE/OpenCLIP backend used to construct the relational cache.
SPLICE_MODEL="open_clip:ViT-B-32"
SPLICE_PRETRAINED="laion2b_s34b_b79k"
SPLICE_VOCAB="laion"
SPLICE_VOCAB_SIZE="10000"
SPLICE_L1_PENALTY="0.25"
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
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
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
WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:-machine_cluster}"

# Diagnostic jobs may reuse the exact cache/graph preparation path without training.
PREPARE_ONLY="${PREPARE_ONLY:-false}"

# =============================================================================
# IMPLEMENTATION BELOW. Normal experiment changes should not require edits here.
# =============================================================================

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -d "${PROJECT_DIR}" ]] || fail "PROJECT_DIR does not exist: ${PROJECT_DIR}"

case "${MODE}" in
    crp_relational|cqt_relational) ;;
    *) fail "Relational entry points support only crp_relational or cqt_relational; got MODE=${MODE}" ;;
esac
case "${USE_DINO}" in true|false) ;; *) fail "USE_DINO must be true or false." ;; esac
case "${COBALT}" in true|false) ;; *) fail "COBALT must be true or false." ;; esac

# The root training CLI accepts a few historical CelebA spellings, while the
# CRP cache CLI intentionally uses the canonical lowercase spelling.
case "${DATASET}" in
    CelebA|celebA) DATASET="celeba" ;;
esac
# Recompute dataset-derived paths after canonicalization.
SPLICE_AUTO_OUT_DIR="outputs/auto_concepts/${DATASET}_seed${SEED}"
if [[ "${USE_DINO}" == "true" ]]; then
    CRP_CACHE_PATH="outputs/crp/${DATASET}_train_features.pt"
else
    CRP_CACHE_PATH="outputs/crp/${DATASET}_train_features_no_dino.pt"
fi
CRP_GRAPH_DIR="outputs/crp/${DATASET}"
CRP_TEACHER_GRAPH="${CRP_GRAPH_DIR}/teacher_graph_seed${SEED}.json"
CQT_GRAPH_DIR="outputs/cqt/${DATASET}"
CQT_TEACHER_GRAPH="${CQT_GRAPH_DIR}/teacher_graph_seed${SEED}.json"
RELATIONAL_TEACHER_GRAPH="${CRP_TEACHER_GRAPH}"
if [[ "${MODE}" == "cqt_relational" ]]; then
    RELATIONAL_TEACHER_GRAPH="${CQT_TEACHER_GRAPH}"
fi
if [[ -z "${COBALT_CONCEPTS_PATH}" ]]; then
    COBALT_CONCEPTS_PATH="output/CoBalT/${DATASET}/seed${SEED}/concepts.pt"
fi

if [[ "${DATASET}" == "spur_cifar10" && ( "${MODEL}" == *_large || "${MODEL}" == "resnet50_pretrained" ) ]]; then
    fail "spur_cifar10 requires resnet18 or resnet50, not ${MODEL}."
fi
if [[ "${COBALT}" == "true" && "${DATASET}" == "spur_cifar10" ]]; then
    fail "COBALT=true currently supports waterbirds and celeba only."
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
    python -c 'import sys, torch; from splice.crp import validate_feature_cache; validate_feature_cache(torch.load(sys.argv[1], map_location="cpu", weights_only=True), require_dino=sys.argv[2].lower()=="true")' \
        "${CRP_CACHE_PATH}" "${USE_DINO}" >/dev/null 2>&1
}

resolve_relational_graph_config() {
    if [[ "${MODE}" == "crp_relational" ]]; then
        RELATIONAL_LABEL="CRP"
        RELATIONAL_ARTIFACT="splice_crp_v2_teacher_graph"
        RELATIONAL_MODULE="splice.crp"
        RELATIONAL_CONFIG_JSON="{\"min_concept_frequency\":${CRP_MIN_CONCEPT_FREQUENCY},\"max_concept_frequency\":${CRP_MAX_CONCEPT_FREQUENCY},\"text_similarity_threshold\":${CRP_TEXT_SIMILARITY_THRESHOLD},\"coactivation_threshold\":${CRP_COACTIVATION_THRESHOLD},\"min_group_size\":${CRP_MIN_GROUP_SIZE},\"max_selected_groups\":${CRP_MAX_SELECTED_GROUPS},\"projected_neighbors\":${CRP_PROJECTED_NEIGHBORS},\"dino_neighbors\":${CRP_DINO_NEIGHBORS},\"activation_difference_quantile\":${CRP_ACTIVATION_DIFFERENCE_QUANTILE},\"min_intervention_gain\":${CRP_MIN_INTERVENTION_GAIN},\"min_coverage\":${CRP_MIN_COVERAGE},\"graph_top_k\":${CRP_GRAPH_TOP_K},\"indegree_factor\":${CRP_INDEGREE_FACTOR},\"null_trials\":${CRP_NULL_TRIALS},\"null_quantile\":${CRP_NULL_QUANTILE},\"similarity_chunk_size\":${CRP_SIMILARITY_CHUNK_SIZE},\"orthogonal_tolerance\":${CRP_ORTHOGONAL_TOLERANCE},\"seed\":${SEED},\"use_dino\":${USE_DINO},\"cobalt\":${COBALT}}"
    else
        RELATIONAL_LABEL="CQT"
        RELATIONAL_ARTIFACT="splice_cqt_v1_teacher_graph"
        RELATIONAL_MODULE="splice.cqt"
        RELATIONAL_CONFIG_JSON="{\"min_concept_frequency\":${CQT_MIN_CONCEPT_FREQUENCY},\"max_concept_frequency\":${CQT_MAX_CONCEPT_FREQUENCY},\"text_similarity_threshold\":${CQT_TEXT_SIMILARITY_THRESHOLD},\"coactivation_threshold\":${CQT_COACTIVATION_THRESHOLD},\"min_group_size\":${CQT_MIN_GROUP_SIZE},\"max_candidate_groups\":${CQT_MAX_CANDIDATE_GROUPS},\"max_factors\":${CQT_MAX_FACTORS},\"min_state_samples\":${CQT_MIN_STATE_SAMPLES},\"min_exclusivity\":${CQT_MIN_EXCLUSIVITY},\"min_context_similarity\":${CQT_MIN_CONTEXT_SIMILARITY},\"min_state_balanced_accuracy\":${CQT_MIN_STATE_BALANCED_ACCURACY},\"min_quotient_efficacy\":${CQT_MIN_QUOTIENT_EFFICACY},\"transport_candidates\":${CQT_TRANSPORT_CANDIDATES},\"transport_mass\":${CQT_TRANSPORT_MASS},\"min_transport_pairs\":${CQT_MIN_TRANSPORT_PAIRS},\"min_word_similarity\":${CQT_MIN_WORD_SIMILARITY},\"dino_neighbors\":${CQT_DINO_NEIGHBORS},\"dino_damage_quantile\":${CQT_DINO_DAMAGE_QUANTILE},\"min_intervention_gain\":${CQT_MIN_INTERVENTION_GAIN},\"min_coverage\":${CQT_MIN_COVERAGE},\"null_trials\":${CQT_NULL_TRIALS},\"null_quantile\":${CQT_NULL_QUANTILE},\"graph_top_k\":${CQT_GRAPH_TOP_K},\"indegree_factor\":${CQT_INDEGREE_FACTOR},\"similarity_chunk_size\":${CQT_SIMILARITY_CHUNK_SIZE},\"seed\":${SEED},\"use_dino\":${USE_DINO},\"cobalt\":${COBALT}}"
    fi
}

teacher_graph_matches_config() {
    python -c 'import json, sys; graph=json.load(open(sys.argv[1], encoding="utf-8")); expected=json.loads(sys.argv[3]); raise SystemExit(0 if graph.get("artifact")==sys.argv[2] and graph.get("config")==expected else 1)' \
        "${RELATIONAL_TEACHER_GRAPH}" "${RELATIONAL_ARTIFACT}" "${RELATIONAL_CONFIG_JSON}" >/dev/null 2>&1
}

crp_config_path() {
    local args=(--config "${RELATIONAL_CONFIG_JSON}")
    if [[ -n "${RELATIONAL_GRAPH_VARIANT}" ]]; then
        args+=(--variant "${RELATIONAL_GRAPH_VARIANT}")
    fi
    python -m scripts.tools.crp_config_path "${args[@]}"
}

prepare_relational_teacher_graph() {
    [[ "${MODE}" == "crp_relational" || "${MODE}" == "cqt_relational" ]] || return 0

    resolve_relational_graph_config
    if [[ "${MODE}" == "crp_relational" ]]; then
        CRP_GRAPH_DIR="outputs/crp/${DATASET}/$(crp_config_path)"
        CRP_TEACHER_GRAPH="${CRP_GRAPH_DIR}/seed${SEED}/teacher_graph.json"
        RELATIONAL_TEACHER_GRAPH="${CRP_TEACHER_GRAPH}"
    else
        CQT_GRAPH_VARIANT="q${CQT_MIN_QUOTIENT_EFFICACY}_dinoq${CQT_DINO_DAMAGE_QUANTILE}"
        if [[ "${USE_DINO}" != "true" || "${COBALT}" == "true" ]]; then
            CQT_GRAPH_VARIANT="${CQT_GRAPH_VARIANT}_useDino${USE_DINO}_cobalt${COBALT}"
        fi
        if [[ -n "${RELATIONAL_GRAPH_VARIANT}" ]]; then
            CQT_GRAPH_VARIANT="${CQT_GRAPH_VARIANT}_${RELATIONAL_GRAPH_VARIANT}"
        fi
        CQT_GRAPH_DIR="outputs/cqt/${DATASET}/${CQT_GRAPH_VARIANT}"
        CQT_TEACHER_GRAPH="${CQT_GRAPH_DIR}/teacher_graph_seed${SEED}.json"
        RELATIONAL_TEACHER_GRAPH="${CQT_TEACHER_GRAPH}"
    fi

    if [[ "${CRP_AUTO_BUILD_GRAPH}" != "true" ]]; then
        echo "[${MODE}] Automatic cache/graph preparation disabled."
        return 0
    fi

    mkdir -p "$(dirname "${RELATIONAL_TEACHER_GRAPH}")" "$(dirname "${CRP_CACHE_PATH}")"

    if [[ "${CRP_FORCE_REBUILD}" != "true" && "${COBALT}" != "true" && -f "${RELATIONAL_TEACHER_GRAPH}" ]]; then
        if teacher_graph_matches_config; then
            echo "[${RELATIONAL_LABEL}] Reusing matching teacher graph: ${RELATIONAL_TEACHER_GRAPH}"
            return 0
        fi
        echo "[${RELATIONAL_LABEL}] Existing graph config differs; rebuilding graph automatically."
    fi
    if [[ "${COBALT}" == "true" && -f "${RELATIONAL_TEACHER_GRAPH}" ]]; then
        echo "[${RELATIONAL_LABEL}] COBALT=true: rebuilding graph from the current concept artifact."
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
            --dino-model "${CRP_DINO_MODEL}" \
            --use-dino "${USE_DINO}"
    else
        echo "[CRP] Reusing existing feature cache: ${CRP_CACHE_PATH}"
    fi

    if ! validate_crp_feature_cache; then
        fail "CRP feature cache is invalid after preparation: ${CRP_CACHE_PATH}"
    fi

    if [[ "${COBALT}" == "true" && ! -f "${COBALT_CONCEPTS_PATH}" ]]; then
        fail "COBALT=true requires fixed Stage-1 concepts at ${COBALT_CONCEPTS_PATH}. Run CoBalT/scripts/prepare_concepts.sbatch first."
    fi

    echo "=== Building ${RELATIONAL_LABEL} teacher graph ==="
    echo "[${RELATIONAL_LABEL}] cache=${CRP_CACHE_PATH} graph=${RELATIONAL_TEACHER_GRAPH}"
    echo "[${RELATIONAL_LABEL}] config=${RELATIONAL_CONFIG_JSON}"
    GRAPH_ARGS=(
        --cache "${CRP_CACHE_PATH}"
        --output "${RELATIONAL_TEACHER_GRAPH}"
        --seed "${SEED}"
        --config "${RELATIONAL_CONFIG_JSON}"
        --use-dino "${USE_DINO}"
        --cobalt "${COBALT}"
    )
    if [[ "${COBALT}" == "true" ]]; then
        GRAPH_ARGS+=(--cobalt-concepts "${COBALT_CONCEPTS_PATH}")
    fi
    python -u -m "${RELATIONAL_MODULE}" "${GRAPH_ARGS[@]}"
}

prepare_relational_teacher_graph

if [[ "${MODE}" == "cqt_relational" && "${DATASET}" == "waterbirds" && \
      "${CQT_ORACLE_AUDIT}" == "true" && -f "${WATERBIRDS_METADATA_CSV}" ]]; then
    python -u -m scripts.tools.audit_cqt_graph_oracle \
        --graph "${RELATIONAL_TEACHER_GRAPH}" \
        --metadata-csv "${WATERBIRDS_METADATA_CSV}"
fi

if [[ ( "${MODE}" == "crp_relational" || "${MODE}" == "cqt_relational" ) && ! -f "${RELATIONAL_TEACHER_GRAPH}" ]]; then
    fail "Relational teacher graph does not exist after preparation: ${RELATIONAL_TEACHER_GRAPH}"
fi

if [[ "${PREPARE_ONLY}" == "true" ]]; then
    echo "[INFO] Frozen cache/graph preparation complete; PREPARE_ONLY=true."
    return 0 2>/dev/null || exit 0
fi

case "${MODE}" in
    none) RUN_LABEL="Baseline_lr_${LEARNING_RATE}" ;;
    crp_relational) RUN_LABEL="CRP_precision_graph_${RELATIONAL_GRAPH_VARIANT:-default}_useDino_${USE_DINO}_cobalt_${COBALT}_simclrWei_${SIMCLR_WEIGHT}_splWei_${SPLICE_WEIGHT}_crpTemp_${CRP_TEMPERATURE}_graphPos_${CRP_GRAPH_POSITIVES}_decay_${CRP_DECAY_START_EPOCH}-${CRP_DECAY_END_EPOCH}_lr_${LEARNING_RATE}" ;;
    cqt_relational) RUN_LABEL="CQT_${CQT_GRAPH_VARIANT}_useDino_${USE_DINO}_cobalt_${COBALT}_simclrWei_${SIMCLR_WEIGHT}_splWei_${SPLICE_WEIGHT}_crpTemp_${CRP_TEMPERATURE}_graphPos_${CRP_GRAPH_POSITIVES}_decay_${CRP_DECAY_START_EPOCH}-${CRP_DECAY_END_EPOCH}_lr_${LEARNING_RATE}" ;;
    corr_reg) RUN_LABEL="Corr_splWei_${SPLICE_WEIGHT}_lr_${LEARNING_RATE}" ;;
    augment) RUN_LABEL="Aug_q_${SPLICE_SCORE_QUANTILE}_route_${SPLICE_ROUTING_MODE}_lr_${LEARNING_RATE}" ;;
    augment_corr_reg) RUN_LABEL="AugCorr_q_${SPLICE_SCORE_QUANTILE}_splWei_${SPLICE_WEIGHT}_route_${SPLICE_ROUTING_MODE}_lr_${LEARNING_RATE}" ;;
esac

WANDB_RUN_NAME="${DATASET}_S${SEED}_${MODEL}_${RUN_LABEL}_e${EPOCHS}"
WANDB_GROUP="${DATASET}_${MODE}_${MODEL}_e${EPOCHS}_seed${SEED}_useDino${USE_DINO}_cobalt${COBALT}"
WANDB_TAGS="dataset_${DATASET},seed_${SEED},mode_${MODE},model_${MODEL},protocol_e${EPOCHS},graph_${RELATIONAL_GRAPH_VARIANT:-default},use_dino_${USE_DINO},cobalt_${COBALT},${WANDB_EXTRA_TAGS}"

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
    --simclr_weight "${SIMCLR_WEIGHT}"
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

if [[ "${MODE}" == "crp_relational" || "${MODE}" == "cqt_relational" ]]; then
    ARGS+=(
        --splice_weight "${SPLICE_WEIGHT}"
        --crp_teacher_graph "${RELATIONAL_TEACHER_GRAPH}"
        --crp_temperature "${CRP_TEMPERATURE}"
        --crp_start_epoch "${CRP_START_EPOCH}"
        --crp_warmup_epochs "${CRP_WARMUP_EPOCHS}"
        --crp_decay_start_epoch "${CRP_DECAY_START_EPOCH}"
        --crp_decay_end_epoch "${CRP_DECAY_END_EPOCH}"
        --crp_graph_positives "${CRP_GRAPH_POSITIVES}"
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

echo "=== Spur SpLiCE relational training ==="
echo "mode=${MODE} dataset=${DATASET} seed=${SEED} epochs=${EPOCHS} use_dino=${USE_DINO} cobalt=${COBALT}"
echo "run=${WANDB_RUN_NAME} checkpoints=${CHECKPOINT_DIR}"
echo "started=$(date --iso-8601=seconds) host=$(hostname)"

python -u "${ARGS[@]}"

echo "finished=$(date --iso-8601=seconds)"
conda deactivate
