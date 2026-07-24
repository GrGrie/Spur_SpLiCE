#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=Waterbirds_SpLiCE_synthesis
#SBATCH --array=0-7
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

# Synthesis--distillation control battery. Every task shares the SimCLR objective,
# architecture, initialization, and epoch budget; only the synthesized CLIP teacher
# target changes.
#   0 baseline (no teacher)          4 shuffled donor (cross-background, unmatched core)
#   1 unedited CLIP teacher          5 same-class random donor
#   2 class-conditional neutralize   6 zero-out
#   3 random non-selected coords     7 oracle core-matched cross-background swap
#
# Pilot first: --array=0-2 decides whether the edited target separates from both the
# baseline and the unedited-teacher control. Only then launch --array=3-7.

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"
DATASET="${DATASET:-waterbirds}"
MODEL="${MODEL:-resnet18_large}"
WANDB_DATASET_LABEL="$([[ "${DATASET}" == "waterbirds" ]] && echo Waterbirds || echo "${DATASET}")"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TEMP="${TEMP:-0.05}"
PRINT_FREQ="${PRINT_FREQ:-25}"
RANK_EVAL_FREQ="${RANK_EVAL_FREQ:-0}"

# Distillation hyperparameters. Sweep these by exporting SPLICE_WEIGHT / SPLICE_INTERVENTION_STRENGTH.
SPLICE_WEIGHT="${SPLICE_WEIGHT:-0.1}"
SPLICE_INTERVENTION_STRENGTH="${SPLICE_INTERVENTION_STRENGTH:-1.0}"

SPLICE_AUTO_TOP_K="${SPLICE_AUTO_TOP_K:-5}"
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
# The spurious probe is the second reported metric for this stage, so it stays on.
LINEAR_SPURIOUS_PROBE="${LINEAR_SPURIOUS_PROBE:-true}"
WANDB_NAME="${WANDB_NAME:-Spur_SpLiCE}"
WANDB_ENTITY="${WANDB_ENTITY:-gsgrechkin-rptu}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) RUN_LABEL="Baseline";      SPLICE_MODE="none";               INTERVENTION="" ;;
    1) RUN_LABEL="OrigCLIP";      SPLICE_MODE="synthesis_distill";  INTERVENTION="original" ;;
    2) RUN_LABEL="SynNeutralize"; SPLICE_MODE="synthesis_distill";  INTERVENTION="class_neutralize" ;;
    3) RUN_LABEL="SynRandCoords"; SPLICE_MODE="synthesis_distill";  INTERVENTION="random_coords" ;;
    4) RUN_LABEL="SynShuffDonor"; SPLICE_MODE="synthesis_distill";  INTERVENTION="shuffled_donor" ;;
    5) RUN_LABEL="SynSameClass";  SPLICE_MODE="synthesis_distill";  INTERVENTION="same_class_random_donor" ;;
    6) RUN_LABEL="SynZeroOut";    SPLICE_MODE="synthesis_distill";  INTERVENTION="zero_out" ;;
    7) RUN_LABEL="SynOracleSwap"; SPLICE_MODE="synthesis_distill";  INTERVENTION="core_matched_swap" ;;
    *) echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Alpha only scales an actual edit, so it is omitted from the unedited-teacher name.
if [[ "${SPLICE_MODE}" == "none" ]]; then
    HYPER_TAG=""
elif [[ "${INTERVENTION}" == "original" ]]; then
    HYPER_TAG="_w${SPLICE_WEIGHT}"
else
    HYPER_TAG="_w${SPLICE_WEIGHT}a${SPLICE_INTERVENTION_STRENGTH}"
fi
WANDB_RUN_NAME="${WANDB_DATASET_LABEL}_S${SEED}_${RUN_LABEL}${HYPER_TAG}_e${EPOCHS}"

WANDB_GROUP="${WANDB_GROUP:-${DATASET}_synthesis_e${EPOCHS}_seed${SEED}}"
WANDB_TAGS="${WANDB_TAGS:-dataset_${DATASET},seed_${SEED},family_synthesis,task_${RUN_LABEL},machine_cluster,protocol_e${EPOCHS},lambda_${SPLICE_WEIGHT},alpha_${SPLICE_INTERVENTION_STRENGTH}}"
AUTO_OUT_DIR="${SPLICE_AUTO_OUT_DIR:-outputs/${DATASET}_synthesis_top${SPLICE_AUTO_TOP_K}_dir${SPLICE_REQUIRE_CONSISTENT_DIRECTION}_dedup${SPLICE_DEDUPLICATE_CONCEPTS}}"
DISCOVERY_JSON="${AUTO_OUT_DIR}/${DATASET}_splice_concepts.json"
CONCEPTS_FILE="${AUTO_OUT_DIR}/${DATASET}_splice_concepts.concepts.txt"
DISCOVERY_LOCK="${AUTO_OUT_DIR}/discovery.lock"
mkdir -p "${AUTO_OUT_DIR}" "${SPLICE_SCORE_CACHE_DIR}"

if ! command -v flock >/dev/null 2>&1; then
    echo "The cluster node does not provide flock; array-safe discovery cannot continue." >&2
    exit 3
fi

# One array task performs discovery; the others wait on the same lock and reuse the file.
exec 9>"${DISCOVERY_LOCK}"
flock 9
if [[ ! -s "${CONCEPTS_FILE}" ]]; then
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
    --wandb_run_name "${WANDB_RUN_NAME}"
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
    --lr_decay_epochs auto
    --weight_decay "${WEIGHT_DECAY}"
    --print_freq "${PRINT_FREQ}"
    --rank_eval_freq "${RANK_EVAL_FREQ}"
    --train_set_linear_layer "${LINEAR_TRAIN_SPLIT}"
    --linear_eval_split "${LINEAR_EVAL_SPLIT}"
    --linear_probe_mode periodic
    --linear_probe_freq "${LINEAR_PROBE_FREQ}"
    --linear_probe_epochs "${LINEAR_PROBE_EPOCHS}"
    --linear_lr_decay_epochs auto
    --linear_spurious_probe "${LINEAR_SPURIOUS_PROBE}"
)

if [[ "${SPLICE_MODE}" == "none" ]]; then
    python spur_splice.py "${COMMON_ARGS[@]}" --splice_mode none
else
    python spur_splice.py "${COMMON_ARGS[@]}" \
        --splice_mode synthesis_distill \
        --splice_intervention "${INTERVENTION}" \
        --splice_intervention_strength "${SPLICE_INTERVENTION_STRENGTH}" \
        --splice_weight "${SPLICE_WEIGHT}" \
        --splice_concepts "${SPLICE_CONCEPTS}" \
        --splice_auto_top_k "${SPLICE_AUTO_TOP_K}" \
        --splice_auto_out_dir "${AUTO_OUT_DIR}" \
        --splice_auto_require_consistent_direction "${SPLICE_REQUIRE_CONSISTENT_DIRECTION}" \
        --splice_auto_deduplicate_concepts "${SPLICE_DEDUPLICATE_CONCEPTS}" \
        --splice_score_reduction "${SPLICE_SCORE_REDUCTION}" \
        --splice_batch_size "${SPLICE_BATCH_SIZE}" \
        --splice_num_workers "${SPLICE_NUM_WORKERS}" \
        --splice_l1_penalty "${SPLICE_L1_PENALTY}" \
        --splice_model "${SPLICE_MODEL}" \
        --splice_pretrained "${SPLICE_PRETRAINED}" \
        --splice_vocab "${SPLICE_VOCAB}" \
        --splice_vocab_size "${SPLICE_VOCAB_SIZE}" \
        --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
fi

echo "Finished ${RUN_LABEL}; seed=${SEED}; epochs=${EPOCHS}; lambda=${SPLICE_WEIGHT}; alpha=${SPLICE_INTERVENTION_STRENGTH}; concepts=${SPLICE_CONCEPTS}; $(date --iso-8601=seconds)"
conda deactivate
