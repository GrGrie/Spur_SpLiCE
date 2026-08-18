#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --job-name=SpLiCE_utility_training
#SBATCH --array=0-4
#SBATCH --mem=64G
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

# Matched downstream control battery for the intervention-utility method.
# Submit this five-task array once per dataset with DATASET=waterbirds,
# celeba, or spur_cifar10:
#   0 baseline SimCLR
#   1 unedited CLIP distillation
#   2 proposed: utility-discovered concepts + class-conditional neutralization
#   3 utility-discovered concepts + zero-out ablation
#   4 matched random coordinates + class-conditional neutralization
#
# Submit another matched seed with, for example:
#   sbatch --export=ALL,DATASET=waterbirds,SEED=1 scripts/SpLiCE_intervention_utility_training_array.sbatch

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"
DATASET="${DATASET:-waterbirds}"
MODEL="${MODEL:-resnet18_large}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TEMP="${TEMP:-0.05}"
PRINT_FREQ="${PRINT_FREQ:-25}"
RANK_EVAL_FREQ="${RANK_EVAL_FREQ:-0}"

SPLICE_WEIGHT="${SPLICE_WEIGHT:-0.1}"
SPLICE_INTERVENTION_STRENGTH="${SPLICE_INTERVENTION_STRENGTH:-1.0}"
SPLICE_AUTO_TOP_K="${SPLICE_AUTO_TOP_K:-5}"
SPLICE_BATCH_SIZE="${SPLICE_BATCH_SIZE:-128}"
SPLICE_NUM_WORKERS="${SPLICE_NUM_WORKERS:-1}"
SPLICE_L1_PENALTY="${SPLICE_L1_PENALTY:-0.25}"
SPLICE_MODEL="${SPLICE_MODEL:-open_clip:ViT-B-32}"
SPLICE_PRETRAINED="${SPLICE_PRETRAINED:-laion2b_s34b_b79k}"
SPLICE_VOCAB="${SPLICE_VOCAB:-laion}"
SPLICE_VOCAB_SIZE="${SPLICE_VOCAB_SIZE:-10000}"
SPLICE_SCORE_CACHE_DIR="${SPLICE_SCORE_CACHE_DIR:-outputs/splice_score_cache}"
PROBE_C="${PROBE_C:-1.0}"
PROBE_MAX_ITER="${PROBE_MAX_ITER:-5000}"
PROBE_CV_FOLDS="${PROBE_CV_FOLDS:-5}"
UTILITY_MAX_SAMPLES="${UTILITY_MAX_SAMPLES:-20000}"
UTILITY_CANDIDATE_POOL="${UTILITY_CANDIDATE_POOL:-100}"
UTILITY_MIN_REPAIR="${UTILITY_MIN_REPAIR:-1}"
UTILITY_MIN_MARGINAL="${UTILITY_MIN_MARGINAL:-0.0}"

LINEAR_TRAIN_SPLIT="${LINEAR_TRAIN_SPLIT:-ds_train}"
LINEAR_EVAL_SPLIT="${LINEAR_EVAL_SPLIT:-val}"
LINEAR_PROBE_FREQ="${LINEAR_PROBE_FREQ:-25}"
LINEAR_PROBE_EPOCHS="${LINEAR_PROBE_EPOCHS:-100}"
LINEAR_SPURIOUS_PROBE="${LINEAR_SPURIOUS_PROBE:-true}"
WANDB_NAME="${WANDB_NAME:-Spur_SpLiCE}"
WANDB_ENTITY="${WANDB_ENTITY:-gsgrechkin-rptu}"

CONFIG_INDEX="${SLURM_ARRAY_TASK_ID}"

case "${DATASET}" in
    waterbirds|celeba|spur_cifar10) ;;
    *) echo "Unsupported DATASET=${DATASET}; use waterbirds, celeba, or spur_cifar10." >&2; exit 2 ;;
esac

case "${CONFIG_INDEX}" in
    0) RUN_LABEL=Baseline;          SPLICE_MODE=none;              INTERVENTION=none ;;
    1) RUN_LABEL=OrigCLIP;          SPLICE_MODE=synthesis_distill; INTERVENTION=original ;;
    2) RUN_LABEL=UtilityNeutralize; SPLICE_MODE=synthesis_distill; INTERVENTION=class_neutralize ;;
    3) RUN_LABEL=UtilityZeroOut;    SPLICE_MODE=synthesis_distill; INTERVENTION=zero_out ;;
    4) RUN_LABEL=UtilityRandCoords; SPLICE_MODE=synthesis_distill; INTERVENTION=random_coords ;;
    *) echo "Unknown CONFIG_INDEX=${CONFIG_INDEX}" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p ./output outputs "${SPLICE_SCORE_CACHE_DIR}"

AUTO_OUT_DIR="outputs/${DATASET}_intervention_utility_top${SPLICE_AUTO_TOP_K}_seed${SEED}"
DISCOVERY_JSON="${AUTO_OUT_DIR}/${DATASET}_splice_concepts.json"
CONCEPTS_FILE="${AUTO_OUT_DIR}/${DATASET}_splice_concepts.concepts.txt"
DISCOVERY_LOCK="${AUTO_OUT_DIR}/discovery.lock"
mkdir -p "${AUTO_OUT_DIR}"

if [[ "${CONFIG_INDEX}" -ge 2 ]]; then
    if ! command -v flock >/dev/null 2>&1; then
        echo "The cluster node does not provide flock; array-safe discovery cannot continue." >&2
        exit 3
    fi
    exec 9>"${DISCOVERY_LOCK}"
    flock 9
    if [[ ! -s "${CONCEPTS_FILE}" ]]; then
        python scripts/tools/discover_splice_spurious_concepts.py \
            --dataset "${DATASET}" \
            --data_folder "${DATA_FOLDER}" \
            --split train \
            --out_path "${DISCOVERY_JSON}" \
            --ranking_method intervention_utility \
            --top_k "${SPLICE_AUTO_TOP_K}" \
            --probe_c "${PROBE_C}" \
            --probe_max_iter "${PROBE_MAX_ITER}" \
            --probe_cv_folds "${PROBE_CV_FOLDS}" \
            --utility_max_samples "${UTILITY_MAX_SAMPLES}" \
            --utility_candidate_pool "${UTILITY_CANDIDATE_POOL}" \
            --utility_min_repair "${UTILITY_MIN_REPAIR}" \
            --utility_min_marginal "${UTILITY_MIN_MARGINAL}" \
            --seed "${SEED}" \
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
            --deduplicate_concepts true
    fi
    flock -u 9
    if [[ ! -s "${CONCEPTS_FILE}" ]]; then
        echo "Intervention-utility discovery did not produce ${CONCEPTS_FILE}" >&2
        exit 4
    fi
    SPLICE_CONCEPTS="$(tr -d '\r\n' < "${CONCEPTS_FILE}")"
elif [[ "${CONFIG_INDEX}" -eq 1 ]]; then
    # The original teacher performs no sparse edit; one inert coordinate keeps
    # the common synthesis loader valid without running concept discovery.
    SPLICE_CONCEPTS=0
else
    SPLICE_CONCEPTS=""
fi

WANDB_RUN_NAME="${DATASET}_S${SEED}_${RUN_LABEL}_w${SPLICE_WEIGHT}a${SPLICE_INTERVENTION_STRENGTH}_e${EPOCHS}"
WANDB_GROUP="${WANDB_GROUP:-${DATASET}_intervention_utility_e${EPOCHS}_seed${SEED}}"
WANDB_TAGS="${WANDB_TAGS:-dataset_${DATASET},seed_${SEED},family_intervention_utility,task_${RUN_LABEL},machine_cluster,protocol_e${EPOCHS}}"

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

if [[ "${SPLICE_MODE}" == none ]]; then
    python spur_splice.py "${COMMON_ARGS[@]}" --splice_mode none
else
    python spur_splice.py "${COMMON_ARGS[@]}" \
        --splice_mode synthesis_distill \
        --splice_intervention "${INTERVENTION}" \
        --splice_intervention_strength "${SPLICE_INTERVENTION_STRENGTH}" \
        --splice_weight "${SPLICE_WEIGHT}" \
        --splice_concepts "${SPLICE_CONCEPTS}" \
        --splice_auto_ranking_method intervention_utility \
        --splice_score_reduction max \
        --splice_batch_size "${SPLICE_BATCH_SIZE}" \
        --splice_num_workers "${SPLICE_NUM_WORKERS}" \
        --splice_l1_penalty "${SPLICE_L1_PENALTY}" \
        --splice_model "${SPLICE_MODEL}" \
        --splice_pretrained "${SPLICE_PRETRAINED}" \
        --splice_vocab "${SPLICE_VOCAB}" \
        --splice_vocab_size "${SPLICE_VOCAB_SIZE}" \
        --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
fi

echo "Finished dataset=${DATASET} task=${RUN_LABEL} seed=${SEED} concepts=${SPLICE_CONCEPTS}; $(date --iso-8601=seconds)"
conda deactivate
