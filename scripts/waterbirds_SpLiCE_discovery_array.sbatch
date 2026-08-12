#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=Waterbirds_SpLiCE_discovery
#SBATCH --array=0-3
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

# Concept-discovery comparison on a frozen CLIP model. No SSL training.
# Each task discovers a concept list with a different signal, then measures the
# same intervention (zero the concept coefficients, re-evaluate worst-group
# accuracy) so the lists are directly comparable.
#
#   0 oracle     conditional_group  uses spurious metadata  -> upper bound
#   1 errcontrast error_contrast    class labels only       -> proposed method
#   2 gradprobe   gradient_probe    class labels only       -> published estimator
#   3 paper       fixed list        human-written           -> SpLiCE appendix reference
#
# Usage:
#   sbatch scripts/waterbirds_SpLiCE_discovery_array.sbatch                       # eval on val
#   sbatch --export=ALL,FINAL_TEST=true scripts/waterbirds_SpLiCE_discovery_array.sbatch
#   sbatch --export=ALL,DATASET=celeba scripts/waterbirds_SpLiCE_discovery_array.sbatch

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"
DATASET="${DATASET:-waterbirds}"

TOP_K="${TOP_K:-5}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROBE_C="${PROBE_C:-1.0}"
PROBE_CV_FOLDS="${PROBE_CV_FOLDS:-5}"
GRADIENT_STEP_SCALE="${GRADIENT_STEP_SCALE:-0.1}"
GRADIENT_SCORE="${GRADIENT_SCORE:-indicator}"

# Fixed reference list from the SpLiCE appendix, resolved in the bundled 10k LAION vocab.
PAPER_CONCEPTS="${PAPER_CONCEPTS:-bamboo,forest,forests,hiking,rainforest}"

SPLICE_L1_PENALTY="${SPLICE_L1_PENALTY:-0.25}"
SPLICE_MODEL="${SPLICE_MODEL:-open_clip:ViT-B-32}"
SPLICE_PRETRAINED="${SPLICE_PRETRAINED:-laion2b_s34b_b79k}"
SPLICE_VOCAB="${SPLICE_VOCAB:-laion}"
SPLICE_VOCAB_SIZE="${SPLICE_VOCAB_SIZE:-10000}"
SPLICE_SCORE_CACHE_DIR="${SPLICE_SCORE_CACHE_DIR:-outputs/splice_score_cache}"

# Development default is validation; export FINAL_TEST=true for the locked test split.
FINAL_TEST="${FINAL_TEST:-false}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) RUN_LABEL="oracle";      RANKING="conditional_group" ;;
    1) RUN_LABEL="errcontrast"; RANKING="error_contrast" ;;
    2) RUN_LABEL="gradprobe";   RANKING="gradient_probe" ;;
    3) RUN_LABEL="paper";       RANKING="fixed" ;;
    *) echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Each method keeps its own discovery directory so the concept lists never
# overwrite one another, while the decomposition cache stays shared.
AUTO_OUT_DIR="outputs/${DATASET}_discovery_${RUN_LABEL}_top${TOP_K}"
OUT_PATH="outputs/${DATASET}_discovery_${RUN_LABEL}_cbm.json"
mkdir -p "${AUTO_OUT_DIR}" "${SPLICE_SCORE_CACHE_DIR}" ./output outputs

ARGS=(
    --dataset "${DATASET}"
    --data_folder "${DATA_FOLDER}"
    --train_split "${TRAIN_SPLIT}"
    --seed "${SEED}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --probe_c "${PROBE_C}"
    --splice_l1_penalty "${SPLICE_L1_PENALTY}"
    --splice_model "${SPLICE_MODEL}"
    --splice_pretrained "${SPLICE_PRETRAINED}"
    --splice_vocab "${SPLICE_VOCAB}"
    --splice_vocab_size "${SPLICE_VOCAB_SIZE}"
    --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
    --out_path "${OUT_PATH}"
)

if [[ "${RANKING}" == "fixed" ]]; then
    ARGS+=(--intervention_concepts "${PAPER_CONCEPTS}")
else
    ARGS+=(
        --intervention_concepts auto
        --auto_ranking_method "${RANKING}"
        --auto_top_k "${TOP_K}"
        --auto_out_dir "${AUTO_OUT_DIR}"
        --auto_probe_cv_folds "${PROBE_CV_FOLDS}"
        --auto_gradient_step_scale "${GRADIENT_STEP_SCALE}"
        --auto_gradient_score "${GRADIENT_SCORE}"
    )
fi

if [[ "${FINAL_TEST}" == "true" ]]; then
    ARGS+=(--final_test)
    EVAL_LABEL="test"
else
    EVAL_LABEL="val"
fi

echo "=== SpLiCE concept-discovery comparison ==="
echo "dataset=${DATASET}  method=${RUN_LABEL} (${RANKING})  top_k=${TOP_K}"
echo "train=${TRAIN_SPLIT}  eval=${EVAL_LABEL}  l1=${SPLICE_L1_PENALTY}  probe_C=${PROBE_C}  seed=${SEED}"
if [[ "${RANKING}" == "conditional_group" ]]; then
    echo "NOTE: this task consumes spurious group metadata and is the oracle upper bound."
fi

python splice_cbm.py "${ARGS[@]}"

echo "Finished ${RUN_LABEL}; eval=${EVAL_LABEL}; results in ${OUT_PATH}; $(date --iso-8601=seconds)"
echo "Compare baseline vs probe_intervention worst_group_accuracy across array tasks 0-3."
conda deactivate
