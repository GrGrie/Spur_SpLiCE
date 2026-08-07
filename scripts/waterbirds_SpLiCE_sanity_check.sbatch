#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=Waterbirds_SpLiCE_sanity
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/%x_%j.out
#SBATCH --error=./output/%x_%j.err

set -euo pipefail

# Frozen-CLIP SpLiCE concept-ablation sanity check (no SSL training).
# Reproduces the SpLiCE appendix Waterbirds intervention: an L1 logistic probe
# on frozen SpLiCE codes, then zero the background-concept coefficients and
# re-measure worst-group accuracy (paper: ~0.48 -> ~0.60 on test).
#
# The concept list is the paper's fixed Waterbirds set resolved in the bundled
# 10k LAION vocab. It is deliberately NOT the automatic-discovery list, which
# omits 'forest'/'hiking' and adds the noise concepts 'whale'/'raven'.
#
# Usage:
#   sbatch scripts/waterbirds_SpLiCE_sanity_check.sbatch                 # eval on val
#   sbatch --export=ALL,FINAL_TEST=true scripts/waterbirds_SpLiCE_sanity_check.sbatch   # eval on test (paper number)

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATA_FOLDER="${DATA_FOLDER:-./datasets}"
DATASET="${DATASET:-waterbirds}"

# Paper-faithful Waterbirds background concepts (resolved in the 10k LAION vocab).
INTERVENTION_CONCEPTS="${INTERVENTION_CONCEPTS:-bamboo,forest,forests,hiking,rainforest}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROBE_C="${PROBE_C:-1.0}"

SPLICE_L1_PENALTY="${SPLICE_L1_PENALTY:-0.25}"
SPLICE_MODEL="${SPLICE_MODEL:-open_clip:ViT-B-32}"
SPLICE_PRETRAINED="${SPLICE_PRETRAINED:-laion2b_s34b_b79k}"
SPLICE_VOCAB="${SPLICE_VOCAB:-laion}"
SPLICE_VOCAB_SIZE="${SPLICE_VOCAB_SIZE:-10000}"
SPLICE_SCORE_CACHE_DIR="${SPLICE_SCORE_CACHE_DIR:-outputs/splice_score_cache}"
OUT_PATH="${OUT_PATH:-outputs/${DATASET}_splice_cbm_sanity.json}"

# Development default is the validation split; export FINAL_TEST=true for the
# test-split number reported by the paper.
FINAL_TEST="${FINAL_TEST:-false}"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${OUT_PATH}")" "${SPLICE_SCORE_CACHE_DIR}" ./output

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

ARGS=(
    --dataset "${DATASET}"
    --data_folder "${DATA_FOLDER}"
    --train_split "${TRAIN_SPLIT}"
    --intervention_concepts "${INTERVENTION_CONCEPTS}"
    --splice_l1_penalty "${SPLICE_L1_PENALTY}"
    --probe_c "${PROBE_C}"
    --seed "${SEED}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --splice_model "${SPLICE_MODEL}"
    --splice_pretrained "${SPLICE_PRETRAINED}"
    --splice_vocab "${SPLICE_VOCAB}"
    --splice_vocab_size "${SPLICE_VOCAB_SIZE}"
    --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
    --out_path "${OUT_PATH}"
)
if [[ "${FINAL_TEST}" == "true" ]]; then
    ARGS+=(--final_test)
    EVAL_LABEL="test"
else
    EVAL_LABEL="val"
fi

echo "=== SpLiCE frozen-CLIP sanity check ==="
echo "dataset=${DATASET}  concepts=${INTERVENTION_CONCEPTS}"
echo "train=${TRAIN_SPLIT}  eval=${EVAL_LABEL}  l1=${SPLICE_L1_PENALTY}  probe_C=${PROBE_C}  seed=${SEED}"

python splice_cbm.py "${ARGS[@]}"

echo "Finished sanity check; eval=${EVAL_LABEL}; results in ${OUT_PATH}; $(date --iso-8601=seconds)"
echo "Paper reference (Waterbirds, test): worst-group 48% -> 60%."
conda deactivate
