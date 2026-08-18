#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpLiCE_utility_discovery
#SBATCH --array=0-3
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

# Frozen SpLiCE-CBM discovery comparison, with no SSL training.
# Submit this array once per dataset with DATASET=waterbirds, celeba, or
# spur_cifar10. Tasks 0-3 are intervention utility, error contrast, gradient
# probe, and metadata oracle respectively.
#
# The first three methods use target labels only. conditional_group is an
# explicitly privileged upper bound and is never used by the proposed method.

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
PROBE_MAX_ITER="${PROBE_MAX_ITER:-5000}"
PROBE_CV_FOLDS="${PROBE_CV_FOLDS:-5}"
UTILITY_MAX_SAMPLES="${UTILITY_MAX_SAMPLES:-20000}"
UTILITY_CANDIDATE_POOL="${UTILITY_CANDIDATE_POOL:-100}"
UTILITY_MIN_REPAIR="${UTILITY_MIN_REPAIR:-1}"
UTILITY_MIN_MARGINAL="${UTILITY_MIN_MARGINAL:-0.0}"

SPLICE_L1_PENALTY="${SPLICE_L1_PENALTY:-0.25}"
SPLICE_MODEL="${SPLICE_MODEL:-open_clip:ViT-B-32}"
SPLICE_PRETRAINED="${SPLICE_PRETRAINED:-laion2b_s34b_b79k}"
SPLICE_VOCAB="${SPLICE_VOCAB:-laion}"
SPLICE_VOCAB_SIZE="${SPLICE_VOCAB_SIZE:-10000}"
SPLICE_SCORE_CACHE_DIR="${SPLICE_SCORE_CACHE_DIR:-outputs/splice_score_cache}"
FINAL_TEST="${FINAL_TEST:-false}"

METHODS=(intervention_utility error_contrast gradient_probe conditional_group)
LABELS=(utility errcontrast gradprobe oracle)
METHOD_INDEX="${SLURM_ARRAY_TASK_ID}"
RANKING="${METHODS[${METHOD_INDEX}]}"
RUN_LABEL="${LABELS[${METHOD_INDEX}]}"

case "${DATASET}" in
    waterbirds|celeba|spur_cifar10) ;;
    *) echo "Unsupported DATASET=${DATASET}; use waterbirds, celeba, or spur_cifar10." >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

AUTO_OUT_DIR="outputs/${DATASET}_discovery_${RUN_LABEL}_top${TOP_K}_seed${SEED}"
OUT_PATH="outputs/${DATASET}_discovery_${RUN_LABEL}_cbm_seed${SEED}.json"
mkdir -p "${AUTO_OUT_DIR}" "${SPLICE_SCORE_CACHE_DIR}" ./output outputs

ARGS=(
    --dataset "${DATASET}"
    --data_folder "${DATA_FOLDER}"
    --train_split "${TRAIN_SPLIT}"
    --intervention_concepts auto
    --auto_ranking_method "${RANKING}"
    --auto_top_k "${TOP_K}"
    --auto_out_dir "${AUTO_OUT_DIR}"
    --auto_probe_cv_folds "${PROBE_CV_FOLDS}"
    --auto_utility_max_samples "${UTILITY_MAX_SAMPLES}"
    --auto_utility_candidate_pool "${UTILITY_CANDIDATE_POOL}"
    --auto_utility_min_repair "${UTILITY_MIN_REPAIR}"
    --auto_utility_min_marginal "${UTILITY_MIN_MARGINAL}"
    --seed "${SEED}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --probe_c "${PROBE_C}"
    --probe_max_iter "${PROBE_MAX_ITER}"
    --splice_l1_penalty "${SPLICE_L1_PENALTY}"
    --splice_model "${SPLICE_MODEL}"
    --splice_pretrained "${SPLICE_PRETRAINED}"
    --splice_vocab "${SPLICE_VOCAB}"
    --splice_vocab_size "${SPLICE_VOCAB_SIZE}"
    --splice_score_cache_dir "${SPLICE_SCORE_CACHE_DIR}"
    --out_path "${OUT_PATH}"
)

if [[ "${FINAL_TEST}" == "true" ]]; then
    ARGS+=(--final_test)
    EVAL_LABEL=test
else
    EVAL_LABEL=val
fi

echo "=== SpLiCE discovery comparison ==="
echo "dataset=${DATASET} method=${RANKING} train=${TRAIN_SPLIT} eval=${EVAL_LABEL} seed=${SEED}"
if [[ "${RANKING}" == "conditional_group" ]]; then
    echo "NOTE: metadata oracle; not part of the group-free proposed method."
fi

python splice_cbm.py "${ARGS[@]}"

echo "Finished dataset=${DATASET} method=${RUN_LABEL}; results=${OUT_PATH}; $(date --iso-8601=seconds)"
conda deactivate
