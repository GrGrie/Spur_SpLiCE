#!/bin/bash
# Render one-seed trained-encoder and teacher-graph comparison.
#
# Defaults point to the completed seed-1 paper SimCLR run and the seed-1
# concept-group-search semantic run. Edit the *_ROOT, METHOD_POLICY, or
# explicit *_CKPT variables below when comparing another completed run.

#SBATCH --job-name=groups_07_compare
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/groups-07_compare-%j.out
#SBATCH --error=logs/groups-07_compare-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
SEED="${SEED:-1}"
DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
CACHE_PATH="${CACHE_PATH:-${PROJECT_DIR}/outputs/crp_controls_logistic_v1_cluster_seeds12/waterbirds_train_features.pt}"
BASELINE_ROOT="${BASELINE_ROOT:-${PROJECT_DIR}/outputs/paper_completion_2026-09-08/core/seed${SEED}/simclr}"
METHOD_POLICY="${METHOD_POLICY:-semantic}"
METHOD_ROOT="${METHOD_ROOT:-${PROJECT_DIR}/outputs/concept_group_search_v1/ssl/seed${SEED}/${METHOD_POLICY}}"
BASELINE_CKPT="${BASELINE_CKPT:-}"
METHOD_CKPT="${METHOD_CKPT:-}"
GRAPH_PATH="${GRAPH_PATH:-${PROJECT_DIR}/outputs/concept_group_search_v1/graphs/${METHOD_POLICY}.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/outputs/concept_group_search_v1/visual/seed${SEED}_simclr_vs_cospro.html}"
MODEL="${MODEL:-resnet18_large}"
METHOD_NAME="${METHOD_NAME:-CoSpRo}"

cd "${PROJECT_DIR}"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
source scripts/load_splice_cluster_env.sh

ARGS=(
    --cache "${CACHE_PATH}"
    --data-folder "${DATA_FOLDER}"
    --graph "${GRAPH_PATH}"
    --output "${OUTPUT_PATH}"
    --model "${MODEL}"
    --method-name "${METHOD_NAME}"
    --seed "${SEED}"
    --max-groups 6
    --edges-per-group 2
    --device auto
)
if [[ -n "${BASELINE_CKPT}" ]]; then
    ARGS+=(--baseline-ckpt "${BASELINE_CKPT}")
else
    ARGS+=(--baseline-root "${BASELINE_ROOT}")
fi
if [[ -n "${METHOD_CKPT}" ]]; then
    ARGS+=(--method-ckpt "${METHOD_CKPT}")
else
    ARGS+=(--method-root "${METHOD_ROOT}")
fi

"${PYTHON_BIN}" -u -m scripts.tools.render_trained_comparison "${ARGS[@]}"


echo "Comparison report: ${OUTPUT_PATH}"
