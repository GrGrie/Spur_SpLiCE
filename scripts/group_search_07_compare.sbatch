#!/bin/bash
# Render one self-contained frozen-CLIP similarity sweep over selected concept groups.
# The job selects four diagnostic pairs (A+B, A+C, D+E, D+F) and two triplets,
# then measures the cosine change after removing each CRP group subspace.

#SBATCH --job-name=groups_07_compare
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/groups-07_compare-%j.out
#SBATCH --error=logs/groups-07_compare-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
CACHE_PATH="${CACHE_PATH:-${PROJECT_DIR}/outputs/crp_controls_logistic_v1_cluster_seeds12/waterbirds_train_features.pt}"
METHOD_POLICY="${METHOD_POLICY:-semantic}"
GRAPH_PATH="${GRAPH_PATH:-${PROJECT_DIR}/outputs/concept_group_search_v1/graphs/${METHOD_POLICY}.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/outputs/concept_group_search_v1/visual/${METHOD_POLICY}/group_similarity_sweep.html}"
GROUP_SCOPE="${GROUP_SCOPE:-selected}"

cd "${PROJECT_DIR}"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
source scripts/load_splice_cluster_env.sh

"${PYTHON_BIN}" -u -m scripts.tools.render_group_similarity_sweep \
    --cache "${CACHE_PATH}" \
    --data-folder "${DATA_FOLDER}" \
    --graph "${GRAPH_PATH}" \
    --output "${OUTPUT_PATH}" \
    --scope "${GROUP_SCOPE}"

echo "Concept-group similarity sweep: ${OUTPUT_PATH}"
