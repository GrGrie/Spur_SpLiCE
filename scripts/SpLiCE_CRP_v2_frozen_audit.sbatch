#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=SpLiCE_CRP_v2_audit
#SBATCH --array=0-0
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=./output/%x_%A_%a.out
#SBATCH --error=./output/%x_%A_%a.err

set -euo pipefail

# Label-free frozen audit only: this job does not read target/group annotations
# and does not start SSL training. Override --array=0-2 to repeat null calibration
# with three paired seeds.
#
# Edit the values in this file and submit the pipeline again. No terminal-side
# exports or JSON editing are required. The Python defaults in CrpAuditConfig
# remain useful for direct Python launches; this Slurm job passes every setting
# explicitly so the effective experiment is visible in the job log and graph.

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
DATASET="${DATASET:-waterbirds}"
CACHE_PATH="${CACHE_PATH:-outputs/crp/${DATASET}_train_features.pt}"
OUT_DIR="${OUT_DIR:-outputs/crp/${DATASET}}"
SEED_BASE="${SEED_BASE:-0}"

# Concept filtering and grouping. These are the first settings to tune when
# the frozen audit reports that the projection is too weak.
MIN_CONCEPT_FREQUENCY="0.01"
MAX_CONCEPT_FREQUENCY="0.95"
TEXT_SIMILARITY_THRESHOLD="0.82"
COACTIVATION_THRESHOLD="0.35"
MIN_GROUP_SIZE="1"

# Frozen relation audit and graph construction.
PROJECTED_NEIGHBORS="20"
DINO_NEIGHBORS="50"
ACTIVATION_DIFFERENCE_QUANTILE="0.75"
MIN_INTERVENTION_GAIN="1e-4"
MIN_COVERAGE="0.01"
GRAPH_TOP_K="10"
INDEGREE_FACTOR="3.0"

# Null calibration and numerical settings.
NULL_TRIALS="16"
NULL_QUANTILE="0.95"
SIMILARITY_CHUNK_SIZE="512"
ORTHOGONAL_TOLERANCE="1e-6"

CRP_CONFIG_JSON="{\"min_concept_frequency\":${MIN_CONCEPT_FREQUENCY},\"max_concept_frequency\":${MAX_CONCEPT_FREQUENCY},\"text_similarity_threshold\":${TEXT_SIMILARITY_THRESHOLD},\"coactivation_threshold\":${COACTIVATION_THRESHOLD},\"min_group_size\":${MIN_GROUP_SIZE},\"projected_neighbors\":${PROJECTED_NEIGHBORS},\"dino_neighbors\":${DINO_NEIGHBORS},\"activation_difference_quantile\":${ACTIVATION_DIFFERENCE_QUANTILE},\"min_intervention_gain\":${MIN_INTERVENTION_GAIN},\"min_coverage\":${MIN_COVERAGE},\"graph_top_k\":${GRAPH_TOP_K},\"indegree_factor\":${INDEGREE_FACTOR},\"null_trials\":${NULL_TRIALS},\"null_quantile\":${NULL_QUANTILE},\"similarity_chunk_size\":${SIMILARITY_CHUNK_SIZE},\"orthogonal_tolerance\":${ORTHOGONAL_TOLERANCE}}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((SEED_BASE + TASK_ID))
OUTPUT_PATH="${OUTPUT_PATH:-${OUT_DIR}/teacher_graph_seed${SEED}.json}"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}" ./output

if [[ ! -f "${CACHE_PATH}" ]]; then
    echo "CRP feature cache not found: ${CACHE_PATH}" >&2
    echo "Set CACHE_PATH to a version-1 cache described in README.md." >&2
    exit 2
fi

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

ARGS=(
    --cache "${CACHE_PATH}"
    --output "${OUTPUT_PATH}"
    --seed "${SEED}"
    --config "${CRP_CONFIG_JSON}"
)

echo "=== SpLiCE-CRP v2 frozen audit ==="
echo "dataset=${DATASET} seed=${SEED} cache=${CACHE_PATH}"
echo "output=${OUTPUT_PATH}"
echo "config=${CRP_CONFIG_JSON}"

python -u -m splice.crp "${ARGS[@]}"

echo "Finished CRP v2 frozen audit; output=${OUTPUT_PATH}; $(date --iso-8601=seconds)"
conda deactivate
