#!/bin/bash
# Legacy filename retained for existing commands; this launcher now emits CRPv3.
#SBATCH --time=12:00:00
#SBATCH --job-name=SpLiCE_CRP_v3_audit
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
DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
CACHE_PATH="${CACHE_PATH:-}"
OUT_DIR="${OUT_DIR:-}"
SEED_BASE="${SEED_BASE:-0}"
COBALT="${COBALT:-false}"
COBALT_CONCEPTS_PATH="${COBALT_CONCEPTS_PATH:-}"

if [[ "${COBALT}" != "true" && "${COBALT}" != "false" ]]; then
    echo "COBALT must be true or false, got: ${COBALT}" >&2
    exit 2
fi
if [[ "${COBALT}" == "true" && "${DATASET}" == "spur_cifar10" ]]; then
    echo "COBALT=true currently supports waterbirds and celeba only." >&2
    exit 2
fi
if [[ -z "${CACHE_PATH}" ]]; then
    CACHE_PATH="outputs/crp/${DATASET}_train_features.pt"
fi
# Concept filtering and grouping. These are the first settings to tune when
# the frozen audit reports that the projection is too weak.
MIN_CONCEPT_FREQUENCY="${MIN_CONCEPT_FREQUENCY:-0.01}"
MAX_CONCEPT_FREQUENCY="${MAX_CONCEPT_FREQUENCY:-0.95}"
TEXT_SIMILARITY_THRESHOLD="${TEXT_SIMILARITY_THRESHOLD:-0.80}"
COACTIVATION_THRESHOLD="${COACTIVATION_THRESHOLD:-0.35}"
MIN_GROUP_SIZE="${MIN_GROUP_SIZE:-1}"
MAX_SELECTED_GROUPS="${MAX_SELECTED_GROUPS:-0}"

# Frozen relation audit and graph construction.
PROJECTED_NEIGHBORS="${PROJECTED_NEIGHBORS:-20}"
ACTIVATION_DIFFERENCE_QUANTILE="${ACTIVATION_DIFFERENCE_QUANTILE:-0.85}"
MIN_INTERVENTION_GAIN="${MIN_INTERVENTION_GAIN:-5e-4}"
MIN_COVERAGE="${MIN_COVERAGE:-0.01}"
GRAPH_TOP_K="${GRAPH_TOP_K:-3}"
MAX_INDEGREE="${MAX_INDEGREE:-10}"
INDEGREE_FACTOR="${INDEGREE_FACTOR:-3.0}"

# Null calibration and numerical settings.
NULL_TRIALS="${NULL_TRIALS:-32}"
NULL_QUANTILE="${NULL_QUANTILE:-0.95}"
SIMILARITY_CHUNK_SIZE="${SIMILARITY_CHUNK_SIZE:-512}"
ORTHOGONAL_TOLERANCE="${ORTHOGONAL_TOLERANCE:-1e-6}"
USE_RESIDUAL_SPLICE_GATE="${USE_RESIDUAL_SPLICE_GATE:-true}"
RESIDUAL_SPLICE_SIMILARITY_THRESHOLD="${RESIDUAL_SPLICE_SIMILARITY_THRESHOLD:-0.25}"
USE_COBALT_CONFIDENCE="${USE_COBALT_CONFIDENCE:-true}"

CRP_CONFIG_JSON="{\"cobalt\":${COBALT},\"min_concept_frequency\":${MIN_CONCEPT_FREQUENCY},\"max_concept_frequency\":${MAX_CONCEPT_FREQUENCY},\"text_similarity_threshold\":${TEXT_SIMILARITY_THRESHOLD},\"coactivation_threshold\":${COACTIVATION_THRESHOLD},\"min_group_size\":${MIN_GROUP_SIZE},\"max_selected_groups\":${MAX_SELECTED_GROUPS},\"projected_neighbors\":${PROJECTED_NEIGHBORS},\"activation_difference_quantile\":${ACTIVATION_DIFFERENCE_QUANTILE},\"min_intervention_gain\":${MIN_INTERVENTION_GAIN},\"min_coverage\":${MIN_COVERAGE},\"graph_top_k\":${GRAPH_TOP_K},\"max_indegree\":${MAX_INDEGREE},\"indegree_factor\":${INDEGREE_FACTOR},\"null_trials\":${NULL_TRIALS},\"null_quantile\":${NULL_QUANTILE},\"similarity_chunk_size\":${SIMILARITY_CHUNK_SIZE},\"orthogonal_tolerance\":${ORTHOGONAL_TOLERANCE},\"use_residual_splice_gate\":${USE_RESIDUAL_SPLICE_GATE},\"residual_splice_similarity_threshold\":${RESIDUAL_SPLICE_SIMILARITY_THRESHOLD},\"use_cobalt_confidence\":${USE_COBALT_CONFIDENCE}}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((SEED_BASE + TASK_ID))
if [[ -z "${COBALT_CONCEPTS_PATH}" ]]; then
    COBALT_CONCEPTS_PATH="output/CoBalT/${DATASET}/seed${SEED}/concepts.pt"
fi

cd "${PROJECT_DIR}"
mkdir -p ./output

if [[ ! -f "${CACHE_PATH}" ]]; then
    echo "CRP feature cache not found: ${CACHE_PATH}" >&2
    echo "Set CACHE_PATH to a version-1 cache described in README.md." >&2
    exit 2
fi
if [[ "${COBALT}" == "true" && ! -f "${COBALT_CONCEPTS_PATH}" ]]; then
    echo "CoBalT concept artifact not found: ${COBALT_CONCEPTS_PATH}" >&2
    echo "Run CoBalT/scripts/prepare_concepts.sbatch first or set COBALT_CONCEPTS_PATH." >&2
    exit 2
fi

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

OUT_ROOT="${OUT_DIR:-outputs/crp/${DATASET}}"
CONFIG_PATH="$(python -m scripts.tools.crp_config_path --config "${CRP_CONFIG_JSON}")"
OUT_DIR="${OUT_ROOT}/${CONFIG_PATH}/seed${SEED}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUT_DIR}/teacher_graph.json}"
mkdir -p "${OUT_DIR}"

ARGS=(
    --cache "${CACHE_PATH}"
    --output "${OUTPUT_PATH}"
    --seed "${SEED}"
    --config "${CRP_CONFIG_JSON}"
    --cobalt "${COBALT}"
)
if [[ "${COBALT}" == "true" ]]; then
    ARGS+=(--cobalt-concepts "${COBALT_CONCEPTS_PATH}")
fi

echo "=== SpLiCE-CRP v3 frozen audit ==="
echo "dataset=${DATASET} seed=${SEED} cache=${CACHE_PATH}"
echo "cobalt=${COBALT} cobalt_concepts=${COBALT_CONCEPTS_PATH}"
echo "output=${OUTPUT_PATH}"
echo "config=${CRP_CONFIG_JSON}"

python -u -m splice.crp "${ARGS[@]}"

if [[ "${DATASET}" == "waterbirds" ]]; then
    REPORT_PATH="$(dirname "${OUTPUT_PATH}")/graph_audit.html"
    python -u -m scripts.tools.render_concept_ablation_examples \
        --cache "${CACHE_PATH}" \
        --graph "${OUTPUT_PATH}" \
        --data-folder "${DATA_FOLDER}" \
        --scope selected \
        --max-interventions 12 \
        --edges-per-group 1 \
        --output "${REPORT_PATH}"
    echo "Graph audit report: ${REPORT_PATH}"
fi

echo "Finished CRP v3 frozen audit; output=${OUTPUT_PATH}; $(date --iso-8601=seconds)"
conda deactivate
