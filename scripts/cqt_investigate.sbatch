#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=CQT_investigate
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
# Slurm opens these files before the script can create directories.
# Keep them in the submission directory so a fresh checkout works.
#SBATCH --output=./slurm-%x_%j.out
#SBATCH --error=./slurm-%x_%j.err

set -euo pipefail

# Graph-only CQT sweep. It does not launch SSL training and therefore does not
# create W&B runs. Submit with: sbatch scripts/cqt_investigate.sbatch

Q_VALUES=(0.01 0.02 0.03 0.04 0.05 0.1 0.15 0.2 0.25 0.3 0.4)
DINOQ_VALUES=(0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50)

DATASET="waterbirds"
SEED="0"
PROJECT_DIR="/home/xar68reb/Spur_SpLiCE"
CONDA_ENV="grgrie-train"
DATA_FOLDER="/home/xar68reb/Datasets"
FORCE_REBUILD_CACHE="false"
FORCE_REBUILD_GRAPHS="false"

CRP_CACHE_BATCH_SIZE="64"
CRP_CACHE_NUM_WORKERS="4"
CRP_DINO_MODEL="vit_small_patch16_dinov3.lvd1689m"
SPLICE_MODEL="open_clip:ViT-B-32"
SPLICE_PRETRAINED="laion2b_s34b_b79k"
SPLICE_VOCAB="laion"
SPLICE_VOCAB_SIZE="10000"
SPLICE_L1_PENALTY="0.25"

# Fixed CQT settings. Only q=min_quotient_efficacy and
# dinoq=dino_damage_quantile vary in this investigation.
CQT_MIN_CONCEPT_FREQUENCY="0.01"
CQT_MAX_CONCEPT_FREQUENCY="0.95"
CQT_TEXT_SIMILARITY_THRESHOLD="0.8"
CQT_COACTIVATION_THRESHOLD="0.35"
CQT_MIN_GROUP_SIZE="1"
CQT_MAX_CANDIDATE_GROUPS="128"
CQT_MAX_FACTORS="8"
CQT_MIN_STATE_SAMPLES="16"
CQT_MIN_EXCLUSIVITY="0.8"
CQT_MIN_CONTEXT_SIMILARITY="0.2"
CQT_MIN_STATE_BALANCED_ACCURACY="0.7"
CQT_TRANSPORT_CANDIDATES="32"
CQT_TRANSPORT_MASS="0.25"
CQT_MIN_TRANSPORT_PAIRS="8"
CQT_MIN_WORD_SIMILARITY="0.2"
CQT_DINO_NEIGHBORS="10"
CQT_MIN_INTERVENTION_GAIN="1e-4"
CQT_MIN_COVERAGE="0.01"
CQT_NULL_TRIALS="4"
CQT_NULL_QUANTILE="0.95"
CQT_GRAPH_TOP_K="10"
CQT_INDEGREE_FACTOR="3.0"
CQT_SIMILARITY_CHUNK_SIZE="512"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

canonical_number() {
    local value="$1"
    while [[ "${value}" == *0 ]]; do
        value="${value%0}"
    done
    if [[ "${value}" == *. ]]; then
        value="${value%.}"
    fi
    printf '%s' "${value}"
}

[[ -d "${PROJECT_DIR}" ]] || fail "PROJECT_DIR does not exist: ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

CRP_CACHE_PATH="outputs/crp/${DATASET}_train_features.pt"
CQT_GRAPH_ROOT="outputs/cqt/${DATASET}"
INVESTIGATION_JSON="${CQT_GRAPH_ROOT}/cqt_investigation_seed${SEED}.json"
Q_CSV="$(IFS=,; echo "${Q_VALUES[*]}")"
DINOQ_CSV="$(IFS=,; echo "${DINOQ_VALUES[*]}")"

validate_feature_cache() {
    python -c 'import sys, torch; from splice.crp import validate_feature_cache; validate_feature_cache(torch.load(sys.argv[1], map_location="cpu", weights_only=True))' \
        "${CRP_CACHE_PATH}" >/dev/null 2>&1
}

graph_matches_config() {
    local graph_path="$1"
    local summary_path="$2"
    local config_json="$3"
    python -c 'import json, pathlib, sys, torch; graph=torch.load(sys.argv[1], map_location="cpu", weights_only=True); summary=json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")); expected=json.loads(sys.argv[3]); valid=graph.get("artifact")=="splice_cqt_v1_teacher_graph" and graph.get("config")==expected and summary.get("artifact")==graph.get("artifact") and summary.get("config")==expected; raise SystemExit(0 if valid else 1)' \
        "${graph_path}" "${summary_path}" "${config_json}" >/dev/null 2>&1
}

aggregate_results() {
    python -u -m scripts.tools.aggregate_cqt_investigation \
        --root "${CQT_GRAPH_ROOT}" \
        --output "${INVESTIGATION_JSON}" \
        --dataset "${DATASET}" \
        --seed "${SEED}" \
        --q-values "${Q_CSV}" \
        --dinoq-values "${DINOQ_CSV}" \
        "$@"
}

aggregate_partial_results() {
    aggregate_results --allow-missing || true
}

trap aggregate_partial_results EXIT

cache_is_valid="false"
if [[ "${FORCE_REBUILD_CACHE}" != "true" && -f "${CRP_CACHE_PATH}" ]]; then
    if validate_feature_cache; then
        cache_is_valid="true"
    else
        echo "[CQT] Existing frozen cache is invalid; rebuilding ${CRP_CACHE_PATH}"
    fi
fi

if [[ "${FORCE_REBUILD_CACHE}" == "true" || "${cache_is_valid}" != "true" ]]; then
    echo "=== Building shared frozen feature cache ==="
    mkdir -p "$(dirname "${CRP_CACHE_PATH}")"
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
        --dino-model "${CRP_DINO_MODEL}"
else
    echo "[CQT] Reusing frozen feature cache: ${CRP_CACHE_PATH}"
fi
validate_feature_cache || fail "Frozen feature cache is invalid: ${CRP_CACHE_PATH}"

expected_count=$(( ${#Q_VALUES[@]} * ${#DINOQ_VALUES[@]} ))
combination=0
for q in "${Q_VALUES[@]}"; do
    for dinoq in "${DINOQ_VALUES[@]}"; do
        combination=$((combination + 1))
        q_path="$(canonical_number "${q}")"
        dinoq_path="$(canonical_number "${dinoq}")"
        variant="q${q_path}_dinoq${dinoq_path}"
        graph_dir="${CQT_GRAPH_ROOT}/${variant}"
        graph_path="${graph_dir}/teacher_graph_seed${SEED}.pt"
        summary_path="${graph_dir}/teacher_graph_seed${SEED}.json"
        config_json="{\"min_concept_frequency\":${CQT_MIN_CONCEPT_FREQUENCY},\"max_concept_frequency\":${CQT_MAX_CONCEPT_FREQUENCY},\"text_similarity_threshold\":${CQT_TEXT_SIMILARITY_THRESHOLD},\"coactivation_threshold\":${CQT_COACTIVATION_THRESHOLD},\"min_group_size\":${CQT_MIN_GROUP_SIZE},\"max_candidate_groups\":${CQT_MAX_CANDIDATE_GROUPS},\"max_factors\":${CQT_MAX_FACTORS},\"min_state_samples\":${CQT_MIN_STATE_SAMPLES},\"min_exclusivity\":${CQT_MIN_EXCLUSIVITY},\"min_context_similarity\":${CQT_MIN_CONTEXT_SIMILARITY},\"min_state_balanced_accuracy\":${CQT_MIN_STATE_BALANCED_ACCURACY},\"min_quotient_efficacy\":${q},\"transport_candidates\":${CQT_TRANSPORT_CANDIDATES},\"transport_mass\":${CQT_TRANSPORT_MASS},\"min_transport_pairs\":${CQT_MIN_TRANSPORT_PAIRS},\"min_word_similarity\":${CQT_MIN_WORD_SIMILARITY},\"dino_neighbors\":${CQT_DINO_NEIGHBORS},\"dino_damage_quantile\":${dinoq},\"min_intervention_gain\":${CQT_MIN_INTERVENTION_GAIN},\"min_coverage\":${CQT_MIN_COVERAGE},\"null_trials\":${CQT_NULL_TRIALS},\"null_quantile\":${CQT_NULL_QUANTILE},\"graph_top_k\":${CQT_GRAPH_TOP_K},\"indegree_factor\":${CQT_INDEGREE_FACTOR},\"similarity_chunk_size\":${CQT_SIMILARITY_CHUNK_SIZE},\"seed\":${SEED}}"

        echo "=== [${combination}/${expected_count}] CQT ${variant} ==="
        if [[ "${FORCE_REBUILD_GRAPHS}" != "true" && -f "${graph_path}" && -f "${summary_path}" ]] && \
            graph_matches_config "${graph_path}" "${summary_path}" "${config_json}"; then
            echo "[CQT] Reusing matching graph: ${graph_path}"
            continue
        fi

        mkdir -p "${graph_dir}"
        python -u -m splice.cqt \
            --cache "${CRP_CACHE_PATH}" \
            --output "${graph_path}" \
            --seed "${SEED}" \
            --config "${config_json}"
    done
done

aggregate_results
trap - EXIT
echo "[CQT] Investigation complete: ${INVESTIGATION_JSON}"
