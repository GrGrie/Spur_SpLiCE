#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --job-name=CRPv4_group_screen
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --output=./output/%x_%j.out
#SBATCH --error=./output/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/scripts/crpv4_group_screen.conf}"
source "${PROJECT_DIR}/scripts/load_config.sh"
load_splice_config "${CONFIG_FILE}"

cd "${PROJECT_DIR}"
mkdir -p ./output "${CRPV4_GROUP_SCREEN_OUTPUT_ROOT}"

[[ -f "${CRP_CACHE_PATH}" ]] || {
    echo "CRP cache not found: ${CRP_CACHE_PATH}" >&2
    exit 2
}

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

for record in "${CRPV4_GROUP_SCREEN_VARIANTS[@]}"; do
    IFS='|' read -r NAME MIN_FREQUENCY MAX_FREQUENCY TEXT_SIMILARITY \
        COACTIVATION SPATIAL_VARIANT SPATIAL_FLOOR FREQUENCY_POWER <<< "${record}"
    OUT_DIR="${CRPV4_GROUP_SCREEN_OUTPUT_ROOT}/${DATASET}/${NAME}"
    mkdir -p "${OUT_DIR}"
    ARGS=(
        --dataset "${DATASET}"
        --data-folder "${DATA_FOLDER}"
        --cache "${CRP_CACHE_PATH}"
        --output-json "${OUT_DIR}/group_screen.json"
        --output-html "${OUT_DIR}/group_screen.html"
        --min-concept-frequency "${MIN_FREQUENCY}"
        --max-concept-frequency "${MAX_FREQUENCY}"
        --text-similarity-threshold "${TEXT_SIMILARITY}"
        --coactivation-threshold "${COACTIVATION}"
        --spatial-floor "${SPATIAL_FLOOR}"
        --spatial-frequency-power "${FREQUENCY_POWER}"
        --fidelity-threshold "${CRPV4_GROUP_FIDELITY_THRESHOLD}"
        --target-image-coverage "${CRPV4_GROUP_TARGET_COVERAGE}"
        --coherence-warning-threshold "${CRPV4_GROUP_COHERENCE_WARNING}"
        --mini-audit "${CRPV4_GROUP_MINI_AUDIT}"
        --mini-samples "${CRPV4_GROUP_MINI_SAMPLES}"
        --mini-max-groups "${CRPV4_GROUP_MINI_MAX_GROUPS}"
        --mini-neighbors "${CRPV4_GROUP_MINI_NEIGHBORS}"
        --mini-null-trials "${CRPV4_GROUP_MINI_NULL_TRIALS}"
        --mini-null-quantile "${CRPV4_GROUP_MINI_NULL_QUANTILE}"
        --mini-min-coverage "${CRPV4_GROUP_MINI_MIN_COVERAGE}"
        --mini-min-intervention-gain "${CRPV4_GROUP_MINI_GAIN}"
        --mini-activation-difference-quantile "${CRPV4_GROUP_MINI_ACTIVATION_QUANTILE}"
        --mini-residual-threshold "${CRPV4_GROUP_MINI_RESIDUAL_THRESHOLD}"
        --seed "${SEED}"
        --report-max-groups "${CRPV4_GROUP_REPORT_MAX_GROUPS}"
        --report-images-per-band "${CRPV4_GROUP_REPORT_IMAGES_PER_BAND}"
    )
    if [[ "${SPATIAL_VARIANT}" != "none" ]]; then
        BALANCE_PATH="${CRPV4_SPATIAL_ROOT}/${SPATIAL_VARIANT}/balance.pt"
        [[ -f "${BALANCE_PATH}" ]] || {
            echo "Spatial artifact not found: ${BALANCE_PATH}" >&2
            exit 2
        }
        ARGS+=(--spatial-variant "${SPATIAL_VARIANT}" --spatial-balance-artifact "${BALANCE_PATH}")
    fi
    echo "[INFO] Running ${NAME}: text=${TEXT_SIMILARITY} coactivation=${COACTIVATION} spatial=${SPATIAL_VARIANT}"
    python -u -m splice.crp_group_screen "${ARGS[@]}"
done

echo "[INFO] CRPv4 group screens are ready under ${CRPV4_GROUP_SCREEN_OUTPUT_ROOT}/${DATASET}"
conda deactivate
