#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "${SCRIPT_DIR}" == "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="."
fi
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODE="${1:---scan}"
ARCHIVE_LABEL="${2:-pre-unification}"
LOG_DIR="${PROJECT_DIR}/outputs/SLURM"

case "${MODE}" in
    --scan|--archive|--delete-after-verify)
        ;;
    *)
        echo "Usage: bash scripts/submit_legacy_archive.sh [--scan|--archive|--delete-after-verify] [ARCHIVE_LABEL]" >&2
        exit 2
        ;;
esac

if [[ ! "${ARCHIVE_LABEL}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: ARCHIVE_LABEL must contain only letters, numbers, '.', '_' or '-'" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
JOB_ID=$(sbatch \
    --parsable \
    --chdir="${PROJECT_DIR}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},LEGACY_ARCHIVE_MODE=${MODE},LEGACY_ARCHIVE_LABEL=${ARCHIVE_LABEL}" \
    --output="${LOG_DIR}/legacy_archive_%j.out" \
    --error="${LOG_DIR}/legacy_archive_%j.err" \
    "${PROJECT_DIR}/scripts/archive_legacy.sbatch")
JOB_ID="${JOB_ID%%;*}"

echo "legacy_archive_job=${JOB_ID}"
echo "mode=${MODE}"
echo "label=${ARCHIVE_LABEL}"
echo "log=${LOG_DIR}/legacy_archive_${JOB_ID}.out"
echo "error_log=${LOG_DIR}/legacy_archive_${JOB_ID}.err"
