#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---scan}"
SOURCE="${2:-${PROJECT_DIR}/outputs}"
LOG_DIR="${PROJECT_DIR}/outputs/SLURM"

case "${MODE}" in
    --scan|--dry-run|--apply)
        ;;
    *)
        echo "Usage: bash scripts/submit_outputs_migration.sh [--scan|--dry-run|--apply] [SOURCE]" >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_DIR}"

echo "Submitting outputs migration"
echo "  mode:    ${MODE}"
echo "  source:  ${SOURCE}"
echo "  scratch: ${SPUR_SPLICE_ARTIFACT_ROOT:-/scratch/xar68reb/CoSpRo}"

sbatch \
    --output="${LOG_DIR}/migrate_outputs_%j.out" \
    --error="${LOG_DIR}/migrate_outputs_%j.err" \
    "${PROJECT_DIR}/scripts/migrate_outputs.sbatch" \
    "${MODE}" \
    "${SOURCE}"
