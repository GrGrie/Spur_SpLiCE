#!/usr/bin/env bash
#SBATCH --job-name=splice-legacy-archive
#SBATCH --partition=informatik-mind
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR must be exported by submit_legacy_archive.sh}"
MODE="${LEGACY_ARCHIVE_MODE:---scan}"
LABEL="${LEGACY_ARCHIVE_LABEL:-pre-unification}"
SOURCE="${PROJECT_DIR}/outputs/shared/legacy"
ARCHIVE="${SPUR_SPLICE_SCRATCH_ROOT:-${SPUR_SPLICE_ARTIFACT_ROOT:-/scratch/xar68reb/CoSpRo}}/legacy_archives/Spur_SpLiCE/${LABEL}.tar.gz"
MANIFEST="${PROJECT_DIR}/outputs/reports/legacy-archive/${LABEL}.json"

cd "${PROJECT_DIR}"
source scripts/load_splice_cluster_env.sh

echo "[legacy-archive] mode=${MODE}"
echo "[legacy-archive] source=${SOURCE}"
echo "[legacy-archive] archive=${ARCHIVE}"
echo "[legacy-archive] manifest=${MANIFEST}"

common_args=(--source "${SOURCE}" --archive "${ARCHIVE}" --manifest "${MANIFEST}")
case "${MODE}" in
    --scan)
        "${PYTHON_BIN}" -u -m scripts.tools.archive_legacy --source "${SOURCE}"
        ;;
    --archive)
        "${PYTHON_BIN}" -u -m scripts.tools.archive_legacy "${common_args[@]}" --apply
        ;;
    --delete-after-verify)
        "${PYTHON_BIN}" -u -m scripts.tools.archive_legacy \
            "${common_args[@]}" --apply --delete-source-after-verify
        ;;
    *)
        echo "ERROR: unsupported legacy archive mode: ${MODE}" >&2
        exit 2
        ;;
esac

echo "[legacy-archive] finished successfully"
