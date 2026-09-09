#!/usr/bin/env bash
#SBATCH --job-name=splice-migrate
#SBATCH --partition=informatik-mind
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR must be exported by submit_outputs_migration.sh}"
MODE="${1:---scan}"
SOURCE="${2:-${PROJECT_DIR}/outputs}"

cd "${PROJECT_DIR}"
. "${PROJECT_DIR}/scripts/load_splice_cluster_env.sh"

run_migration() {
    local migration_mode="$1"
    shift
    "${PYTHON_BIN}" -m scripts.tools.migrate_outputs "${SOURCE}" --mode "${migration_mode}" "$@"
}

echo "[migration] project=${PROJECT_DIR}"
echo "[migration] source=${SOURCE}"
echo "[migration] scratch=${SPUR_SPLICE_ARTIFACT_ROOT}"

case "${MODE}" in
    --scan)
        run_migration scan
        ;;
    --dry-run)
        run_migration dry-run
        ;;
    --apply)
        echo "[migration] planned moves (no writes yet):"
        run_migration dry-run
        echo "[migration] applying verified moves:"
        run_migration apply --delete-source-after-verify
        ;;
    *)
        echo "ERROR: unsupported mode: ${MODE}" >&2
        exit 2
        ;;
esac

echo "[migration] finished successfully"
