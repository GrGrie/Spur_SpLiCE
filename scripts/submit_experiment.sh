#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
MANIFEST="${1:-experiments/manifests/waterbirds_cospro.json}"
LOCKED_TEST=0
if [[ "${2:-}" == "--locked-test" ]]; then
    LOCKED_TEST=1
elif [[ -n "${2:-}" ]]; then
    echo "Usage: $0 [manifest.json] [--locked-test]" >&2
    exit 2
fi
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

TASK_COUNT=$("${PYTHON_BIN}" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(len(m["seeds"])*len(m["arms"]))' "${MANIFEST}")
LAST_TASK=$((TASK_COUNT - 1))
ARRAY_JOB=$(sbatch --parsable --array="0-${LAST_TASK}" --export="ALL,PROJECT_DIR=${PROJECT_DIR},MANIFEST=${MANIFEST},LOCKED_TEST=${LOCKED_TEST}" scripts/run_experiment.sbatch)
ARRAY_JOB="${ARRAY_JOB%%;*}"
COLLECT_JOB=$(sbatch --parsable --dependency="afterany:${ARRAY_JOB}" --export="ALL,PROJECT_DIR=${PROJECT_DIR},MANIFEST=${MANIFEST}" scripts/collect_results.sbatch)

echo "training_job=${ARRAY_JOB}"
echo "collector_job=${COLLECT_JOB%%;*}"
