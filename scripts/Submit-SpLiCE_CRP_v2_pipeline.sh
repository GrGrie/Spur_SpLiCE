#!/bin/bash

set -euo pipefail

# Submit from the repository root with one command:
#   bash scripts/Submit-SpLiCE_CRP_v2_pipeline.sh
# To train even when the label-free report returns NO_GO:
#   bash scripts/Submit-SpLiCE_CRP_v2_pipeline.sh --skip-gate
# Paths and hyperparameters live inside the corresponding .sbatch files.

MODE="${1:-}"
if [[ -n "${MODE}" && "${MODE}" != "--skip-gate" ]]; then
    echo "Usage: $0 [--skip-gate]" >&2
    exit 2
fi

cd "$(dirname "$0")/.."
mkdir -p ./output

CACHE_SUBMISSION="$(sbatch --parsable scripts/SpLiCE_CRP_v2_cache_features.sbatch)"
CACHE_JOB="${CACHE_SUBMISSION%%;*}"
AUDIT_SUBMISSION="$(sbatch --parsable --dependency="afterok:${CACHE_JOB}" scripts/SpLiCE_CRP_v2_frozen_audit.sbatch)"
AUDIT_JOB="${AUDIT_SUBMISSION%%;*}"
if [[ "${MODE}" == "--skip-gate" ]]; then
    REPORT_JOB="skipped"
    TRAIN_DEPENDENCY="${AUDIT_JOB}"
else
    REPORT_SUBMISSION="$(sbatch --parsable --dependency="afterok:${AUDIT_JOB}" scripts/SpLiCE_CRP_v2_report.sbatch)"
    REPORT_JOB="${REPORT_SUBMISSION%%;*}"
    TRAIN_DEPENDENCY="${REPORT_JOB}"
fi
TRAIN_SUBMISSION="$(sbatch --parsable --dependency="afterok:${TRAIN_DEPENDENCY}" scripts/SpLiCE_CRP_v2_training.sbatch)"
TRAIN_JOB="${TRAIN_SUBMISSION%%;*}"

echo "Submitted CRP v2 pipeline:"
echo "  cache job:    ${CACHE_JOB}"
echo "  audit job:    ${AUDIT_JOB}"
echo "  go/no-go job: ${REPORT_JOB}"
echo "  training job: ${TRAIN_JOB}"
if [[ "${MODE}" == "--skip-gate" ]]; then
    echo "UNSAFE/EXPLORATORY: training starts after audit even if the separate go/no-go report would fail."
else
    echo "Training starts only after cache, audit, and the label-free go/no-go gate succeed."
fi
