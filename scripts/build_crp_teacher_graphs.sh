#!/usr/bin/env bash
#SBATCH --job-name=Build-Teacher-Graphs
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/SLURM/crp-graphs-%j.out
#SBATCH --error=outputs/SLURM/crp-graphs-%j.err

set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SPLICE_DATASET_CACHE.pt CONCEPT_GROUPS.json|SWEEP_DIR [audit options]" >&2
  exit 2
fi

# Slurm executes a submitted script from a temporary spool copy, so
# BASH_SOURCE[0] is not the path inside the checkout.  SLURM_SUBMIT_DIR is
# the directory from which sbatch was invoked and remains the repository
# root for the documented invocation.  Keep the BASH_SOURCE fallback for
# direct local execution outside Slurm.
if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

cd "${PROJECT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source "${PROJECT_DIR}/scripts/load_splice_cluster_env.sh"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

SPLICE_DATASET_CACHE_PATH="$1"
CONCEPT_GROUPS_PATH="$2"
shift 2
"${PYTHON_BIN}" -u -m scripts.tools.build_crp_teacher_graphs \
  --splice-dataset-cache "${SPLICE_DATASET_CACHE_PATH}" \
  --concept-groups "${CONCEPT_GROUPS_PATH}" "$@"
