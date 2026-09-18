#!/usr/bin/env bash
#SBATCH --job-name=CoSpRo-Concept-Groups
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=5
# CelebA's full SpLiCE code matrix is about 13.6 GiB; grouping needs one
# additional concept-by-sample workspace even when it is normalized in place.
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/SLURM/%x-%j.out
#SBATCH --error=outputs/SLURM/%x-%j.err

set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SPLICE_DATASET_CACHE.pt [grouping options]" >&2
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
shift
source scripts/announce_results.sh
announce_results "${SPUR_SPLICE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs}/shared/<dataset>/graphs/concept_groups/<grouping config>/" "concept_groups.json plus concept_groups.html; --output-root overrides the sweep directory"

"${PYTHON_BIN}" -u -m scripts.tools.generate_cospro_concept_groups \
  --splice-dataset-cache "${SPLICE_DATASET_CACHE_PATH}" "$@"
