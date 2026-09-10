#!/usr/bin/env bash
#SBATCH --job-name=Concept-Groups-Generator
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/SLURM/%x-%j.out
#SBATCH --error=outputs/SLURM/%x-%j.err

set -euo pipefail
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

CACHE_PATH="${SPUR_SPLICE_SCRATCH_ROOT:-/scratch/xar68reb/CoSpRo}/features/Spur_SpLiCE/waterbirds_crp/crp_features.pt"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CACHE_PATH="$1"
  shift
fi

"${PYTHON_BIN}" -u -m scripts.tools.generate_crp_concept_groups --cache "${CACHE_PATH}" "$@"
