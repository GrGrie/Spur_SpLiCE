#!/usr/bin/env bash
#SBATCH --job-name=crp-graphs
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/SLURM/crp-graphs-%j.out
#SBATCH --error=outputs/SLURM/crp-graphs-%j.err

set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CACHE.pt CONCEPT_GROUPS.json|SWEEP_DIR [audit options]" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  source scripts/load_splice_cluster_env.sh
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CACHE_PATH="$1"
CONCEPT_GROUPS_PATH="$2"
shift 2
"${PYTHON_BIN}" -u -m scripts.tools.build_crp_teacher_graphs \
  --cache "${CACHE_PATH}" --concept-groups "${CONCEPT_GROUPS_PATH}" "$@"
