#!/usr/bin/env bash
#SBATCH --job-name=crp-groups
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/SLURM/crp-groups-%j.out
#SBATCH --error=outputs/SLURM/crp-groups-%j.err

set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CACHE.pt [grouping options]" >&2
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
shift
"${PYTHON_BIN}" -u -m scripts.tools.generate_crp_concept_groups --cache "${CACHE_PATH}" "$@"
