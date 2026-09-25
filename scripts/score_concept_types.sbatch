#!/usr/bin/env bash
# Score one concept-group artifact by what its words denote, for the concept_type_gate rule.
# Text encoder only: no images and no annotations, so this is a short CPU job.
#
#   sbatch scripts/score_concept_types.sbatch CONCEPT_GROUPS.json OUTPUT.json

#SBATCH --job-name=cospro-concept-types
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/SLURM/concept-types-%j.out
#SBATCH --error=outputs/SLURM/concept-types-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch scripts/score_concept_types.sbatch CONCEPT_GROUPS.json OUTPUT.json [options]" >&2
  exit 2
fi
GROUPS="$1"; OUTPUT="$2"; shift 2
source scripts/announce_results.sh
announce_results "${OUTPUT}" "pass it to scripts/build_cospro_teacher_graphs.sh with --concept-type-scores"

"${PYTHON_BIN}" -u -m cospro.cli.score_concept_types --concept-groups "${GROUPS}" --output "${OUTPUT}" --device cpu "$@"
