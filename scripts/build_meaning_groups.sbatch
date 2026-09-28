#!/usr/bin/env bash
# Build meaning concept groups (average linkage on raw CLIP text embeddings) for concept-factor
# training, and report their factors. The GPU builds a missing SpLiCE cache and the text embeddings.
#
#   sbatch scripts/build_meaning_groups.sbatch --dataset metashift --vocab laion [--diagnose]

#SBATCH --job-name=cospro-meaning-groups
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/SLURM/meaning-groups-%j.out
#SBATCH --error=outputs/SLURM/meaning-groups-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh
announce_results "outputs/shared/<dataset>/graphs/concept_groups_meaning/<vocab>_text_<t>_response_<r>/{concept_groups,factor_report}.json" \
  "SpLiCE caches and text embeddings: ${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/"

"${PYTHON_BIN}" -u -m cospro.cli.build_meaning_groups "$@"
