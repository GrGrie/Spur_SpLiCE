#!/usr/bin/env bash
# Show the concept factors and entangled pairs of one dataset before training with them.
#
#   sbatch scripts/inspect_concept_factors.sbatch --dataset metashift --diagnose [--merge-similarity 0.8 ...]
#
# CPU only: it reads the frozen SpLiCE cache from scratch and the concept groups from outputs/shared.

#SBATCH --job-name=cospro-factors
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/SLURM/factors-%j.out
#SBATCH --error=outputs/SLURM/factors-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh
announce_results "outputs/shared/<dataset>/factors/concept_factors_report.json" "add --diagnose to type the factors against the hidden labels (evaluation only)"

"${PYTHON_BIN}" -u -m cospro.cli.inspect_concept_factors "$@"
