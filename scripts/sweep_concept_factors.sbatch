#!/usr/bin/env bash
# Sweep concept grouping thresholds and factor merging for MetaShift and Spur-CIFAR10 under the LAION
# and Open Images dictionaries, scoring every factor set against the hidden labels (evaluation only).
#
#   sbatch scripts/sweep_concept_factors.sbatch [--datasets metashift] [--text-thresholds 0.70 0.80 ...]
#
# The GPU builds the SpLiCE caches that do not exist yet; grouping and scoring run on the CPU.

#SBATCH --job-name=cospro-factor-sweep
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/SLURM/factor-sweep-%j.out
#SBATCH --error=outputs/SLURM/factor-sweep-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh
announce_results "outputs/reports/concept_factor_sweep/summary.md (and summary.json)" \
  "per setting: outputs/shared/<dataset>/factor_sweep/<vocab>/text_<t>_coactivation_<c>/{groups,factors_merge_<m>}.json" \
  "new SpLiCE caches: ${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/<dataset>/splice_dataset_cache/"

"${PYTHON_BIN}" -u -m cospro.cli.sweep_concept_factors "$@"
