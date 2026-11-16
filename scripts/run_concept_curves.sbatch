#!/usr/bin/env bash
# One supervised run of the concept-curves project (docs/PROJECT_PLAN.md): ERM with the concept audit
# or the early concept swap. All options go to cospro.cli.train_concept_curves.
#
#   sbatch scripts/run_concept_curves.sbatch --dataset metashift --seed 1 --mode erm \
#       --concept-maps <maps.pt> --study curves_metashift --arm erm

#SBATCH --job-name=cospro-curves
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/SLURM/curves-%j.out
#SBATCH --error=outputs/SLURM/curves-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh
announce_results "outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/run.json (exact path printed below as [results])" \
  "audit: concept_curves.json and flagged_units.json in the same folder" \
  "study summary: python -m cospro.cli.summarize_concept_curves --study <study>" \
  "binaries: none, this job writes no checkpoint"

"${PYTHON_BIN}" -u -m cospro.cli.train_concept_curves --data-folder "${DATA_FOLDER}" --num-workers 4 --wandb "$@"
