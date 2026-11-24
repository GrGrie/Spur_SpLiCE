#!/usr/bin/env bash
# Concept attribution at inference for one finished run (docs/PROJECT_CSA.md, Section 6): occlude every audited
# concept unit in the validation images, classify them with the run's frozen encoder and probe, and record the
# change of each group's accuracy. The first job of a study also builds the validation concept maps.
# scripts/submit_concept_attribution.sh submits one job per seed.
#
#   sbatch scripts/attribute_concepts.sbatch --study factors_metashift --arm csa_audit --seed 1

#SBATCH --job-name=cospro-attribution
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/SLURM/attribution-%j.out
#SBATCH --error=outputs/SLURM/attribution-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh
announce_results "outputs/seeds/<study>/seed_<NN>/<arm>/<attempt>/concept_attribution_val.json" \
  "report: python -m cospro.cli.summarize_concept_curves --study <study> -> outputs/reports/<study>/concept_curves.md" \
  "validation maps: ${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/<dataset>/concept_maps/<name>_val/concept_maps.pt"

"${PYTHON_BIN}" -u -m cospro.cli.attribute_concepts ${DATA_FOLDER:+--data_folder "${DATA_FOLDER}"} --num_workers 4 "$@"
