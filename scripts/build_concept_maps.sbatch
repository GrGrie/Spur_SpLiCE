#!/usr/bin/env bash
# Build the 14 x 14 concept maps of a dataset's training images from frozen CLIP, for the spatial
# concept loss (--factor_spatial_weight). One GPU pass over the training images.
#
#   sbatch scripts/build_concept_maps.sbatch --dataset metashift --concept-groups <groups.json> \
#     --splice-cache <splice_dataset_cache.pt> [--factor-min-frequency 0.02]

#SBATCH --job-name=cospro-concept-maps
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/SLURM/concept-maps-%j.out
#SBATCH --error=outputs/SLURM/concept-maps-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh
announce_results "outputs/shared/<dataset>/graphs/concept_maps/<name>/summary.json" \
  "maps: ${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/<dataset>/concept_maps/<name>/concept_maps.pt"

"${PYTHON_BIN}" -u -m cospro.cli.build_concept_maps ${DATA_FOLDER:+--data-folder "${DATA_FOLDER}"} --num-workers 4 "$@"
