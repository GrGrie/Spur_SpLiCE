#!/usr/bin/env bash
# Patch-concept readout of a frozen I-JEPA encoder on Waterbirds: which CLIP/SpLiCE concepts its patch
# tokens carry, a concept-bottleneck probe and the concepts behind the mean-pool probe's decisions.
# Reuses the Waterbirds concept maps built on the concept-factors branch.
#
#   sbatch scripts/ijepa_concept_readout.sbatch                       # IN-1k, val split
#   sbatch scripts/ijepa_concept_readout.sbatch --final_test          # test split
#   MODEL=ijepa_vith14_in22k sbatch scripts/ijepa_concept_readout.sbatch
#   CONCEPT_MAPS=<..._pca/concept_maps.pt> sbatch scripts/ijepa_concept_readout.sbatch   # PCA control
#
# Further options after the script name go to cospro.cli.ijepa_concept_readout.

#SBATCH --job-name=cospro-ijepa-readout
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/SLURM/ijepa-readout-%j.out
#SBATCH --error=outputs/SLURM/ijepa-readout-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh

MODEL="${MODEL:-ijepa_vith14_in1k}"
SEED="${SEED:-0}"
STUDY="ijepa_concept_readout"
ATTEMPT_ID="job${SLURM_JOB_ID:-local}"
CONCEPT_MAPS="${CONCEPT_MAPS:-${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/waterbirds/concept_maps/openimages_v7_text_0p85_response_0p50_min0.02_merge0_qq448/concept_maps.pt}"

announce_results "outputs/seeds/${STUDY}/seed_$(printf '%02d' "${SEED}")/${MODEL}/${ATTEMPT_ID}/readout.json" \
  "ridge: ${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/${STUDY}/${MODEL}/${ATTEMPT_ID}/ridge.pt" \
  "I-JEPA checkpoint: ${SPUR_SPLICE_SCRATCH_ROOT}/pretrained/ijepa/"

"${PYTHON_BIN}" -u -m cospro.cli.ijepa_concept_readout \
  --model "${MODEL}" \
  --data_folder "${DATA_FOLDER}" \
  --concept_maps "${CONCEPT_MAPS}" \
  --seed "${SEED}" \
  --num_workers 4 \
  --study "${STUDY}" \
  --attempt_id "${ATTEMPT_ID}" \
  "$@"
