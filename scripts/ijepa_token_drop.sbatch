#!/usr/bin/env bash
# Concept shortcut attribution at inference for a frozen I-JEPA encoder on Waterbirds: remove the patches
# of one CSA concept unit at a time, encode again and read the shortcut index and the WGA change of fixed
# linear probes (ds_train and cb_train). Reuses the Waterbirds concept maps built on the concept-factors branch.
#
#   sbatch scripts/ijepa_token_drop.sbatch                          # IN-1k, val split
#   sbatch scripts/ijepa_token_drop.sbatch --final_test             # test split
#   MODEL=ijepa_vith14_in22k sbatch scripts/ijepa_token_drop.sbatch --final_test
#   sbatch scripts/ijepa_token_drop.sbatch --final_test --units 24  # finer units: water apart from waterbirds
#   sbatch scripts/ijepa_token_drop.sbatch --final_test --units 0   # single factors instead of 12 units
#
# Verdicts come from the shortcut index: the accuracy gain of the conflict groups over the aligned groups
# when a unit's patches are removed, minus the same for a shuffled region of equal size.
#
# Further options after the script name go to cospro.cli.ijepa_token_drop.

#SBATCH --job-name=cospro-ijepa-tokdrop
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/SLURM/ijepa-token-drop-%j.out
#SBATCH --error=outputs/SLURM/ijepa-token-drop-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh

MODEL="${MODEL:-ijepa_vith14_in1k}"
SEED="${SEED:-0}"
STUDY="ijepa_token_drop"
ATTEMPT_ID="job${SLURM_JOB_ID:-local}"
CONCEPT_MAPS="${CONCEPT_MAPS:-${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/waterbirds/concept_maps/openimages_v7_text_0p85_response_0p50_min0.02_merge0_qq448/concept_maps.pt}"

announce_results "outputs/seeds/${STUDY}/seed_$(printf '%02d' "${SEED}")/${MODEL}/${ATTEMPT_ID}/token_drop.json" \
  "I-JEPA checkpoint: ${SPUR_SPLICE_SCRATCH_ROOT}/pretrained/ijepa/"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PYTHON_BIN}" -u -m cospro.cli.ijepa_token_drop \
  --model "${MODEL}" \
  --data_folder "${DATA_FOLDER}" \
  --concept_maps "${CONCEPT_MAPS}" \
  --seed "${SEED}" \
  --num_workers 4 \
  --study "${STUDY}" \
  --attempt_id "${ATTEMPT_ID}" \
  "$@"
