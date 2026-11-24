#!/usr/bin/env bash
# Linear probe of a frozen, officially pretrained I-JEPA encoder (arXiv:2301.08243) on Waterbirds.
#
# The first run downloads the official checkpoint (about 10 GB) to
# ${SPUR_SPLICE_SCRATCH_ROOT}/pretrained/ijepa/; later runs reuse it. The probe is the project's
# standard one, so its accuracy and worst-group accuracy compare directly with SimCLR runs.
#
#   sbatch scripts/probe_ijepa.sbatch                      # ViT-H/14 IN-1k, val split
#   sbatch scripts/probe_ijepa.sbatch --final_test         # test split
#   MODEL=ijepa_vith14_in22k sbatch scripts/probe_ijepa.sbatch
#   sbatch scripts/probe_ijepa.sbatch --final_test --train_set_linear_layer cb_train
#       # probe on equal classes with train's background correlation (1057/56/56/1057 images)
#
# Further options after the script name go to cospro.cli.linear_probe.

#SBATCH --job-name=cospro-probe-ijepa
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/SLURM/probe-ijepa-%j.out
#SBATCH --error=outputs/SLURM/probe-ijepa-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh

MODEL="${MODEL:-ijepa_vith14_in1k}"
SEED="${SEED:-0}"
STUDY="ijepa_waterbirds"
ATTEMPT_ID="job${SLURM_JOB_ID:-local}"
RESULT_DIR="outputs/seeds/${STUDY}/seed_$(printf '%02d' "${SEED}")/${MODEL}/${ATTEMPT_ID}"

announce_results "${RESULT_DIR}/probe_features_epoch_0_<train split>_<eval split>.json" \
  "I-JEPA checkpoint: ${SPUR_SPLICE_SCRATCH_ROOT}/pretrained/ijepa/" \
  "probe features: ${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/${STUDY}/" \
  "W&B: the run link printed by wandb below"

"${PYTHON_BIN}" -u -m cospro.cli.linear_probe \
  --dataset waterbirds \
  --data_folder "${DATA_FOLDER}" \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --batch_size 64 \
  --num_workers 4 \
  --artifact_dir "${RESULT_DIR}" \
  --study "${STUDY}" \
  --arm "${MODEL}" \
  --attempt_id "${ATTEMPT_ID}" \
  --use_wandb \
  "$@"
