#!/usr/bin/env bash
# Fine-tune the last blocks of an official I-JEPA ViT-H/14 on Waterbirds with a linear head, then
# refit the head on the group-balanced subset (DFR). Trains on cb_train by default: equal bird types,
# 95% of each on its usual background.
#
#   sbatch scripts/finetune_ijepa.sbatch                                  # IN-1k, val only
#   sbatch scripts/finetune_ijepa.sbatch --final_test                     # also evaluate on test
#   MODEL=ijepa_vith14_in22k SEED=1 sbatch scripts/finetune_ijepa.sbatch --final_test
#   sbatch scripts/finetune_ijepa.sbatch --trainable_blocks 4 --gradient_checkpointing   # 16 GB V100
#
# Further options after the script name go to cospro.cli.finetune_ijepa.

#SBATCH --job-name=cospro-ft-ijepa
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/SLURM/finetune-ijepa-%j.out
#SBATCH --error=outputs/SLURM/finetune-ijepa-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh

MODEL="${MODEL:-ijepa_vith14_in1k}"
SEED="${SEED:-0}"
STUDY="ijepa_finetune"
ATTEMPT_ID="job${SLURM_JOB_ID:-local}"
SEED_DIR="seed_$(printf '%02d' "${SEED}")"

announce_results "outputs/seeds/${STUDY}/${SEED_DIR}/${MODEL}/${ATTEMPT_ID}/run.json" \
  "trained parameters: ${SPUR_SPLICE_SCRATCH_ROOT}/checkpoints/Spur_SpLiCE/${STUDY}/${SEED_DIR}/${MODEL}/${ATTEMPT_ID}/finetuned.pt" \
  "I-JEPA checkpoint: ${SPUR_SPLICE_SCRATCH_ROOT}/pretrained/ijepa/" \
  "W&B: the run link printed by wandb below"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PYTHON_BIN}" -u -m cospro.cli.finetune_ijepa \
  --model "${MODEL}" \
  --data_folder "${DATA_FOLDER}" \
  --seed "${SEED}" \
  --num_workers 4 \
  --study "${STUDY}" \
  --attempt_id "${ATTEMPT_ID}" \
  --use_wandb \
  "$@"
