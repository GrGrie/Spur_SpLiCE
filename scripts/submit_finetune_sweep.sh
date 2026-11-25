#!/usr/bin/env bash
# Submit the fine-tuning sweep for one model: validation only, one seed, no checkpoints.
# Every job is scripts/finetune_ijepa.sbatch, so each .out file announces its own run.json.
#
#   bash scripts/submit_finetune_sweep.sh ijepa_vith14_in1k        # 3 depths x 3 lengths x 2 rates = 18 jobs
#   bash scripts/submit_finetune_sweep.sh resnet50_pretrained      # reference: 2 x 3 x 2 = 12 jobs
#
# Afterwards pick each model's setting by last-5-epoch validation WGA:
#   python -m cospro.cli.summarize_finetune
# and rerun that setting with --final_test on seeds 0, 1 and 2.

set -euo pipefail
MODEL="${1:-ijepa_vith14_in1k}"
SEED="${SEED:-0}"
if [[ "${MODEL}" == ijepa_* ]]; then
  BLOCKS=(4 8 16)      # of 32 transformer blocks
  RATES=(1e-5 3e-5)
else
  BLOCKS=(3 16)        # the last stage or all 16 bottleneck blocks
  RATES=(1e-4 3e-4)
fi

for blocks in "${BLOCKS[@]}"; do
  for epochs in 10 30 100; do
    for lr in "${RATES[@]}"; do
      extra=()
      if [[ "${MODEL}" == ijepa_* && "${blocks}" -ge 16 ]]; then
        extra=(--gradient_checkpointing)
      fi
      MODEL="${MODEL}" SEED="${SEED}" sbatch --job-name="ft-${MODEL##*_}-b${blocks}-e${epochs}-lr${lr}" \
        scripts/finetune_ijepa.sbatch --trainable_blocks "${blocks}" --epochs "${epochs}" --lr "${lr}" \
        --no-save_checkpoint "${extra[@]}"
    done
  done
done
