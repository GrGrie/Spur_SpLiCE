#!/usr/bin/env bash
#SBATCH --job-name=cospro-golden-smoke
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/SLURM/golden-smoke-%j.out
#SBATCH --error=outputs/SLURM/golden-smoke-%j.err

# Runs the full test suite on the cluster (including the Linux golden training snapshot)
# and then every training mode for two epochs on the V100 with AMP and channels-last memory.
# The GPU run summary lands in outputs/reports/golden-smoke/ and synchronizes through Git.
# A first run creates tests/golden/training_runs.linux.json; review and commit it.
#
#   sbatch scripts/run_golden_smoke.sbatch

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

RUN_ID="${SLURM_JOB_ID:-local-$$}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[golden] CPU test suite"
"${PYTHON_BIN}" -m pytest -q --basetemp="${SPUR_SPLICE_SCRATCH_ROOT}/golden-smoke/${RUN_ID}/pytest"

echo "[golden] GPU training modes"
SPUR_SPLICE_GOLDEN_DEVICE=cuda \
SPUR_SPLICE_GOLDEN_REPORT="outputs/reports/golden-smoke/${RUN_ID}.json" \
  "${PYTHON_BIN}" -m pytest -q -rs tests/test_golden_training.py \
  --basetemp="${SPUR_SPLICE_SCRATCH_ROOT}/golden-smoke/${RUN_ID}/pytest-gpu"

rm -rf "${SPUR_SPLICE_SCRATCH_ROOT}/golden-smoke/${RUN_ID}"
echo "[golden] report: outputs/reports/golden-smoke/${RUN_ID}.json"
