#!/usr/bin/env bash
#SBATCH --job-name=cospro-test-probes
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-4
#SBATCH --output=outputs/SLURM/test-probes-%A_%a.out
#SBATCH --error=outputs/SLURM/test-probes-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.evaluate_submission_checkpoints run \
    --seed "${SLURM_ARRAY_TASK_ID:?Submit this script as a SLURM array}" "$@"
