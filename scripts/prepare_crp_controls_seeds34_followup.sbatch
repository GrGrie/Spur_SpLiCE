#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=SpurSpLiCE_s34_lock
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=./output/crp_followup/%x-%j.out
#SBATCH --error=./output/crp_followup/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
TEMPLATE="${PROJECT_DIR}/scripts/run_crp_controls_seeds34_followup.conf"
OUTPUT="/home/xar68reb/Spur_SpLiCE/outputs/crp_followup_controls_v1_seeds34"
LOCKED_CONFIG="${OUTPUT}/locked_config.json"
SOURCE_OUTPUT="/home/xar68reb/Spur_SpLiCE/outputs/crp_controls_logistic_v1_cluster_seeds12"

cd "${PROJECT_DIR}"
mkdir -p ./output/crp_followup
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-grgrie-train}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u -m scripts.tools.lock_crp_followup \
    --template "${TEMPLATE}" \
    --source-output "${SOURCE_OUTPUT}" \
    --output "${LOCKED_CONFIG}"
python -u -m scripts.tools.run_crp_controls \
    --config "${LOCKED_CONFIG}" \
    --prepare-only
