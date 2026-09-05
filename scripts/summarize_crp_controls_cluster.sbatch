#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=SpurSpLiCE_CRP_summary
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=./output/crp_controls_cluster/summary-%j.out
#SBATCH --error=./output/crp_controls_cluster/summary-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${PROJECT_DIR}/scripts/run_crp_controls_cluster.conf"

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-grgrie-train}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u -m scripts.tools.run_crp_controls \
    --config "${CONFIG_FILE}" \
    --summarize-only
