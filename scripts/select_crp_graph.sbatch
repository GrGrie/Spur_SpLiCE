#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=SpurSpLiCE_graph_select
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=./output/crp_graph_search/select-%j.out
#SBATCH --error=./output/crp_graph_search/select-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${PROJECT_DIR}/scripts/crp_graph_search.conf"

cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-grgrie-train}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u -m scripts.tools.search_crp_graphs \
    --config "${CONFIG_FILE}" \
    --select-only
