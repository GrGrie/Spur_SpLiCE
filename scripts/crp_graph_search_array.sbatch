#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=SpurSpLiCE_graph_search
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --array=0-8%9
#SBATCH --output=./output/crp_graph_search/search-%A_%a.out
#SBATCH --error=./output/crp_graph_search/search-%A_%a.err

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
    --array-task-id "${SLURM_ARRAY_TASK_ID}"
