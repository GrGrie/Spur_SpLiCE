#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --job-name=SpLiCE_CRP_v2_posthoc
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=./output/%x_%j.out
#SBATCH --error=./output/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
GRAPH_PATH="${GRAPH_PATH:-outputs/crp/waterbirds_v2/teacher_graph_seed${TASK_ID}.pt}"
METADATA_CSV="${METADATA_CSV:-./datasets/waterbirds/metadata.csv}"

cd "${PROJECT_DIR}"
mkdir -p ./output

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -u scripts/tools/evaluate_crp_graph_waterbirds.py \
  --graph "${GRAPH_PATH}" \
  --metadata-csv "${METADATA_CSV}"

conda deactivate
