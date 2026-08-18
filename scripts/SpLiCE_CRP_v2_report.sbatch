#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --job-name=SpLiCE_CRP_v2_report
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=./output/%x_%j.out
#SBATCH --error=./output/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONDA_ENV="${CONDA_ENV:-grgrie-train}"
REPORT_FILES="${REPORT_FILES:-outputs/crp/waterbirds/teacher_graph_seed0.json}"

cd "${PROJECT_DIR}"
mkdir -p ./output

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

read -r -a FILES <<< "${REPORT_FILES}"
python -u scripts/tools/summarize_crp_audit.py "${FILES[@]}"

conda deactivate
