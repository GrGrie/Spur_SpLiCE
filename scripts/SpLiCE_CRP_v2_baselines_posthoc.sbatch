#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --job-name=SpLiCE_CRP_v2_compare
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
METADATA_CSV="${METADATA_CSV:-./datasets/waterbirds/metadata.csv}"
GRAPH_FILES="${GRAPH_FILES:-outputs/crp/waterbirds_v2/teacher_graph_seed0.pt outputs/crp/waterbirds_baselines/raw_clip_graph.pt outputs/crp/waterbirds_baselines/dino_graph.pt}"

cd "${PROJECT_DIR}"
mkdir -p ./output
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

read -r -a FILES <<< "${GRAPH_FILES}"
python -u scripts/tools/evaluate_crp_graph_waterbirds.py \
  --graph "${FILES[@]}" \
  --metadata-csv "${METADATA_CSV}"

conda deactivate
