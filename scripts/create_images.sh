#!/usr/bin/env bash
#SBATCH --job-name=cospro-figure
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/SLURM/figure-%j.out
#SBATCH --error=outputs/SLURM/figure-%j.err

set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR}"
WATERBIRDS_ROOT="/home/xar68reb/Datasets/waterbirds"
cd "${PROJECT_DIR}"

source scripts/load_splice_cluster_env.sh

test -f "${WATERBIRDS_ROOT}/metadata.csv"

"${PYTHON_BIN}" -u -m scripts.tools.build_submission_figure \
 --dataset-root "${WATERBIRDS_ROOT}" \
 --artifact-root "${PROJECT_DIR}/outputs" \
 --output-dir "${PROJECT_DIR}/outputs/reports/submission_graph_figure_${SLURM_JOB_ID}"
