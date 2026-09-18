#!/usr/bin/env bash
#SBATCH --job-name=cospro-figures
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/SLURM/graph-figures-%j.out
#SBATCH --error=outputs/SLURM/graph-figures-%j.err

set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh
announce_results "${PROJECT_DIR}/outputs/reports/graph_figures_${SLURM_JOB_ID}"
"${PYTHON_BIN}" -u -m tools.paper.build_submission_figure \
  --output-dir "${PROJECT_DIR}/outputs/reports/graph_figures_${SLURM_JOB_ID}" "$@"
