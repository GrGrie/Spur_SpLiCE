#!/bin/bash
#SBATCH --job-name=paper_test_summary
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/paper-test-summary-%j.out
#SBATCH --error=logs/paper-test-summary-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
cd "${PROJECT_DIR}"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.paper_completion test_summary
