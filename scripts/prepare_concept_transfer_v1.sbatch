#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=splice_transfer_prepare
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=outputs/next_tests_2026-09-07/logs/transfer-prepare-%j.out
#SBATCH --error=outputs/next_tests_2026-09-07/logs/transfer-prepare-%j.err
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"; SPLICE_CONDA_ENV="${SPLICE_CONDA_ENV:-grgrie-train}"; export PROJECT_DIR SPLICE_CONDA_ENV
cd "${PROJECT_DIR}"; mkdir -p outputs/next_tests_2026-09-07/logs outputs/next_tests_2026-09-07/concept_transfer
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.run_concept_transfer --config scripts/concept_transfer_v1.conf --stage prepare
