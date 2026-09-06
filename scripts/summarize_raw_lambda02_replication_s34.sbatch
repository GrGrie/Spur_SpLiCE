#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=splice_summary_raw02
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
cd "${PROJECT_DIR}"
mkdir -p output
source "${PROJECT_DIR}/scripts/load_splice_cluster_env.sh"

"${PYTHON_BIN}" -u -m scripts.tools.summarize_crp_replication \
    --config "${PROJECT_DIR}/scripts/raw_lambda02_replication_s34_summary.conf"
