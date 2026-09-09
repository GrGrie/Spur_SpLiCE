#!/usr/bin/env bash
#SBATCH --job-name=spur-splice-collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/SLURM/collect-%j.out
#SBATCH --error=outputs/SLURM/collect-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
MANIFEST="${MANIFEST:-experiments/manifests/waterbirds_crp.json}"
cd "${PROJECT_DIR}"
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.collect_results "${MANIFEST}"
