#!/usr/bin/env bash
#SBATCH --job-name=spur-splice-collect
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/SLURM/collect-%j.out
#SBATCH --error=outputs/SLURM/collect-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR must be exported by submit_experiment.sh}"
MANIFEST="${MANIFEST:-experiments/manifests/waterbirds_cospro.json}"
cd "${PROJECT_DIR}"
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m scripts.tools.collect_results "${MANIFEST}"
