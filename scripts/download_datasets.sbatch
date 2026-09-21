#!/usr/bin/env bash
# Download Waterbirds and CelebA into DATA_FOLDER and verify them against the paper's files.
# CPU only. Compute nodes need outbound HTTPS for this; without it, run the same command on any
# machine with internet and copy DATA_FOLDER over:
#   python -m cospro.cli.download_datasets --data-folder DATA_FOLDER
#
#   sbatch scripts/download_datasets.sbatch                       # both datasets
#   sbatch scripts/download_datasets.sbatch --datasets waterbirds
#   sbatch scripts/download_datasets.sbatch --datasets celeba --celeba-archive-dir /path/to/official

#SBATCH --job-name=cospro-download-datasets
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/SLURM/download-datasets-%j.out
#SBATCH --error=outputs/SLURM/download-datasets-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh
source scripts/announce_results.sh
announce_results "${DATA_FOLDER}/waterbirds and ${DATA_FOLDER}/celeba" "verification: the [verify] lines below and dataset_manifest.json in each dataset folder"

"${PYTHON_BIN}" -u -m cospro.cli.download_datasets --data-folder "${DATA_FOLDER}" "$@"
