#!/usr/bin/env bash
#SBATCH --job-name=la-ssl
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/SLURM/la-ssl-%j.out
#SBATCH --error=outputs/SLURM/la-ssl-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export DATA_FOLDER="${DATA_FOLDER:-/home/xar68reb/Datasets}"
cd "${PROJECT_DIR}"
source scripts/load_splice_cluster_env.sh
"${PYTHON_BIN}" -u -m experiments.runner experiments/manifests/waterbirds_la_ssl.json --arm la_ssl "$@"
