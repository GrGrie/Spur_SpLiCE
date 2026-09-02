#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CQT
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/CRPv2/slurm-%x_%j.out
#SBATCH --error=./output/CRPv2/slurm-%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/scripts/train_cqt.conf}"
CONFIG_ALREADY_LOADED="${CONFIG_ALREADY_LOADED:-false}"
if [[ "${CONFIG_ALREADY_LOADED}" != "true" ]]; then
    source "${PROJECT_DIR}/scripts/load_config.sh"
    load_splice_config "${CONFIG_FILE}"
fi

# train_crp.sbatch contains the shared implementation. The CQT config has
# already supplied MODE and all shared values, so it must not be loaded again.
CONFIG_ALREADY_LOADED="true"
source "${PROJECT_DIR}/scripts/train_crp.sbatch"
