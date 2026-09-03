#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=SpurSpLiCE_CRP_semantic
#SBATCH --array=0-2%2
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=informatik-mind
#SBATCH --gpu_cmode=shared
#SBATCH --output=./output/CRPv3/slurm-%x_%A_%a.out
#SBATCH --error=./output/CRPv3/slurm-%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${PROJECT_DIR}/scripts/prepare_crp_semantic_family_sweep.conf"
source "${PROJECT_DIR}/scripts/prepare_crp_group_sweep.sbatch"
