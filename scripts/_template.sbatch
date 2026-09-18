#!/usr/bin/env bash
# Template for a new cluster launcher. Copy it to scripts/<task>.sbatch and edit the marked parts.
#
# Resource rules (checked by tests/test_slurm_launchers.py):
#   one V100 through --gres=gpu:1 for GPU work (delete the line for CPU-only jobs)
#   --cpus-per-task at most 5 and --mem at most 8G per CPU (1 -> 8G, 2 -> 16G, 5 -> 40G)
#   logs under outputs/SLURM/
# Hyperparameters come from the experiment manifest or from "$@"; the launcher holds
# resources, paths and the environment only.
#
#   sbatch scripts/<task>.sbatch [python options]

#SBATCH --job-name=cospro-TASK
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/SLURM/TASK-%j.out
#SBATCH --error=outputs/SLURM/TASK-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

# Binaries go to "${SPUR_SPLICE_SCRATCH_ROOT}", compact JSON results go under outputs/.
"${PYTHON_BIN}" -u -m package.module "$@"
