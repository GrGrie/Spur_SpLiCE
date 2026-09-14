#!/usr/bin/env bash
#SBATCH --job-name=Build-CoSpRo-Teacher-Graphs
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/SLURM/cospro-graphs-%j.out
#SBATCH --error=outputs/SLURM/cospro-graphs-%j.err

# Backward-compatible wrapper; use scripts/build_cospro_teacher_graphs.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/build_cospro_teacher_graphs.sh" "$@"
