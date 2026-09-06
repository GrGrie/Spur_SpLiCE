#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=crp_probe_sensitivity
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
CONFIG_FILE="${PROJECT_DIR}/scripts/crp_signal_checks.conf"
cd "${PROJECT_DIR}"
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-grgrie-train}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5

python -u -m scripts.tools.run_crp_signal_checks --config "${CONFIG_FILE}" --stage probes
