#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=splice_prepare_lambda05
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
cd "${PROJECT_DIR}"
mkdir -p output outputs/crp_lambda05_replication_s34
module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-grgrie-train}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python -u -m scripts.tools.lock_crp_followup \
    --template "${PROJECT_DIR}/scripts/crp_lambda05_replication_s34.conf" \
    --source-output "${PROJECT_DIR}/outputs/crp_controls_logistic_v1_cluster_seeds12" \
    --output "${PROJECT_DIR}/outputs/crp_lambda05_replication_s34/locked_config.json"
python -u -m scripts.tools.run_crp_controls \
    --config "${PROJECT_DIR}/outputs/crp_lambda05_replication_s34/locked_config.json" \
    --prepare-only
