#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=SpLiCE_CRP_v2_compare
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --mail-type=END,FAIL
#SBATCH --output=./output/%x_%j.out
#SBATCH --error=./output/%x_%j.err

set -euo pipefail

# One-command matched baseline experiment:
# 1) build raw-CLIP and DINO-only 10-NN graphs;
# 2) compare them with all three existing CRP v2 graphs post hoc.
# No custom shell variables or dependency jobs are required.

PROJECT_DIR="/home/xar68reb/Spur_SpLiCE"
CONDA_ENV="grgrie-train"
CACHE_PATH="outputs/crp/waterbirds_train_features.pt"
BASELINE_DIR="outputs/crp/waterbirds_baselines"
METADATA_CSV="./datasets/waterbirds/metadata.csv"
CRP_DIR="outputs/crp/waterbirds_v2"

cd "${PROJECT_DIR}"
mkdir -p "${BASELINE_DIR}" ./output

if [[ ! -f "${CACHE_PATH}" ]]; then
    echo "CRP feature cache not found: ${CACHE_PATH}" >&2
    exit 2
fi

for graph in \
    "${CRP_DIR}/teacher_graph_seed0.pt" \
    "${CRP_DIR}/teacher_graph_seed1.pt" \
    "${CRP_DIR}/teacher_graph_seed2.pt"; do
    if [[ ! -f "${graph}" ]]; then
        echo "CRP graph not found: ${graph}" >&2
        exit 2
    fi
done

module purge
module load miniforge3/latest
. "${ANACONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "=== Building matched raw-CLIP and DINO-only graphs ==="
python -u -m scripts.tools.build_crp_baseline_graphs \
    --cache "${CACHE_PATH}" \
    --output-dir "${BASELINE_DIR}" \
    --top-k 10 \
    --chunk-size 512

echo "=== Comparing CRP v2, raw CLIP, and DINO graphs ==="
python -u -m scripts.tools.evaluate_crp_graph_waterbirds \
    --graph \
        "${CRP_DIR}/teacher_graph_seed0.pt" \
        "${CRP_DIR}/teacher_graph_seed1.pt" \
        "${CRP_DIR}/teacher_graph_seed2.pt" \
        "${BASELINE_DIR}/raw_clip_graph.pt" \
        "${BASELINE_DIR}/dino_graph.pt" \
    --metadata-csv "${METADATA_CSV}"

echo "Finished CRP v2 baseline comparison; $(date --iso-8601=seconds)"
conda deactivate
