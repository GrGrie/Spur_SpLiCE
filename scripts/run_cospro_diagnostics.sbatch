#!/usr/bin/env bash
#SBATCH --job-name=cospro-diagnostics
#SBATCH --partition=informatik-mind
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/SLURM/cospro-diagnostics-%j.out
#SBATCH --error=outputs/SLURM/cospro-diagnostics-%j.err

# Concept-group and teacher-graph diagnostics with every metric tier: label-free metrics, metrics
# that read the frozen SpLiCE cache from scratch and post-hoc label metrics. CPU only.
#
#   sbatch scripts/run_cospro_diagnostics.sbatch                       # Waterbirds defaults
#   DATASET=celeba sbatch scripts/run_cospro_diagnostics.sbatch --graph main=PATH --reference main
#
# Environment: DATASET (waterbirds), CACHE (auto: the single cache under scratch),
# BOOTSTRAP_TRIALS (2), SWEEP_DIR (outputs/shared/<dataset>/graphs/concept_groups).
# Render the dashboard locally afterwards:
#   python -m cospro.diagnostics dashboard outputs/reports/cospro_diagnostics/<dataset>/diagnostics.json \
#       --data-folder <datasets> --output diagnostics.html

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

DATASET="${DATASET:-waterbirds}"
OUTPUT="${PROJECT_DIR}/outputs/reports/cospro_diagnostics/${DATASET}/diagnostics.json"
SWEEP_DIR="${SWEEP_DIR:-${PROJECT_DIR}/outputs/shared/${DATASET}/graphs/concept_groups}"
source scripts/announce_results.sh
announce_results "${OUTPUT}" "render locally: python -m cospro.diagnostics dashboard ${OUTPUT#${PROJECT_DIR}/} --output diagnostics.html"

if [[ -z "${CACHE:-}" ]]; then
  mapfile -t caches < <(find "${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/${DATASET}/splice_dataset_cache" \
    -name splice_dataset_cache.pt 2>/dev/null | sort)
  if [[ "${#caches[@]}" != "1" ]]; then
    echo "ERROR: found ${#caches[@]} SpLiCE caches for ${DATASET}; set CACHE=PATH." >&2
    printf '  %s\n' "${caches[@]}" >&2
    exit 2
  fi
  CACHE="${caches[0]}"
fi
echo "[diagnostics] cache=${CACHE}"

graph_args=()
if [[ "${DATASET}" == "waterbirds" && $# -eq 0 ]]; then
  graphs="${PROJECT_DIR}/outputs/shared/waterbirds/graphs"
  graph_args=(
    --graph "cospro=${graphs}/crp_graph.json"
    --graph "raw_clip=${graphs}/raw_clip_graph.json"
    --graph "semantic_splice=${graphs}/semantic_splice_graph.json"
    --reference raw_clip
  )
fi

"${PYTHON_BIN}" -u -m cospro.diagnostics evaluate \
  --dataset "${DATASET}" \
  --sweep-dir "${SWEEP_DIR}" \
  --splice-dataset-cache "${CACHE}" \
  --data-folder "${DATA_FOLDER}" \
  --bootstrap-trials "${BOOTSTRAP_TRIALS:-2}" \
  --output "${OUTPUT}" \
  "${graph_args[@]}" "$@"
