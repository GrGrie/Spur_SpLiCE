#!/bin/bash
# One-command launcher for docs/NEXT_ACTIONS_AFTER_TRANSFER_2026-09-07.md.
# It submits frozen-cache/graph preparation, integrity/preflight, corrected
# direct transfer (8 runs), CRP versus semantic-SpLiCE graph ablation (4 runs),
# and independent visual panels.
# All child stdout/stderr files are placed below RUN_ROOT/logs.
# Full SSL jobs use --mem=40G and --cpus-per-task=5; data loaders use 4 workers.
#
# Usage from the repository root on the cluster:
#   sbatch scripts/run_next_actions_after_transfer_2026-09-07.sbatch

#SBATCH --time=00:30:00
#SBATCH --job-name=splice_next_actions
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --partition=informatik-mind
#SBATCH --output=logs/next-actions-orchestrator-%j.out
#SBATCH --error=logs/next-actions-orchestrator-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xar68reb/Spur_SpLiCE}"
SPLICE_CONDA_ENV="${SPLICE_CONDA_ENV:-grgrie-train}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/outputs/next_actions_after_transfer_2026-09-07}"
PARTITION="${PARTITION:-informatik-mind}"
GPU_TYPE="${GPU_TYPE:-v100}"
TRAIN_MEMORY="40G"
TRAIN_CPUS=5
export PROJECT_DIR SPLICE_CONDA_ENV RUN_ROOT

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}/logs" \
    "${RUN_ROOT}/integrity" "${RUN_ROOT}/direct_transfer" \
    "${RUN_ROOT}/graph_ablation" "${RUN_ROOT}/visual"
source scripts/load_splice_cluster_env.sh

submit() {
    local name="$1" time_limit="$2" memory="$3" output_pattern="$4" error_pattern="$5"
    shift 5
    sbatch --parsable \
        --job-name="${name}" \
        --time="${time_limit}" \
        --mem="${memory}" \
        --ntasks=1 \
        --cpus-per-task="${TRAIN_CPUS}" \
        --nodes=1 \
        --partition="${PARTITION}" \
        --export="ALL,PROJECT_DIR=${PROJECT_DIR},SPLICE_CONDA_ENV=${SPLICE_CONDA_ENV},RUN_ROOT=${RUN_ROOT}" \
        --output="${RUN_ROOT}/logs/${output_pattern}" \
        --error="${RUN_ROOT}/logs/${error_pattern}" \
        "$@"
}

wrap_python() {
    local command="$1"
    printf 'set -euo pipefail; export PROJECT_DIR=%q SPLICE_CONDA_ENV=%q RUN_ROOT=%q; cd %q; source scripts/load_splice_cluster_env.sh; "${PYTHON_BIN}" -u -m %s' \
        "${PROJECT_DIR}" "${SPLICE_CONDA_ENV}" "${RUN_ROOT}" "${PROJECT_DIR}" "${command}"
}

DIRECT_CONFIG="${PROJECT_DIR}/scripts/next_actions_direct_transfer_2026-09-07.conf"
GRAPH_CONFIG="${PROJECT_DIR}/scripts/next_actions_graph_ablation_2026-09-07.conf"
BASE_CONFIG="${PROJECT_DIR}/scripts/run_crp_controls_cluster.conf"
CACHE_PATH="${PROJECT_DIR}/outputs/crp_controls_logistic_v1_cluster_seeds12/waterbirds_train_features.pt"
CRP_GRAPH_PATH="${PROJECT_DIR}/outputs/crp_controls_logistic_v1_cluster_seeds12/graphs/crp_graph.json"
TARGET_PATH="${PROJECT_DIR}/outputs/next_actions_after_transfer_2026-09-07/direct_transfer/targets_v1.pt"
SEMANTIC_GRAPH_PATH="${PROJECT_DIR}/outputs/next_actions_after_transfer_2026-09-07/graph_ablation/graphs/semantic_splice_graph.json"

INTEGRITY_CMD="scripts.tools.check_saved_probes --config scripts/downstream_probe_integrity_v1.conf --output '${RUN_ROOT}/integrity'"
INTEGRITY="$(submit splice-next-integrity 01:00:00 8G integrity-%j.out integrity-%j.err --wrap="$(wrap_python "${INTEGRITY_CMD}")")"

BASE_PREP_CMD="scripts.tools.run_crp_controls --config '${BASE_CONFIG}' --prepare-only"
BASE_PREP="$(submit splice-next-base-prep 08:00:00 "${TRAIN_MEMORY}" base-prepare-%j.out base-prepare-%j.err --wrap="$(wrap_python "${BASE_PREP_CMD}")")"

DIRECT_PREP_CMD="scripts.tools.run_concept_transfer --config '${DIRECT_CONFIG}' --stage prepare"
DIRECT_PREP="$(submit splice-next-direct-prep 00:30:00 8G direct-prepare-%j.out direct-prepare-%j.err --dependency="afterok:${INTEGRITY}:${BASE_PREP}" --wrap="$(wrap_python "${DIRECT_PREP_CMD}")")"

GRAPH_PREP_CMD="scripts.tools.run_graph_ablation --config '${GRAPH_CONFIG}' --stage prepare"
# Graph preparation is label-free and independent of the saved-probe gate, but
# it must wait for the shared frozen cache and CRP graph to be restored.
GRAPH_PREP="$(submit splice-next-graph-prep 01:00:00 8G graph-prepare-%j.out graph-prepare-%j.err --dependency="afterok:${BASE_PREP}" --wrap="$(wrap_python "${GRAPH_PREP_CMD}")")"

SMOKE_CMD="scripts.tools.run_concept_transfer --config '${DIRECT_CONFIG}' --stage smoke && '${PYTHON_BIN}' -u -m scripts.tools.run_next_actions_smoke --cache '${CACHE_PATH}' --targets '${TARGET_PATH}' --graph '${CRP_GRAPH_PATH}' --graph '${SEMANTIC_GRAPH_PATH}' --output '${RUN_ROOT}/integrity/preflight_smoke.json'"
SMOKE="$(submit splice-next-smoke 00:30:00 8G smoke-%j.out smoke-%j.err --dependency="afterok:${DIRECT_PREP}:${GRAPH_PREP}" --wrap="$(wrap_python "${SMOKE_CMD}")")"

DIRECT_RUN_CMD="scripts.tools.run_concept_transfer --config '${DIRECT_CONFIG}' --stage run --array-task-id \${SLURM_ARRAY_TASK_ID}"
DIRECT_ARRAY="$(submit splice-next-direct 24:00:00 "${TRAIN_MEMORY}" direct-%A_%a.out direct-%A_%a.err --gres="gpu:${GPU_TYPE}:1" --gpu_cmode=shared --array=0-7%8 --dependency="afterok:${SMOKE}" --wrap="$(wrap_python "${DIRECT_RUN_CMD}")")"

GRAPH_RUN_CMD="scripts.tools.run_graph_ablation --config '${GRAPH_CONFIG}' --stage run --array-task-id \${SLURM_ARRAY_TASK_ID}"
GRAPH_ARRAY="$(submit splice-next-graph 24:00:00 "${TRAIN_MEMORY}" graph-%A_%a.out graph-%A_%a.err --gres="gpu:${GPU_TYPE}:1" --gpu_cmode=shared --array=0-3%4 --dependency="afterok:${SMOKE}" --wrap="$(wrap_python "${GRAPH_RUN_CMD}")")"

DIRECT_SUMMARY_CMD="scripts.tools.run_concept_transfer --config '${DIRECT_CONFIG}' --stage summary"
DIRECT_SUMMARY="$(submit splice-next-direct-summary 00:30:00 8G direct-summary-%j.out direct-summary-%j.err --dependency="afterany:${DIRECT_ARRAY}" --wrap="$(wrap_python "${DIRECT_SUMMARY_CMD}")")"

GRAPH_SUMMARY_CMD="scripts.tools.run_graph_ablation --config '${GRAPH_CONFIG}' --stage summary"
GRAPH_SUMMARY="$(submit splice-next-graph-summary 00:30:00 8G graph-summary-%j.out graph-summary-%j.err --dependency="afterany:${GRAPH_ARRAY}" --wrap="$(wrap_python "${GRAPH_SUMMARY_CMD}")")"

VISUAL_CMD="scripts.tools.render_next_actions_panels --cache '${CACHE_PATH}' --graph '${CRP_GRAPH_PATH}' --data-folder /home/xar68reb/Datasets --output-png '${RUN_ROOT}/visual/concept_panels.png' --output-pdf '${RUN_ROOT}/visual/concept_panels.pdf' --output-json '${RUN_ROOT}/visual/concept_panels.json'"
VISUAL="$(submit splice-next-visual 04:00:00 16G visual-%j.out visual-%j.err --dependency="afterok:${GRAPH_PREP}" --wrap="$(wrap_python "${VISUAL_CMD}")")"

MANIFEST="${RUN_ROOT}/submitted_${SLURM_JOB_ID}.json"
"${PYTHON_BIN}" - "${MANIFEST}" "${INTEGRITY}" "${BASE_PREP}" "${DIRECT_PREP}" "${GRAPH_PREP}" "${SMOKE}" "${DIRECT_ARRAY}" "${GRAPH_ARRAY}" "${DIRECT_SUMMARY}" "${GRAPH_SUMMARY}" "${VISUAL}" <<'PY'
import json
import sys
from pathlib import Path

names = [
    "integrity", "base_prepare", "direct_prepare", "graph_prepare",
    "preflight_smoke", "direct_array", "graph_array", "direct_summary",
    "graph_summary", "visual",
]
payload = {name: job_id for name, job_id in zip(names, sys.argv[2:])}
payload.update({
    "status": "submitted",
    "protocol": "next_actions_after_transfer_2026-09-07",
    "direct_runs": 8,
    "graph_ablation_runs": 4,
    "ssl_epochs": 6000,
    "training_resources": {"mem": "40G", "cpus_per_task": 5, "num_workers": 4},
    "temperature": {"simclr": 0.05, "graph_kl": 0.25},
    "results_root": str(Path(sys.argv[1]).parent),
    "logs": str(Path(sys.argv[1]).parent / "logs"),
})
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Submitted frozen-cache preparation, corrected direct transfer, CRP-vs-semantic graph ablation, and visual panel jobs."
echo "Job manifest: ${MANIFEST}"
echo "All child .out/.err files: ${RUN_ROOT}/logs/"
