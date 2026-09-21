#!/usr/bin/env bash
# One-off check of phase 8b: concept dictionaries did not change the SpLiCE cache.
#
# The job builds the Waterbirds SpLiCE cache three times on one V100 and compares them:
#   baseline    the code before phase 8b (git worktree at BASELINE_COMMIT), openimages_v7
#   current     this checkout, openimages_v7
#   file        this checkout, a 'file' dictionary holding the same Open Images words
# baseline == current must hold for every field, provenance included. current == file must hold for
# every representation; their provenance differs on purpose, since one names a file dictionary.
# Each build gets its own HOME, so no build reuses another's cached dictionary embeddings; the CLIP
# weights come from the shared Hugging Face cache.
#
#   sbatch scripts/verify_concept_dictionary.sbatch
#   BASELINE_COMMIT=<sha> sbatch scripts/verify_concept_dictionary.sbatch
#
# Read the verdict at the end of the .out file: "[verify] PASS" or "[verify] FAIL".

#SBATCH --job-name=cospro-verify-dictionary
#SBATCH --partition=informatik-mind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/SLURM/verify-dictionary-%j.out
#SBATCH --error=outputs/SLURM/verify-dictionary-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

RUN_ID="${SLURM_JOB_ID:-local-$$}"
BASELINE_COMMIT="${BASELINE_COMMIT:-8398090}"
DATASET="${DATASET:-waterbirds}"
REPORT_DIR="${PROJECT_DIR}/outputs/reports/concept_dictionary_check/${RUN_ID}"
WORK_DIR="${SPUR_SPLICE_SCRATCH_ROOT}/verify-dictionary/${RUN_ID}"
source scripts/announce_results.sh
announce_results "${REPORT_DIR}/" "verdict: the last line of this .out file" "temporary caches: ${WORK_DIR} (deleted at the end)"

# The comparison needs the real CLIP weights. Every build shares this cache and keeps its own HOME.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${REPORT_DIR}" "${WORK_DIR}"
cleanup() {
  git -C "${PROJECT_DIR}" worktree remove --force "${WORK_DIR}/baseline-code" 2>/dev/null || true
  git -C "${PROJECT_DIR}" worktree prune || true
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

echo "[verify] baseline ${BASELINE_COMMIT}, current $(git rev-parse --short HEAD), dataset ${DATASET}"
git worktree add --detach "${WORK_DIR}/baseline-code" "${BASELINE_COMMIT}"

# build NAME CODE_DIR [cache options...]: one cache, from the code in CODE_DIR, with its own HOME.
build() {
  local name="$1" code_dir="$2"
  shift 2
  mkdir -p "${WORK_DIR}/home-${name}"
  echo "[verify] building the ${name} cache"
  (cd "${code_dir}" && HOME="${WORK_DIR}/home-${name}" PYTHONPATH="${code_dir}" \
    "${PYTHON_BIN}" -u -m scripts.tools.cache_splice_dataset \
      --dataset "${DATASET}" --data-folder "${DATA_FOLDER}" \
      --output-root "${WORK_DIR}/caches-${name}" --device cuda --num-workers 4 "$@")
}
cache_of() {
  find "${WORK_DIR}/caches-$1" -name splice_dataset_cache.pt | head -n 1
}

build baseline "${WORK_DIR}/baseline-code" --splice-vocab openimages_v7
build current "${PROJECT_DIR}" --splice-vocab openimages_v7
build file "${PROJECT_DIR}" --splice-vocab file --splice-vocab-file "${PROJECT_DIR}/data/vocab/openimages_v7.txt"

set +e
"${PYTHON_BIN}" -u -m scripts.tools.compare_splice_caches "$(cache_of baseline)" "$(cache_of current)" \
  --label "baseline vs current (openimages_v7)" --report "${REPORT_DIR}/baseline_vs_current.json"
named_status=$?
"${PYTHON_BIN}" -u -m scripts.tools.compare_splice_caches "$(cache_of current)" "$(cache_of file)" \
  --ignore-provenance --label "bundled vs file dictionary" --report "${REPORT_DIR}/bundled_vs_file.json"
file_status=$?
set -e

if [[ "${named_status}" == "0" && "${file_status}" == "0" ]]; then
  echo "[verify] PASS: the cache is unchanged and a file dictionary reproduces the bundled one"
else
  echo "[verify] FAIL: see ${REPORT_DIR}/*.json (baseline vs current exit=${named_status}, file exit=${file_status})"
  exit 1
fi
