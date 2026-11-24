#!/usr/bin/env bash
# Concept Shortcut Attribution at inference (docs/PROJECT_CSA.md, Section 6): one scripts/attribute_concepts.sbatch
# job per seed on the finished runs of an arm that kept its final encoder (default csa_audit).
#
#   bash scripts/submit_concept_attribution.sh metashift [SEED ...]            (default seeds: 1 to 8)
#   TRAIN=1 bash scripts/submit_concept_attribution.sh metashift 1 2 3 4 5 6 7 8
#       first submits the training of the arm through scripts/submit_factor_study.sh (MAX_PARALLEL lanes, default 8);
#       each attribution job then waits for the training of its seed
#   ARM=csa_audit STUDY=factors_metashift ...                                   (defaults)
#
# The attribution jobs run one after another: the first one builds the validation concept maps, and each job takes
# a few minutes. With TRAIN=1 an attribution job starts only after its seed's training has ended, so the number of
# jobs running at once never exceeds MAX_PARALLEL.
# Report on any machine after git pull: python -m cospro.cli.summarize_concept_curves --study factors_<dataset>

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: [TRAIN=1] [ARM=csa_audit] bash scripts/submit_concept_attribution.sh <dataset> [SEED ...]" >&2
  exit 2
fi
DATASET="$1"
shift
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(1 2 3 4 5 6 7 8)
fi
ARM="${ARM:-csa_audit}"
STUDY="${STUDY:-factors_${DATASET}}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM

declare -A TRAINING_JOB
if [[ "${TRAIN:-0}" == "1" ]]; then
  lanes="${MAX_PARALLEL}"
  if (( ${#SEEDS[@]} > MAX_PARALLEL )); then
    # Training lanes then outlive the first attribution jobs; one lane less keeps the total at MAX_PARALLEL.
    lanes=$((MAX_PARALLEL - 1))
  fi
  submitted="$(ARMS="${ARM}" MAX_PARALLEL="${lanes}" bash scripts/submit_factor_study.sh "${DATASET}" "${SEEDS[@]}")"
  echo "${submitted}"
  for seed in "${SEEDS[@]}"; do
    job="$(sed -n "s/^${ARM} seed=${seed} job=\([0-9]*\).*/\1/p" <<<"${submitted}")"
    if [[ -z "${job}" ]]; then
      echo "No training job found for ${ARM} seed ${seed}." >&2
      exit 2
    fi
    TRAINING_JOB[${seed}]="${job}"
  done
fi

previous=""
for seed in "${SEEDS[@]}"; do
  conditions=()
  if [[ -n "${TRAINING_JOB[${seed}]:-}" ]]; then
    # afterany: a failed training must not block the chain; the attribution then stops with a clear message.
    conditions+=("afterany:${TRAINING_JOB[${seed}]}")
  fi
  if [[ -n "${previous}" ]]; then
    conditions+=("afterany:${previous}")
  fi
  dependency=()
  if [[ ${#conditions[@]} -gt 0 ]]; then
    dependency=(--dependency="$(IFS=,; echo "${conditions[*]}")")
  fi
  job="$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --job-name "${STUDY}-attribution-${ARM}-s${seed}" \
    scripts/attribute_concepts.sbatch --study "${STUDY}" --arm "${ARM}" --seed "${seed}")"
  previous="${job%%;*}"
  echo "attribution ${ARM} seed=${seed} job=${previous}${dependency[0]:+ (${dependency[0]})}"
done
echo "Per run: outputs/seeds/${STUDY}/seed_<NN>/${ARM}/<attempt>/concept_attribution_val.json"
echo "Report: python -m cospro.cli.summarize_concept_curves --study ${STUDY}"
