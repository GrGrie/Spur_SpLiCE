#!/usr/bin/env bash
# Submit the kill test of the concept-curves project (docs/PROJECT_PLAN.md, Section 7).
#
#   STAGE=k1 bash scripts/submit_concept_curves.sh metashift 1 2 3 4   # ERM with the concept audit
#   STAGE=k2 bash scripts/submit_concept_curves.sh metashift 1 2 3 4   # swap arms, after K1 has finished
#
# Stage k1 runs the arm erm (resnet18_large from scratch). Stage k2 runs swap, swap_random, swap_bottom,
# swap_shuffled and swap_oracle; every swap run reads the curves of the erm run of its own seed.
# Environment:
#   ARMS="erm erm_rn50"   run these arms instead of the stage's default arms
#   MAPS=<concept_maps.pt>  concept maps; defaults to the maps of the spatial runs of the dataset
#   MAX_PARALLEL=8        at most this many jobs of the submission run at once
#   STUDY=<name>          study name, default curves_<dataset>
# Arms: erm, erm_rn50 (ImageNet-pretrained ResNet-50), swap, swap_random, swap_bottom, swap_shuffled,
# swap_oracle, swap_rn50 (reads erm_rn50).
set -euo pipefail

DATASET="${1:?Usage: STAGE=k1|k2 $0 metashift|waterbirds [SEED ...]}"
shift
SEEDS=("$@")
if [[ "${#SEEDS[@]}" == "0" ]]; then
  SEEDS=(1 2 3 4)
fi
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

STUDY="${STUDY:-curves_${DATASET}}"
case "${DATASET}" in
  metashift) MAPS_NAME="laion_text_0p85_response_0p50_min0.02_merge0_qq448" ;;
  *) MAPS_NAME="openimages_v7_text_0p85_response_0p50_min0.02_merge0_qq448" ;;
esac
MAPS="${MAPS:-${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/${DATASET}/concept_maps/${MAPS_NAME}/concept_maps.pt}"
if [[ ! -f "${MAPS}" ]]; then
  echo "ERROR: concept maps not found: ${MAPS}" >&2
  echo "Build them with scripts/build_concept_maps.sbatch or pass MAPS=<path>." >&2
  exit 2
fi

declare -A ARM_ARGS=(
  [erm]="--mode erm"
  [erm_rn50]="--mode erm --model resnet50_pretrained"
  [swap]="--mode swap --flag-rule top"
  [swap_random]="--mode swap --flag-rule random"
  [swap_bottom]="--mode swap --flag-rule bottom"
  [swap_shuffled]="--mode swap --flag-rule top --shuffle-maps"
  [swap_oracle]="--mode swap --flag-rule oracle"
  [swap_rn50]="--mode swap --flag-rule top --model resnet50_pretrained --curves-arm erm_rn50"
)
case "${STAGE:-k1}" in
  k1) DEFAULT_ARMS="erm" ;;
  k2) DEFAULT_ARMS="swap swap_random swap_bottom swap_shuffled swap_oracle" ;;
  *) echo "ERROR: STAGE is k1 or k2." >&2; exit 2 ;;
esac
read -r -a SELECTED <<< "${ARMS:-${DEFAULT_ARMS}}"
for arm in "${SELECTED[@]}"; do
  if [[ -z "${ARM_ARGS[${arm}]+set}" ]]; then
    echo "ERROR: unknown arm ${arm}. Arms: ${!ARM_ARGS[*]}" >&2
    exit 2
  fi
done

# At most MAX_PARALLEL jobs of this submission run at once: job i waits for job i - MAX_PARALLEL to end.
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SUBMITTED=()
for arm in "${SELECTED[@]}"; do
  for seed in "${SEEDS[@]}"; do
    dependency=()
    if (( ${#SUBMITTED[@]} >= MAX_PARALLEL )); then
      dependency=(--dependency="afterany:${SUBMITTED[${#SUBMITTED[@]} - MAX_PARALLEL]}")
    fi
    # shellcheck disable=SC2206
    arm_args=(${ARM_ARGS[${arm}]})
    job=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --job-name "${STUDY}-${arm}-s${seed}" \
      scripts/run_concept_curves.sbatch --dataset "${DATASET}" --seed "${seed}" --concept-maps "${MAPS}" \
      --study "${STUDY}" --arm "${arm}" "${arm_args[@]}")
    SUBMITTED+=("${job%%;*}")
    echo "${arm} seed=${seed} job=${job%%;*}${dependency[0]:+ (${dependency[0]})}"
  done
done
echo "Run records: outputs/seeds/${STUDY}/seed_<NN>/<arm>/<attempt>/run.json"
echo "Summary: python -m cospro.cli.summarize_concept_curves --study ${STUDY}"
