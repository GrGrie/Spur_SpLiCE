#!/usr/bin/env bash
# Submit the concept-factor (F2) study of one dataset: every arm is one scripts/run_training.sbatch
# job per seed with explicit flags, recorded under outputs/seeds/<study>/ and summarized in
# outputs/reports/<study>/summary.md after each run.
#
#   bash scripts/submit_factor_study.sh metashift [SEED ...]      (default seeds: 1 2)
#   bash scripts/submit_factor_study.sh spur_cifar10 [SEED ...]
#
# MetaShift, study factors_metashift, 11 arms. Each F2 arm changes one thing from f2_std:
#   simclr, f2_std            baseline and F2 (weight 1, merge 0.9, standardized targets)
#   f2_shuffled               control: targets of other images
#   f2_w3, f2_w10             loss weight 3 and 10
#   f2_presence               targets are presence indicators
#   f2_coarse                 fewer, frequent factors: merge 0.7 and minimum frequency 0.05
#   f2_laion                  LAION dictionary in place of Open Images
#   f2_start0                 F2 from the first epoch, without warm-up
#   simclr_lt, f2_std_lt      the SSL hyperparameters LateTVG used on MetaShift
#                             (learning rate 0.05, batch 256, weight decay 1e-3)
# Spur-CIFAR10, study factors_spur_cifar10: simclr, f2_std and f2_shuffled.
set -euo pipefail

DATASET="${1:?Usage: $0 metashift|spur_cifar10 [SEED ...]}"
shift
SEEDS=("$@")
if [[ "${#SEEDS[@]}" == "0" ]]; then
  SEEDS=(1 2)
fi
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p outputs/SLURM
source scripts/load_splice_cluster_env.sh

STUDY="factors_${DATASET}"
COMMON=(--preset matched --dataset "${DATASET}" --study "${STUDY}" --wandb_group "${STUDY}")
F2=(--splice_mode concept_factors --factor_distill_weight 1.0 --factor_merge_similarity 0.9)
declare -A ARMS
if [[ "${DATASET}" == "metashift" ]]; then
  CACHE_ROOT="${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/metashift/splice_dataset_cache"
  LAION_CACHE="${CACHE_ROOT}/cache_v1__model_open_clip_ViT-B-32__pretrained_laion2b_s34b_b79k__vocab_laion_10000__l1_0p25/splice_dataset_cache.pt"
  LAION_GROUPS="outputs/shared/metashift/graphs/concept_groups_laion/text_0p8_coactivation_0p3/concept_groups.json"
  LT=(--learning_rate 0.05 --batch_size 256 --weight_decay 0.001)
  ARMS=(
    [simclr]="--splice_mode none"
    [f2_std]="${F2[*]} --factor_targets standardized"
    [f2_shuffled]="${F2[*]} --factor_targets shuffled"
    [f2_w3]="--splice_mode concept_factors --factor_distill_weight 3.0 --factor_merge_similarity 0.9 --factor_targets standardized"
    [f2_w10]="--splice_mode concept_factors --factor_distill_weight 10.0 --factor_merge_similarity 0.9 --factor_targets standardized"
    [f2_presence]="${F2[*]} --factor_targets presence"
    [f2_coarse]="--splice_mode concept_factors --factor_distill_weight 1.0 --factor_merge_similarity 0.7 --factor_min_frequency 0.05 --factor_targets standardized"
    [f2_laion]="${F2[*]} --factor_targets standardized --factor_concept_groups ${LAION_GROUPS} --factor_splice_cache ${LAION_CACHE}"
    [f2_start0]="${F2[*]} --factor_targets standardized --factor_start_epoch 0 --factor_warmup_epochs 0"
    [simclr_lt]="--splice_mode none ${LT[*]}"
    [f2_std_lt]="${F2[*]} --factor_targets standardized ${LT[*]}"
  )
elif [[ "${DATASET}" == "spur_cifar10" ]]; then
  ARMS=(
    [simclr]="--splice_mode none"
    [f2_std]="${F2[*]} --factor_targets standardized"
    [f2_shuffled]="${F2[*]} --factor_targets shuffled"
  )
else
  echo "Unknown dataset ${DATASET}; use metashift or spur_cifar10." >&2
  exit 2
fi

# The LAION arm needs LAION concept groups at the default thresholds; build them first when missing.
GROUP_DEPENDENCY=()
if [[ -n "${LAION_GROUPS:-}" && ! -f "${LAION_GROUPS}" ]]; then
  GROUP_JOB=$(sbatch --parsable scripts/generate_cospro_concept_groups.sh "${LAION_CACHE}" \
    --output-root outputs/shared/metashift/graphs/concept_groups_laion \
    --text-similarity-threshold 0.80 --coactivation-threshold 0.30 --no-embed-images)
  GROUP_DEPENDENCY=(--dependency="afterok:${GROUP_JOB%%;*}")
  echo "laion_groups_job=${GROUP_JOB%%;*}"
fi

for arm in $(printf '%s\n' "${!ARMS[@]}" | sort); do
  for seed in "${SEEDS[@]}"; do
    dependency=()
    if [[ "${arm}" == "f2_laion" ]]; then
      dependency=(${GROUP_DEPENDENCY[@]+"${GROUP_DEPENDENCY[@]}"})
    fi
    # shellcheck disable=SC2206
    arm_args=(${ARMS[${arm}]})
    job=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --job-name "${STUDY}-${arm}-s${seed}" \
      scripts/run_training.sbatch "${COMMON[@]}" --seed "${seed}" --arm "${arm}" "${arm_args[@]}")
    echo "${arm} seed=${seed} job=${job%%;*}"
  done
done
echo "Summary after each run: outputs/reports/${STUDY}/summary.md"
