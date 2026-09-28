#!/usr/bin/env bash
# Submit the concept-factor (F2) study of one dataset: every arm is one scripts/run_training.sbatch
# job per seed with explicit flags, recorded under outputs/seeds/<study>/ and summarized in
# outputs/reports/<study>/summary.md after each run.
#
#   bash scripts/submit_factor_study.sh metashift [SEED ...]      (default seeds: 1 2)
#   bash scripts/submit_factor_study.sh spur_cifar10 [SEED ...]
#   bash scripts/submit_factor_study.sh waterbirds|celeba [SEED ...]   (held-out datasets)
#   ARMS="simclr f2_w3" bash scripts/submit_factor_study.sh metashift 3 4   (only these arms)
#   SBATCH_PARTITION=gpuidle bash scripts/submit_factor_study.sh ...   (another partition: sbatch reads
#   SBATCH_PARTITION and it overrides the #SBATCH --partition line of every launcher)
#
# MetaShift, study factors_metashift. Each F2 arm changes one thing from f2_std:
#   simclr, f2_std            baseline and F2 (weight 1, merge 0.9, standardized targets)
#   f2_shuffled               control: targets of other images
#   f2_w3, f2_w10             loss weight 3 and 10
#   f2_presence               targets are presence indicators
#   f2_coarse                 fewer, frequent factors: merge 0.7 and minimum frequency 0.05
#   f2_laion                  LAION dictionary in place of Open Images
#   f2_start0                 F2 from the first epoch, without warm-up
#   simclr_lt, f2_std_lt      the SSL hyperparameters LateTVG used on MetaShift
#                             (learning rate 0.05, batch 256, weight decay 1e-3)
# Second round, around the best first-round arm f2_w3 (weight 3):
#   f2_shuffled_w3            the control at weight 3
#   f2_white_w3, f2_white_w10 whitened targets, which weight 1 never learned
#   f2_w3_atyp                standardized targets, images weighted by concept atypicality
#   f2_white_w3_atyp          whitened targets and atypicality weights
# Third round, making the co-occurrence shortcut useless or wrong:
#   f2_w3_balanced            F2 weight 3 with weights under which the factors are uncorrelated
#   cbc                       F3 concept blocks (weight 1, cross-context pairs weigh more)
#   cbc_plain                 control: all positive pairs weigh alike
#   cbc_shuffled              control: each image carries the concept set of another image
#   cbc_f2_balanced           F3 together with f2_w3_balanced
# Fourth round, cross-fitted F2 (a ridge fitted on one half of the batch predicts the other half, so
# memorizing images cannot lower the loss):
#   xfit                      cross-fitted F2, weight 3, standardized targets
#   xfit_balanced             with balancing weights in the fit and the loss
#   xfit_shuffled             control: targets of other images, which cross-fitting cannot learn
#   Their ridge was relative and small, so fits on 64 images in 512 dimensions were noisy and the
#   solve diverged on Spur-CIFAR10. The xfit_norm arms use unit-norm features and an absolute ridge:
#   xfit_norm                 cross-fitted F2, weight 1, ridge 1
#   xfit_norm_r10             ridge 10
#   xfit_norm_w3              weight 3
#   xfit_norm_shuffled        control
#   xfit_norm_balanced        with balancing weights (MetaShift only)
# Fifth round, meaning groups (MetaShift only): LAION concepts joined by average linkage on raw CLIP
# text embeddings (cospro.cli.build_meaning_groups, text 0.85, response 0.5), factors from 5 percent,
# no merge. The groups file is built locally and synchronized through Git.
#   f2_meaning, xfit_meaning  F2 and cross-fitted F2 (weight 1, ridge 1) on these factors
#   f2_meaning_shuffled, xfit_meaning_shuffled  their controls
# Spur-CIFAR10, study factors_spur_cifar10: simclr, f2_std, f2_shuffled, f2_w3, f2_w3_atyp,
# f2_w3_balanced, cbc, xfit, xfit_shuffled and the xfit_norm arms.
# Waterbirds and CelebA, studies factors_waterbirds and factors_celeba: the frozen F2 of the
# development datasets (f2_std: weight 1, merge 0.9, standardized targets) against simclr and the
# f2_shuffled control. CelebA trains 250 epochs, as its CoSpRo pipeline did. Their concept groups at
# the default thresholds are built first when missing.
set -euo pipefail

DATASET="${1:?Usage: $0 metashift|spur_cifar10|waterbirds|celeba [SEED ...]}"
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
F2W3=(--splice_mode concept_factors --factor_distill_weight 3.0 --factor_merge_similarity 0.9)
CBC=(--splice_mode concept_factors --factor_block_weight 1.0 --factor_merge_similarity 0.9)
XFIT=(--splice_mode concept_factors --factor_distill_weight 3.0 --factor_merge_similarity 0.9 --factor_cross_fit true)
XFIT_NORM=(--splice_mode concept_factors --factor_distill_weight 1.0 --factor_merge_similarity 0.9 --factor_cross_fit true --factor_ridge 1.0)
OPENIMAGES_CACHE_NAME="cache_v1__model_open_clip_ViT-B-32__pretrained_laion2b_s34b_b79k__vocab_openimages_v7_all__l1_0p25"
DEFAULT_GROUPS="outputs/shared/${DATASET}/graphs/concept_groups/text_0p8_coactivation_0p3/concept_groups.json"
GROUPS_CACHE=""
# ARMS in the environment selects arms by name; the table below then reuses the name.
SELECTED_ARMS="${ARMS:-}"
unset ARMS
declare -A ARMS
if [[ "${DATASET}" == "metashift" ]]; then
  CACHE_ROOT="${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/metashift/splice_dataset_cache"
  LAION_CACHE="${CACHE_ROOT}/cache_v1__model_open_clip_ViT-B-32__pretrained_laion2b_s34b_b79k__vocab_laion_10000__l1_0p25/splice_dataset_cache.pt"
  LAION_GROUPS="outputs/shared/metashift/graphs/concept_groups_laion/text_0p8_coactivation_0p3/concept_groups.json"
  LT=(--learning_rate 0.05 --batch_size 256 --weight_decay 0.001)
  # Meaning groups (cospro.cli.build_meaning_groups): factors of several synonyms, band from 5 percent:
  # at 1 percent most factors occur less than once per half batch, so the cross-fit cannot predict them.
  MEANING_GROUPS="outputs/shared/metashift/graphs/concept_groups_meaning/laion_text_0p85_response_0p50/concept_groups.json"
  MEANING=(--factor_concept_groups "${MEANING_GROUPS}" --factor_splice_cache "${LAION_CACHE}" --factor_min_frequency 0.05 --factor_merge_similarity 0)
  F2_MEANING=(--splice_mode concept_factors --factor_distill_weight 1.0 "${MEANING[@]}")
  XFIT_MEANING=(--splice_mode concept_factors --factor_distill_weight 1.0 --factor_cross_fit true --factor_ridge 1.0 "${MEANING[@]}")
  ARMS=(
    [f2_meaning]="${F2_MEANING[*]} --factor_targets standardized"
    [f2_meaning_shuffled]="${F2_MEANING[*]} --factor_targets shuffled"
    [xfit_meaning]="${XFIT_MEANING[*]} --factor_targets standardized"
    [xfit_meaning_shuffled]="${XFIT_MEANING[*]} --factor_targets shuffled"
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
    [f2_shuffled_w3]="${F2W3[*]} --factor_targets shuffled"
    [f2_white_w3]="${F2W3[*]} --factor_targets whitened"
    [f2_white_w10]="--splice_mode concept_factors --factor_distill_weight 10.0 --factor_merge_similarity 0.9 --factor_targets whitened"
    [f2_w3_atyp]="${F2W3[*]} --factor_targets standardized --factor_sample_weighting atypicality"
    [f2_white_w3_atyp]="${F2W3[*]} --factor_targets whitened --factor_sample_weighting atypicality"
    [f2_w3_balanced]="${F2W3[*]} --factor_targets standardized --factor_sample_weighting balanced"
    [cbc]="${CBC[*]}"
    [cbc_plain]="${CBC[*]} --factor_block_context_weight 0"
    [cbc_shuffled]="${CBC[*]} --factor_block_presence shuffled"
    [cbc_f2_balanced]="${CBC[*]} --factor_distill_weight 3.0 --factor_targets standardized --factor_sample_weighting balanced"
    [xfit]="${XFIT[*]} --factor_targets standardized"
    [xfit_balanced]="${XFIT[*]} --factor_targets standardized --factor_sample_weighting balanced"
    [xfit_shuffled]="${XFIT[*]} --factor_targets shuffled"
    [xfit_norm]="${XFIT_NORM[*]} --factor_targets standardized"
    [xfit_norm_r10]="${XFIT_NORM[*]} --factor_targets standardized --factor_ridge 10.0"
    [xfit_norm_w3]="${XFIT_NORM[*]} --factor_targets standardized --factor_distill_weight 3.0"
    [xfit_norm_shuffled]="${XFIT_NORM[*]} --factor_targets shuffled"
    [xfit_norm_balanced]="${XFIT_NORM[*]} --factor_targets standardized --factor_sample_weighting balanced"
  )
elif [[ "${DATASET}" == "spur_cifar10" ]]; then
  ARMS=(
    [simclr]="--splice_mode none"
    [f2_std]="${F2[*]} --factor_targets standardized"
    [f2_shuffled]="${F2[*]} --factor_targets shuffled"
    [f2_w3]="${F2W3[*]} --factor_targets standardized"
    [f2_w3_atyp]="${F2W3[*]} --factor_targets standardized --factor_sample_weighting atypicality"
    [f2_w3_balanced]="${F2W3[*]} --factor_targets standardized --factor_sample_weighting balanced"
    [cbc]="${CBC[*]}"
    [xfit]="${XFIT[*]} --factor_targets standardized"
    [xfit_shuffled]="${XFIT[*]} --factor_targets shuffled"
    [xfit_norm]="${XFIT_NORM[*]} --factor_targets standardized"
    [xfit_norm_r10]="${XFIT_NORM[*]} --factor_targets standardized --factor_ridge 10.0"
    [xfit_norm_w3]="${XFIT_NORM[*]} --factor_targets standardized --factor_distill_weight 3.0"
    [xfit_norm_shuffled]="${XFIT_NORM[*]} --factor_targets shuffled"
  )
elif [[ "${DATASET}" == "waterbirds" || "${DATASET}" == "celeba" ]]; then
  if [[ "${DATASET}" == "celeba" ]]; then
    COMMON+=(--epochs 250)
  fi
  GROUPS_CACHE="${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/${DATASET}/splice_dataset_cache/${OPENIMAGES_CACHE_NAME}/splice_dataset_cache.pt"
  ARMS=(
    [simclr]="--splice_mode none"
    [f2_std]="${F2[*]} --factor_targets standardized"
    [f2_shuffled]="${F2[*]} --factor_targets shuffled"
  )
else
  echo "Unknown dataset ${DATASET}; use metashift, spur_cifar10, waterbirds or celeba." >&2
  exit 2
fi

SELECTED=()
if [[ -n "${SELECTED_ARMS}" ]]; then
  for arm in ${SELECTED_ARMS}; do
    if [[ -z "${ARMS[${arm}]+set}" ]]; then
      echo "Unknown arm ${arm} for ${DATASET}; known: $(printf '%s ' "${!ARMS[@]}")" >&2
      exit 2
    fi
    SELECTED+=("${arm}")
  done
else
  mapfile -t SELECTED < <(printf '%s\n' "${!ARMS[@]}" | sort)
fi

# The LAION arm needs LAION concept groups at the default thresholds; build them first when missing.
GROUP_DEPENDENCY=()
if [[ " ${SELECTED[*]} " == *" f2_laion "* && -n "${LAION_GROUPS:-}" && ! -f "${LAION_GROUPS}" ]]; then
  GROUP_JOB=$(sbatch --parsable scripts/generate_cospro_concept_groups.sh "${LAION_CACHE}" \
    --output-root outputs/shared/metashift/graphs/concept_groups_laion \
    --text-similarity-threshold 0.80 --coactivation-threshold 0.30 --no-embed-images)
  GROUP_DEPENDENCY=(--dependency="afterok:${GROUP_JOB%%;*}")
  echo "laion_groups_job=${GROUP_JOB%%;*}"
fi

# Held-out datasets have concept groups only at older thresholds; build the default ones first.
DEFAULT_GROUP_DEPENDENCY=()
if [[ -n "${GROUPS_CACHE}" && ! -f "${DEFAULT_GROUPS}" ]]; then
  if [[ ! -f "${GROUPS_CACHE}" ]]; then
    echo "SpLiCE cache not found: ${GROUPS_CACHE}; build it with scripts/cache_splice_dataset.sh first." >&2
    exit 2
  fi
  DEFAULT_GROUP_JOB=$(sbatch --parsable scripts/generate_cospro_concept_groups.sh "${GROUPS_CACHE}" \
    --output-root "outputs/shared/${DATASET}/graphs/concept_groups" \
    --text-similarity-threshold 0.80 --coactivation-threshold 0.30 --no-embed-images)
  DEFAULT_GROUP_DEPENDENCY=(--dependency="afterok:${DEFAULT_GROUP_JOB%%;*}")
  echo "default_groups_job=${DEFAULT_GROUP_JOB%%;*}"
fi

for arm in "${SELECTED[@]}"; do
  for seed in "${SEEDS[@]}"; do
    dependency=()
    if [[ "${arm}" == "f2_laion" ]]; then
      dependency=(${GROUP_DEPENDENCY[@]+"${GROUP_DEPENDENCY[@]}"})
    elif [[ "${arm}" != "simclr" ]]; then
      dependency=(${DEFAULT_GROUP_DEPENDENCY[@]+"${DEFAULT_GROUP_DEPENDENCY[@]}"})
    fi
    # shellcheck disable=SC2206
    arm_args=(${ARMS[${arm}]})
    job=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --job-name "${STUDY}-${arm}-s${seed}" \
      scripts/run_training.sbatch "${COMMON[@]}" --seed "${seed}" --arm "${arm}" "${arm_args[@]}")
    echo "${arm} seed=${seed} job=${job%%;*}"
  done
done
echo "Summary after each run: outputs/reports/${STUDY}/summary.md"
