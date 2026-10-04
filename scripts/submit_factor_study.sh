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
#   xfit_meaning_w3, xfit_meaning_w10, xfit_meaning_w3_shuffled  loss weight 3 and 10: at weight 1 the
#                             held-out explained variance was still rising at epoch 500
# Sixth round, generalization of the learned factors: the *_holdout arms keep a fifth of the training
# images out of the concept loss, and factor_learnability_epoch_<E>.json scores the factors on them.
#   xfit_meaning_w10_holdout, xfit_meaning_w10_shuffled_holdout   MetaShift
#   xfit_norm_holdout, xfit_norm_shuffled_holdout                 Spur-CIFAR10
# Waterbirds and CelebA also take the cross-fitted F2 frozen on Spur-CIFAR10: xfit_norm, xfit_norm_shuffled
# and their *_holdout variants. Spur-CIFAR10, Waterbirds and CelebA also take xfit_meaning_w10 (Open Images
# meaning groups, text 0.85, response 0.5, factors from 5 percent, weight 10), its shuffled control and, on
# Waterbirds and CelebA, their *_holdout variants.
# Seventh round, the loss weight on one factor set per dataset: xfit_meaning_w{1,3,5,10}_holdout. The weight
# that maximizes the explained variance on unseen images (no labels) is the candidate rule for choosing it.
# Eighth round, residual targets: each factor minus its regression on the others, the part a fused direction
# cannot predict (xfit_meaning_w3_residual_holdout on MetaShift, xfit_meaning_w10_residual_holdout on
# Waterbirds and CelebA, with residual_shuffled controls).
# Ninth round, the CLIP control: as many principal components of the CLIP image embeddings as there are
# factors, in place of the factors (xfit_meaning_w5_clippca_holdout; xfit_norm_clippca on Spur-CIFAR10).
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
# A fifth of the training images stays out of the concept loss; the learnability record scores them.
HOLDOUT=(--factor_holdout_fraction 0.2)
OPENIMAGES_CACHE_NAME="cache_v1__model_open_clip_ViT-B-32__pretrained_laion2b_s34b_b79k__vocab_openimages_v7_all__l1_0p25"
# Cross-fitted F2 at weight 10 on Open Images meaning groups at the thresholds frozen on MetaShift (text 0.85,
# response 0.5), factors from 5 percent. Spur-CIFAR10, Waterbirds and CelebA; the groups are built when missing.
MEANING_OI_GROUPS="outputs/shared/${DATASET}/graphs/concept_groups_meaning/openimages_v7_text_0p85_response_0p50/concept_groups.json"
MEANING_OI_CACHE="${SPUR_SPLICE_SCRATCH_ROOT}/features/Spur_SpLiCE/${DATASET}/splice_dataset_cache/${OPENIMAGES_CACHE_NAME}/splice_dataset_cache.pt"
XFIT_MEANING_OI=(--splice_mode concept_factors --factor_distill_weight 10.0 --factor_cross_fit true --factor_ridge 1.0
  --factor_concept_groups "${MEANING_OI_GROUPS}" --factor_splice_cache "${MEANING_OI_CACHE}"
  --factor_min_frequency 0.05 --factor_merge_similarity 0)
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
    [xfit_meaning_w3]="${XFIT_MEANING[*]} --factor_targets standardized --factor_distill_weight 3.0"
    [xfit_meaning_w10]="${XFIT_MEANING[*]} --factor_targets standardized --factor_distill_weight 10.0"
    [xfit_meaning_w3_shuffled]="${XFIT_MEANING[*]} --factor_targets shuffled --factor_distill_weight 3.0"
    [xfit_meaning_w10_holdout]="${XFIT_MEANING[*]} --factor_targets standardized --factor_distill_weight 10.0 ${HOLDOUT[*]}"
    [xfit_meaning_w10_shuffled_holdout]="${XFIT_MEANING[*]} --factor_targets shuffled --factor_distill_weight 10.0 ${HOLDOUT[*]}"
    [xfit_meaning_w1_holdout]="${XFIT_MEANING[*]} --factor_targets standardized --factor_distill_weight 1.0 ${HOLDOUT[*]}"
    [xfit_meaning_w3_holdout]="${XFIT_MEANING[*]} --factor_targets standardized --factor_distill_weight 3.0 ${HOLDOUT[*]}"
    [xfit_meaning_w5_holdout]="${XFIT_MEANING[*]} --factor_targets standardized --factor_distill_weight 5.0 ${HOLDOUT[*]}"
    [xfit_meaning_w3_residual_holdout]="${XFIT_MEANING[*]} --factor_targets residual --factor_distill_weight 3.0 ${HOLDOUT[*]}"
    [xfit_meaning_w3_residual_shuffled_holdout]="${XFIT_MEANING[*]} --factor_targets residual_shuffled --factor_distill_weight 3.0 ${HOLDOUT[*]}"
    [xfit_meaning_w5_clippca_holdout]="${XFIT_MEANING[*]} --factor_targets clip_pca --factor_distill_weight 5.0 ${HOLDOUT[*]}"
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
    [xfit_meaning_w10]="${XFIT_MEANING_OI[*]} --factor_targets standardized"
    [xfit_meaning_w10_shuffled]="${XFIT_MEANING_OI[*]} --factor_targets shuffled"
    [xfit_meaning_w1_holdout]="${XFIT_MEANING_OI[*]} --factor_targets standardized --factor_distill_weight 1.0 ${HOLDOUT[*]}"
    [xfit_meaning_w3_holdout]="${XFIT_MEANING_OI[*]} --factor_targets standardized --factor_distill_weight 3.0 ${HOLDOUT[*]}"
    [xfit_meaning_w5_holdout]="${XFIT_MEANING_OI[*]} --factor_targets standardized --factor_distill_weight 5.0 ${HOLDOUT[*]}"
    [xfit_meaning_w10_holdout]="${XFIT_MEANING_OI[*]} --factor_targets standardized ${HOLDOUT[*]}"
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
    [xfit_norm_holdout]="${XFIT_NORM[*]} --factor_targets standardized ${HOLDOUT[*]}"
    [xfit_norm_shuffled_holdout]="${XFIT_NORM[*]} --factor_targets shuffled ${HOLDOUT[*]}"
    [xfit_norm_clippca]="${XFIT_NORM[*]} --factor_targets clip_pca"
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
    [xfit_norm]="${XFIT_NORM[*]} --factor_targets standardized"
    [xfit_norm_shuffled]="${XFIT_NORM[*]} --factor_targets shuffled"
    [xfit_norm_holdout]="${XFIT_NORM[*]} --factor_targets standardized ${HOLDOUT[*]}"
    [xfit_norm_shuffled_holdout]="${XFIT_NORM[*]} --factor_targets shuffled ${HOLDOUT[*]}"
    [xfit_meaning_w10]="${XFIT_MEANING_OI[*]} --factor_targets standardized"
    [xfit_meaning_w10_shuffled]="${XFIT_MEANING_OI[*]} --factor_targets shuffled"
    [xfit_meaning_w10_holdout]="${XFIT_MEANING_OI[*]} --factor_targets standardized ${HOLDOUT[*]}"
    [xfit_meaning_w10_shuffled_holdout]="${XFIT_MEANING_OI[*]} --factor_targets shuffled ${HOLDOUT[*]}"
    [xfit_meaning_w1_holdout]="${XFIT_MEANING_OI[*]} --factor_targets standardized --factor_distill_weight 1.0 ${HOLDOUT[*]}"
    [xfit_meaning_w3_holdout]="${XFIT_MEANING_OI[*]} --factor_targets standardized --factor_distill_weight 3.0 ${HOLDOUT[*]}"
    [xfit_meaning_w5_holdout]="${XFIT_MEANING_OI[*]} --factor_targets standardized --factor_distill_weight 5.0 ${HOLDOUT[*]}"
    [xfit_meaning_w10_residual_holdout]="${XFIT_MEANING_OI[*]} --factor_targets residual ${HOLDOUT[*]}"
    [xfit_meaning_w10_residual_shuffled_holdout]="${XFIT_MEANING_OI[*]} --factor_targets residual_shuffled ${HOLDOUT[*]}"
    [xfit_meaning_w5_clippca_holdout]="${XFIT_MEANING_OI[*]} --factor_targets clip_pca --factor_distill_weight 5.0 ${HOLDOUT[*]}"
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

# Open Images meaning groups of Spur-CIFAR10, Waterbirds and CelebA; build them first when missing.
MEANING_GROUP_DEPENDENCY=()
if [[ "${DATASET}" != "metashift" && " ${SELECTED[*]} " == *meaning* && ! -f "${MEANING_OI_GROUPS}" ]]; then
  MEANING_GROUP_JOB=$(sbatch --parsable scripts/build_meaning_groups.sbatch --dataset "${DATASET}" --vocab openimages_v7)
  MEANING_GROUP_DEPENDENCY=(--dependency="afterok:${MEANING_GROUP_JOB%%;*}")
  echo "meaning_groups_job=${MEANING_GROUP_JOB%%;*}"
fi

# At most MAX_PARALLEL training jobs of this submission run at once: job i waits for job i - MAX_PARALLEL
# to end (afterany), so the jobs run in MAX_PARALLEL lanes one after another.
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SUBMITTED=()
for arm in "${SELECTED[@]}"; do
  for seed in "${SEEDS[@]}"; do
    conditions=()
    if [[ "${arm}" == *meaning* && "${#MEANING_GROUP_DEPENDENCY[@]}" -gt 0 ]]; then
      conditions+=("${MEANING_GROUP_DEPENDENCY[0]#--dependency=}")
    elif [[ "${arm}" == "f2_laion" && "${#GROUP_DEPENDENCY[@]}" -gt 0 ]]; then
      conditions+=("${GROUP_DEPENDENCY[0]#--dependency=}")
    elif [[ "${arm}" != "simclr" && "${#DEFAULT_GROUP_DEPENDENCY[@]}" -gt 0 ]]; then
      conditions+=("${DEFAULT_GROUP_DEPENDENCY[0]#--dependency=}")
    fi
    if (( ${#SUBMITTED[@]} >= MAX_PARALLEL )); then
      conditions+=("afterany:${SUBMITTED[${#SUBMITTED[@]} - MAX_PARALLEL]}")
    fi
    dependency=()
    if [[ "${#conditions[@]}" -gt 0 ]]; then
      dependency=(--dependency="$(IFS=,; echo "${conditions[*]}")")
    fi
    # shellcheck disable=SC2206
    arm_args=(${ARMS[${arm}]})
    job=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --job-name "${STUDY}-${arm}-s${seed}" \
      scripts/run_training.sbatch "${COMMON[@]}" --seed "${seed}" --arm "${arm}" "${arm_args[@]}")
    SUBMITTED+=("${job%%;*}")
    echo "${arm} seed=${seed} job=${job%%;*}${dependency[0]:+ (${dependency[0]})}"
  done
done
echo "Summary after each run: outputs/reports/${STUDY}/summary.md"
