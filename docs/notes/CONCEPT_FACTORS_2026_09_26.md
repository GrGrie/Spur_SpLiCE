# Concept factors: keeping entangled concepts apart in SSL

2026-09-26, branch `concept-factors`. This note states the idea, the two mechanisms, their options
and the first experiment on Spur-CIFAR10 and MetaShift.

## Idea

Label-free methods against spurious correlations need a prior on what is spurious. LateTVG and LA-SSL
assume the spurious feature is the easy one; the CoSpRo audit gate assumes it names a context. The
concept-factor methods avoid that choice.

The probe of this project is trained on group-balanced data (`ds_train`). Deep feature reweighting
(Kirichenko et al., 2023) shows that a balanced linear head recovers worst-group accuracy whenever
the representation encodes the core feature separately from the spurious one. The SSL failure to
prevent is therefore the fusion of two correlated features into one direction, or the suppression
of the harder one by the easier one (Chen et al., 2021). Keeping every strongly correlated pair of
factors apart serves both sides of the pair, and the balanced probe picks the side the task needs.

SpLiCE supplies the two ingredients without labels:

* **Factors and entangled pairs.** A factor is one concept group of the grouping stage; its value on
  an image is the summed SpLiCE code of the group's concepts. Pairs of factors whose presence is
  strongly correlated across the training images (phi coefficient), and whose words name different
  things (text similarity), are the entangled pairs. In Waterbirds these are pairs such as
  `water` and `gull`.
* **Decorrelated targets.** ZCA whitening of the factor activations gives each factor its part the
  other factors leave unexplained. These targets are large exactly on the images that break a
  correlation, the label-free counterpart of the minority groups.

## F1: concept-conditioned batches

With probability `--factor_condition_fraction`, a batch is filled from the unused images that show
one factor of an entangled pair; the other batches are random. Inside a conditioned batch the factor
distinguishes no image from another, so InfoNCE separates the images by everything else, including
the partner factor. This is conditional contrastive learning (Tsai et al., 2021) with the discovered
factor in place of an annotated attribute. Every image still occurs once per epoch.

Diagnostic: `method/factor_conditioned_batch_fraction`, the realized share of conditioned batches.

## F2: decorrelated factor distillation

A linear head on the backbone predicts the whitened factor targets of both views under a mean
squared error, weighted by `--factor_distill_weight` after `--factor_start_epoch` and a linear
warm-up. The head is linear so the factors must be linearly readable from the backbone, as the probe
reads it. `--factor_targets standardized` keeps the correlations in the targets and is the ablation
that isolates the effect of whitening.

Diagnostics: `method/factor_mse` and `method/factor_explained_variance` (targets have unit variance,
so this is 1 minus the MSE).

## Options

| Option | Default | Meaning |
|---|---|---|
| `--splice_mode concept_factors` | | selects the method |
| `--factor_condition_fraction` | 0 | F1 share of conditioned batches; 0 disables F1 |
| `--factor_distill_weight` | 0 | F2 loss weight; 0 disables F2 |
| `--factor_targets` | whitened | F2 targets, `whitened` or `standardized` |
| `--factor_min_frequency`, `--factor_max_frequency` | 0.02, 0.9 | frequency band of a factor |
| `--factor_max_count` | 0 | most concept groups, the most balanced first; 0 keeps all in the band |
| `--factor_merge_similarity` | 0 | merge groups whose image alignment correlates this much; 0 disables |
| `--factor_condition_pairs` | 8 | entangled pairs whose factors condition batches |
| `--factor_min_correlation` | 0.2 | least phi of a pair |
| `--factor_max_text_similarity` | 0.75 | largest text similarity of a pair |
| `--factor_whitening_eps` | 0.1 | ridge of the whitening |
| `--factor_start_epoch`, `--factor_warmup_epochs` | 10, 10 | F2 schedule |
| `--factor_concept_groups`, `--factor_splice_cache` | found automatically | inputs |

The inputs are the concept groups under `outputs/shared/<dataset>/graphs/concept_groups/` and the
SpLiCE cache they were built from under `<scratch>/features/Spur_SpLiCE/<dataset>/`. Both exist for
Spur-CIFAR10 and MetaShift from the CoSpRo pipeline runs. F1 and F2 combine with each other and with
`--latetvg_prune_rate`.

## First experiment

Inspect the factors of each dataset before training. The report lands in
`outputs/shared/<dataset>/factors/concept_factors_report.json`:

```bash
sbatch scripts/inspect_concept_factors.sbatch --dataset spur_cifar10
sbatch scripts/inspect_concept_factors.sbatch --dataset metashift
```

Spur-CIFAR10 is the sharper test of the discovery step: its spurious feature is a thin coloured line,
and the report shows whether CLIP names that colour at all.

Arms per dataset, seeds 1 and 2, all through `scripts/run_training.sbatch --preset matched`:
SimCLR, F1, F2, F2 with standardized targets and F1+F2. That is 10 runs per dataset.

## Revision after the first inspection (2026-09-26)

The first factor reports showed two failures. Taking the 64 most frequent groups dropped the scene
concepts of MetaShift (Bed, Couch, Lawn, Window) and most line colours of Spur-CIFAR10 (magenta, lime,
teal, violet). Ranking pairs by phi then filled the top eight with two names for one visual content:
cat breeds, lorry and truck, delta and aircraft. Co-occurrence alone cannot tell a synonym from a
correlated but distinct concept, and text similarity misses brands and polysemy.

The revision keeps every group in the frequency band (`--factor_max_count 0`) and adds redundancy
merging (`--factor_merge_similarity m`): two groups join one factor when the dataset's images align
with their directions together, that is when the correlation of `E d_A` and `E d_B` over the centered
CLIP image embeddings `E` reaches `m`. Entangled pairs are then searched between the merged factors.

## Evaluating a factor set

`cospro.diagnostics.factor_validity` types every factor against the hidden labels, for evaluation
only. Because `y` and `a` are correlated in the training images, it scores conditional information:
`u_attribute = I(F; a | y) / H(F)` and `u_class = I(F; y | a) / H(F)`. A factor is `attribute` or
`class` when its score reaches 0.05 and doubles the other one. A pair is `cross` when it joins a class
factor to an attribute factor, and pair precision is the share of cross pairs. Two numbers matter:

* **pair precision** decides whether F1 conditions on the right factors;
* **attribute signal**, the largest `u_attribute`, and the number of attribute factors decide whether
  F2 has the spurious attribute among its targets at all.

```bash
sbatch scripts/inspect_concept_factors.sbatch --dataset metashift --diagnose --merge-similarity 0.8
```

Reading a report by eye: a good pair names two different parts of the image (animal and background,
object and line colour); a bad pair names one part twice (breeds, brands, synonyms, or two words that
each already fuse object and context such as "Disc dog" and "Lure coursing").

Settings are chosen on MetaShift and Spur-CIFAR10 with this diagnostic and then frozen; Waterbirds and
CelebA stay untouched for the final evaluation, so the label-free claim holds for them.

## Grouping and merging sweep

```bash
sbatch scripts/sweep_concept_factors.sbatch
```

For MetaShift and Spur-CIFAR10 under the LAION (10k) and Open Images dictionaries it builds the missing
SpLiCE caches, groups the concepts at text similarity 0.60 to 0.90 in steps of 0.05 and co-activation
0.20, 0.30 and 0.40, builds factors at merge thresholds off, 0.70, 0.80 and 0.90 and scores each set.
The cache does not depend on the grouping thresholds; only the groups do. The summary lands in
`outputs/reports/concept_factor_sweep/summary.md`, the groups and factor sets under
`outputs/shared/<dataset>/factor_sweep/`.

## F2 results of 2026-09-27 and the next study

Validation WGA at SSL epoch 500, seeds 1 and 2, groups at text 0.80 and co-activation 0.30, merge 0.9:

| Dataset | SimCLR | F2 whitened | F2 standardized |
|---|---|---|---|
| Spur-CIFAR10 | 15.6 | 37.0 | 36.4 |
| MetaShift | 47.2 | 49.3 | 42.4 |

On Spur-CIFAR10 F2 lifts WGA by about 21 points and average accuracy by about 10, with either target
kind: the gain comes from distilling the dataset's concept factors, and whitening adds nothing there.
On MetaShift the student explains only 9 to 13 percent of the factor variance and the arms differ by
less than the noise of 72-image validation groups. LateTVG reports a SimSiam MetaShift baseline of
55.8 average accuracy, as low as ours, and lifts it to 70.1, so MetaShift can reward a method; F2 in
its first configuration does not.

`scripts/submit_factor_study.sh` runs the next study. On MetaShift it varies one F2 setting at a time
(loss weight, target kind, coarser factors, the LAION dictionary, the start epoch) and tries the SSL
hyperparameters LateTVG used, next to the `shuffled` control: the same target statistics, taken from
other images. On Spur-CIFAR10 it adds the `shuffled` control to SimCLR and F2.

```bash
bash scripts/submit_factor_study.sh metashift
bash scripts/submit_factor_study.sh spur_cifar10
```

Each run records itself under `outputs/seeds/factors_<dataset>/seed_<NN>/<arm>/<job id>/` and refreshes
`outputs/reports/factors_<dataset>/summary.md`, whose headline column averages the last four
validation probes.

## MetaShift round 1 (2026-09-27) and round 2

Validation, seeds 1 and 2, mean of the last four probes (the full table is
`outputs/reports/results/metashift/concept_factors.md`):

| arm | val WGA | val acc | factor expl. var. |
|---|---|---|---|
| simclr | 47.7 ± 4.2 | 55.8 | |
| f2_std (weight 1) | 43.2 ± 1.2 | 58.2 | 0.12 |
| f2_shuffled (control) | 42.2 ± 4.7 | 51.0 | 0.06 |
| f2_w3 | 51.4 ± 1.5 | 63.1 | 0.41 |
| f2_w10 | 47.0 ± 7.1 | 59.7 | 0.85 |
| f2_coarse (7 factors) | 43.4 ± 2.5 | 60.6 | 0.89 |

Three readings. First, concept content matters: every F2 arm with real targets raises average
accuracy over SimCLR (58 to 63 against 55.8) and the shuffled control lowers it (51.0). Second, WGA on
MetaShift is too noisy for two seeds: SimCLR over ten seeds of `metashift_cospro` scores 44.6 ± 6.9.
Third, the gain goes to the majority groups. Standardized targets reproduce the dataset's
correlations, and several Open Images concepts fuse class and context ("Cat bed", "Dog walking"), so
F2 teaches the student both together; the minority groups stay near 45 to 55 while cat-indoor and
dog-outdoor climb to 70 to 80.

Round 2 therefore tests the two ways of pointing F2 at the minority images without labels, at weight 3
where the targets are learned: whitened targets, which weight 1 never learned (explained variance
0.09), and atypicality weights, which scale each image's F2 loss by how far its concepts break the
dataset's correlations (`--factor_sample_weighting atypicality`; on the synthetic fixture the minority
groups receive 2.1 times the majority weight). `inspect_concept_factors --diagnose` reports the same
ratio on real data.

```bash
ARMS="f2_shuffled_w3 f2_white_w3 f2_white_w10 f2_w3_atyp f2_white_w3_atyp" bash scripts/submit_factor_study.sh metashift 1 2 3 4
ARMS="simclr f2_w3" bash scripts/submit_factor_study.sh metashift 3 4
```

## MetaShift round 2 and the third round

Round 2 (seeds 1 to 4 for the new arms) left the minority groups (cat outdoors, dog indoors) at 47 to
54 percent in every arm, SimCLR included. F2 raised only the majority groups: whitened targets and
atypicality weights narrowed the gap between them by lowering the majority gain. The regression on
the factors learns through the co-occurrence itself: in 88 percent of the training images the indoor
features predict the cat factors, so the student keeps one fused "cat and indoor" direction.

The third round makes that shortcut useless or wrong.

* **F2 with balancing weights** (`--factor_sample_weighting balanced`). Per-image weights, mean 1,
  minimize the mean squared correlation between the factors' presences plus an entropy term that
  keeps them near uniform. Under these weights the cat factors are no longer predictable from the
  indoor factors, so relying on the co-occurrence gains the regression nothing. This is sample
  reweighting for independence as in stable learning (Zhang et al., 2021), applied to concept
  presences.
* **F3, concept blocks** (`--factor_block_weight`). One linear block per factor on the backbone, for
  the 32 most balanced factors. Inside block k a supervised contrastive loss treats images that both
  show factor k as positives and every other image as a negative; a positive pair weighs
  `1 + context_weight * d / mean(d)`, where `d` is how much the two images' full factor sets differ.
  "Cat and sofa" with "cat and street" is a strong positive of the cat block and a negative of the
  sofa and street blocks; "cat and street" with "dog and street" is a positive of the street block
  only. A single fused "cat and indoor" direction places "cat and street" level with "dog and sofa",
  so the cat block cannot separate them: the backbone has to hold a cat direction that survives the
  context, and symmetrically a sofa direction that survives the animal. No factor is declared
  spurious. Pairs that share a factor and differ elsewhere identify that factor (Locatello et al.,
  2020; Yao et al., 2024); the SpLiCE concepts supply such pairs without labels.

Controls: `cbc_plain` (all positive pairs weigh alike) isolates the cross-context weighting and
`cbc_shuffled` (every image carries another image's concept set) isolates the concept content.
`inspect_concept_factors --diagnose` reports how the atypicality and balancing weights split between
minority and majority groups.

```bash
ARMS="f2_w3_balanced cbc cbc_plain cbc_shuffled cbc_f2_balanced" bash scripts/submit_factor_study.sh metashift 1 2 3 4
```

## Third round: memorization, and the cross-fitted F2

The third round lifted no minority group either (47 to 53 percent in every arm, cat outdoors and dog
indoors). Its controls show why. The shuffled controls fit almost as well as the real targets: F2 at
weight 3 explains 0.41 of the real factor variance and 0.33 of the shuffled one, and the block loss
ends at 3.34 with real concepts and 3.37 with shuffled ones. On 1,700 images the student memorizes a
per-image target through the features SimCLR already learns to tell images apart, so any auxiliary
loss attached to single images, real or random, is met without learning concepts.

The cross-fitted F2 (`--factor_cross_fit true`) removes that route. Each half of the batch is
predicted by a ridge regression fitted in closed form on the other half (kernel form, one n-by-n solve,
differentiable, in the manner of R2D2 and MetaOptNet), and the loss is that held-out error, weighted
when `--factor_sample_weighting` is set. A feature that identifies one image carries nothing to the
other half; only concept directions shared across images lower the loss, and with balancing weights a
direction that fuses a concept with its usual context mispredicts the heavily weighted images that
break the co-occurrence. Every F2 run now logs `method/factor_heldout_explained_variance`, whose gap to
the trained-head explained variance measures memorization.

```bash
ARMS="xfit xfit_balanced xfit_shuffled" bash scripts/submit_factor_study.sh metashift 1 2 3 4
ARMS="simclr f2_std f2_shuffled xfit xfit_shuffled" bash scripts/submit_factor_study.sh spur_cifar10 1 2
```

## Spur-CIFAR10 control, cross-fitting repaired and the held-out datasets (2026-09-28)

Spur-CIFAR10, seeds 1 and 2, validation:

| arm | val WGA | val acc | factor expl. var. |
|---|---|---|---|
| simclr | 15.6 ± 1.0 | 59.5 | |
| f2_shuffled | 24.2 ± 0.4 | 58.6 | 0.02 |
| f2_std | 32.9 ± 1.6 | 69.5 | 0.62 |

Real concepts beat the shuffled control by about 9 points of WGA and 11 of accuracy, so the concept
content carries part of the gain; the control itself adds about 9 points of WGA over SimCLR at equal
accuracy, so the auxiliary regression also helps as such. The shuffled targets reach an explained
variance of 0.02 on 45,000 images against 0.33 on MetaShift's 1,700: concept targets shape the
student where images cannot be memorized.

The first cross-fitted runs failed: NaN features on three of four Spur-CIFAR10 runs and negative
held-out explained variance everywhere. The ridge scaled with the feature norm and was small, so fits
of 64 images in 512 dimensions nearly interpolated and the solve became ill conditioned. The fit now
scales every feature row to unit norm and uses an absolute ridge (`--factor_ridge`, default 1), which
keeps the system's eigenvalues at least that large; the `xfit_norm` arms test it.

The held-out datasets get the F2 configuration frozen on the development datasets (`f2_std`) against
SimCLR and the shuffled control, four seeds each; CelebA trains 250 epochs as its CoSpRo pipeline did.

```bash
bash scripts/submit_factor_study.sh waterbirds 1 2 3 4
bash scripts/submit_factor_study.sh celeba 1 2 3 4
ARMS="xfit_norm xfit_norm_r10 xfit_norm_w3 xfit_norm_shuffled xfit_norm_balanced" bash scripts/submit_factor_study.sh metashift 1 2 3 4
ARMS="xfit_norm xfit_norm_shuffled" bash scripts/submit_factor_study.sh spur_cifar10 1 2
```

## Meaning groups and the repaired cross-fit (2026-09-29)

SpLiCE activates one of two synonyms per image and centres its dictionary, so the text-and-co-activation
grouping keeps synonyms apart (heron and egret co-activate at 0.09) and chains a bird to its background
when its thresholds drop. The meaning grouping (`--methods meaning` in the sweep,
`cospro.cli.build_meaning_groups`) clusters concepts by average linkage on raw CLIP text embeddings,
where synonyms reach 0.85 to 0.94 and object-context pairs 0.64 to 0.72.

A ridge fitted on 64 images cannot predict a factor that occurs once per half batch. With CLIP image
embeddings as features, the held-out explained variance of the 278 meaning factors from 1 percent is
-0.015 and of the 24 factors from 5 percent +0.16; shuffled targets give -0.07. These bound what the
student can reach.

Validation, seeds 1 and 2, epoch 500:

| dataset | arm | factors | held-out expl. var. | val WGA | val acc |
|---|---|---|---|---|---|
| Spur-CIFAR10 | simclr | | | 15.6 | 59.5 |
| Spur-CIFAR10 | f2_std | 88 | 0.10 | 32.9 | 69.5 |
| Spur-CIFAR10 | xfit_norm | 88 | 0.13 | 33.6 | 69.7 |
| Spur-CIFAR10 | xfit_shuffled (weight 3) | 88 | -0.02 | 15.3 | 48.8 |
| MetaShift | xfit_norm | 145 | -0.01 | 43.9 | 54.8 |
| MetaShift | xfit_norm_shuffled | 145 | -0.03 | 41.7 | 54.4 |
| MetaShift | xfit_meaning | 24 | 0.05, rising | 46.0 | 58.8 |
| MetaShift | xfit_meaning_shuffled | 24 | -0.02 | 46.7 | 58.6 |

On Spur-CIFAR10 the cross-fitted F2 matches F2 and its shuffled control falls to SimCLR, so the gain
comes from the concept content alone. On MetaShift the meaning factors are learned across images, a
third of the CLIP bound and still rising at epoch 500, and WGA does not move. The next study raises
the loss weight (`xfit_meaning_w3`, `xfit_meaning_w10`) and adds seeds 3 and 4.

`submit_factor_study.sh` runs at most `MAX_PARALLEL` (default 8) of its jobs at once; `submit_experiment.sh`
throttles its array the same way.

## Loss weight on MetaShift and what the student stores (2026-09-30)

Validation, seeds 1 and 2, 24 meaning factors. `train EV` is the five-fold held-out explained variance
of the factors from the training features (`factor_learnability_epoch_500.json`); CLIP features score 0.26.

| arm | train EV | val WGA | val acc | val group acc |
|---|---|---|---|---|
| xfit_meaning_w3_shuffled | 0.02 | 47.6 | 58.1 | 68 / 53 / 50 / 59 |
| xfit_meaning_w3 | 0.33 | 43.2 | 59.6 | 74 / 45 / 48 / 69 |
| xfit_meaning_w10 | 0.79 | 45.5 | 61.0 | 74 / 51 / 48 / 71 |

At weight 10 every factor reaches 0.72 to 0.87 on the training images, three times the CLIP score
and alike for factors CLIP explains well ("cat", 0.68) and badly ("helper", 0.14). A probe on the 24
factor values themselves reaches 77 percent WGA and 89 percent accuracy, and the student stays at 45
and 61 on validation. The encoder therefore stores the factor values of the 1,700 training images:
cross-fitting removes memorization in the head, and the backbone can still place each image's target
in its features, where one shared linear map reads it. The whole-dataset score holds images out of
the ridge only and cannot see this.

`--factor_holdout_fraction` keeps a share of the training images out of the concept loss (SimCLR
still trains on them), and the learnability record adds `student_unseen` and `clip_unseen`: the
factors predicted on those images by a ridge fitted on the others. The `*_holdout` arms measure it on
MetaShift and on Spur-CIFAR10, where the matched control confirms the gain of the cross-fit
(`xfit_norm` 33.6 WGA, `xfit_norm_shuffled` 18.2, SimCLR 15.6).

## Unseen images: memorization on MetaShift, generalization on Spur-CIFAR10 (2026-10-01)

A fifth of the training images stayed out of the concept loss. Explained variance of the factors,
seeds 1 and 2, epoch 500 (`seen` on the training images, `unseen` on the held-out fifth):

| dataset | arm | seen | unseen | CLIP unseen | val WGA | val acc |
|---|---|---|---|---|---|---|
| MetaShift | xfit_meaning_w10_holdout | 0.60 | 0.04 | 0.26 | 47.9 | 60.4 |
| MetaShift | xfit_meaning_w10_shuffled_holdout | 0.01 | 0.02 | 0.26 | 43.8 | 52.6 |
| Spur-CIFAR10 | xfit_norm_holdout | 0.31 | 0.27 | 0.42 | 30.2 | 68.7 |
| Spur-CIFAR10 | xfit_norm_shuffled_holdout | 0.10 | 0.10 | 0.42 | 21.7 | 58.1 |

On Spur-CIFAR10 the student carries the factors to new images, at two thirds of the CLIP score, for
the classes (aircraft 0.75, horse 0.74, truck 0.72) and for the line colour (purple 0.62) alike. On
MetaShift it stores the targets of its 1,700 training images: the unseen score peaks near 0.08 at epoch
100 to 200 and falls to 0.04 while the seen score climbs to 0.60. Only the class factors carry over in
part (cat 0.37, dog 0.23); the context factors do not (cozy 0.03). The method needs enough images for
the distilled concepts to generalize, so CelebA (162,770 training images) is its held-out test and
Waterbirds (4,795) the check of the small-data limit.

## Waterbirds with the frozen cross-fit (2026-10-01)

Seeds 1 and 2, 217 Open Images factors from 2 percent, weight 1, a fifth of the images held out:

| arm | seen | unseen | CLIP unseen | val WGA | val acc |
|---|---|---|---|---|---|
| xfit_norm_holdout | 0.04 | 0.03 | 0.16 | 46.5 | 51.1 |
| xfit_norm_shuffled_holdout | 0.02 | 0.02 | 0.16 | 43.9 | 50.5 |

Waterbirds shows no memorization and little learning: seen and unseen agree, and the student gains a
few hundredths over the control. The factors that carry signal transfer to unseen images ("Water bird"
0.25 against 0.15 shuffled, Bamboo 0.40 against 0.22, Forest 0.18 against 0.11), and most of the 217
are rare species CLIP itself explains at 0.1. Validation accuracy stays at chance. The next arms use
the meaning grouping frozen on MetaShift (text 0.85, response 0.5, factors from 5 percent: 70 Open
Images factors) at weight 10, the setting that memorized on MetaShift's 1,700 images, to see which
side Waterbirds' 4,795 images fall on.

## Waterbirds with meaning factors at weight 10 (2026-10-02)

70 Open Images meaning factors from 5 percent, weight 10, a fifth of the images held out, seeds 1 and 2:

| arm | seen | unseen | CLIP unseen | val WGA | val acc | val group acc |
|---|---|---|---|---|---|---|
| simclr (seeds 1 to 4) | | | | 45.8 | 52.3 | 47 / 54 / 52 / 64 |
| xfit_meaning_w10_shuffled_holdout | 0.01 | 0.02 | 0.27 | 42.8 | 52.6 | 43 / 57 / 58 / 66 |
| xfit_meaning_w10_holdout | 0.30 | 0.09 | 0.27 | 55.3 | 59.2 | 56 / 59 / 66 / 68 |

This is the first configuration that lifts every Waterbirds group, minority groups included, and the
best Waterbirds WGA of the project from scratch (CoSpRo reached about 50, LateTVG 48). The student
stores part of the targets (seen 0.30) and carries a third of the CLIP score to unseen images (0.09
against 0.02 for the control), where MetaShift carried 0.04. The configuration is now frozen: the
next arms run it unchanged on Waterbirds without the hold-out (seeds 1 to 4), on Spur-CIFAR10 and on
CelebA, and nothing more is tuned on Waterbirds.

## Decoupling scores and the weight grid (2026-10-03)

The per-factor explained variance cannot tell two directions from one: a direction shared by "cat" and
"couch" predicts both where they co-occur, and they co-occur on most images. The learnability record
now adds two scores, for the student and for CLIP features alike, on the unseen images when the run
holds images out and out of fold otherwise:

* `*_residual`: each factor minus its linear regression on the other factors (`partial_residuals`),
  label-free. For two factors of correlation rho a fused direction explains (1 - rho) / 2 of each
  residual, separate directions all of it.
* `*_groups`: the explained variance inside every (class, attribute) group against the variance over
  all images, from the hidden labels and for evaluation only. A fused "cat" direction drops on cats
  outdoors against cats indoors. The unseen fifth of MetaShift holds 15 to 21 minority images, so read
  the key factors, not the means.

The seventh round runs `xfit_meaning_w{1,3,5,10}_holdout` on one factor set per dataset (Open Images
meaning groups on Spur-CIFAR10 and Waterbirds, LAION on MetaShift), seeds 1 and 2. It tests whether the
weight that maximizes the unseen explained variance, a label-free choice, also gives a good WGA.

## Weight grid, CelebA and residual targets (2026-10-03)

CelebA, seeds 1 and 2, final probe: SimCLR 85.2 WGA and 87.7 accuracy, `xfit_norm` 85.8 and 88.5, its
shuffled control 82.5 and 87.8. The weight-1 cross-fit changes nothing there.

Spur-CIFAR10 with 24 Open Images meaning factors and a fifth held out, seeds 1 and 2: the unseen score
stays at 0.32 to 0.33 for weights 1, 3, 5 and 10 while the seen score reaches 0.70 to 0.76, and WGA
moves within the seed noise (26.0, 32.2, 26.7, 22.8) below the 33.6 of the default 88 factors. The
unseen score cannot choose the weight, so the label-free weight rule is dropped. On Waterbirds weight 3
gives 45.6 WGA against 51.5 at weight 10, at the same unseen score of 0.09.

The decoupling scores answer the main question. The factor residuals reach 0.03 to 0.04 in the student
against 0.15 to 0.17 for CLIP on MetaShift and Waterbirds, and 0.18 against 0.35 on Spur-CIFAR10. The
MetaShift cat factor scores -0.08 on unseen cats indoors and -0.41 on cats outdoors, where CLIP scores
0.40 and 0.46. The student learns the part the factors share, which is the dataset's correlation
structure: a squared error on standardized targets is dominated by the shared variance of correlated
factors. The eighth round trains on the residuals themselves (`--factor_targets residual`), the part
of each factor a fused direction cannot predict.

## Residual targets fail; what the Waterbirds gain is (2026-10-04)

Residual targets, seeds 1 and 2, a fifth held out: the student cannot predict them. The batch score on
Waterbirds stays at 0.00 for 500 epochs, the unseen residual score at 0.03 on both datasets (CLIP:
0.15 to 0.17), and WGA drops on Waterbirds to $41.7\pm0.9$, the level of the shuffled control. CLIP
itself explains a sixth of the residuals: the SpLiCE factors of these datasets carry little
factor-specific signal beyond what they share, so decoupling them through their residuals has little
to work with.

The weight grid on Waterbirds, seeds 1 and 2: weight 3 gives 45.7 WGA and 54.0 accuracy, weight 5 52.8
and 58.8, weight 10 55.3 and 59.2. MetaShift stays at 47 to 48 WGA and 60 to 62 accuracy for every
weight. The Waterbirds gain lifts the landbird groups most (weight 5: 57 / 66 / 56 / 62 against
47 / 54 / 52 / 64 for SimCLR) while the residual score stays at 0.04: the student gains a bird feature
that SimCLR from scratch lacks (52 percent accuracy), the absent core feature of deep feature
reweighting, and keeps it fused with the background. The ninth round tests whether the concepts matter
for this gain: the same cross-fit distils as many principal components of the CLIP image embeddings
as there are factors (`--factor_targets clip_pca`).

## Dense concept responses (2026-10-04)

SpLiCE keeps about nine concepts per image, so its factor residuals are mostly decomposition noise:
CLIP features explain 0.15 to 0.17 of them on unseen images. `--factor_targets response` replaces the
sparse codes by every image's CLIP alignment with each factor's text direction, and `response_residual`
by their residuals on the other responses. As linear functions of CLIP both are directions CLIP carries:
on unseen images CLIP explains 0.79 (MetaShift) and 0.88 (Waterbirds) of the responses and 0.41 and 0.43
of their residuals. The responses correlate more than the codes (mean absolute correlation 0.34 to 0.38
against 0.04 to 0.06), so their residuals are the part a fused direction misses. When an arm trains on
responses, its learnability record scores responses (`scored_targets`). The tenth round runs both at
weight 5 on MetaShift and Waterbirds.

## CLIP control and dense responses (2026-10-05)

Weight 5, a fifth held out, seeds 1 and 2, final probe (Spur-CIFAR10: weight 1, no hold-out):

| dataset | targets | WGA | acc | unseen EV | residual EV (CLIP) |
|---|---|---|---|---|---|
| Waterbirds | SpLiCE factors (seeds 1 to 4) | 50.6 | 57.9 | 0.10 | 0.04 (0.15) |
| Waterbirds | CLIP principal components | 51.3 | 59.5 | 0.11 | 0.05 (0.15) |
| Waterbirds | dense responses | 54.2 | 65.4 | 0.38 of 0.88 | 0.03 (0.43) |
| Waterbirds | response residuals | 44.2 | 51.4 | 0.13 of 0.88 | 0.00 (0.43) |
| MetaShift | SpLiCE factors | 47.2 | 61.7 | 0.07 | 0.03 (0.17) |
| MetaShift | CLIP principal components | 53.5 | 65.8 | 0.09 | 0.04 (0.17) |
| MetaShift | dense responses | 50.7 | 64.1 | 0.34 of 0.79 | 0.04 (0.41) |
| MetaShift | response residuals | 45.2 | 57.7 | 0.12 of 0.79 | -0.04 (0.41) |
| Spur-CIFAR10 | SpLiCE factors (88) | 33.6 | 69.7 | | |
| Spur-CIFAR10 | CLIP principal components (88) | 36.3 | 70.9 | | |

Three readings. The concepts add nothing over CLIP: principal components of the CLIP embeddings match
or beat the SpLiCE factors on all three datasets, so the gains come from distilling CLIP through a
cross-fit that memorization cannot satisfy. Dense targets carry more of CLIP than sparse codes and lift
accuracy most (Waterbirds 65 against 58). The student does not decouple: its residual score stays at
0.03 to 0.05 for every target kind, and training on residuals fails. The eleventh round asks what the
cross-fit adds to CLIP distillation (a trained head on the same principal components) and whether more
of CLIP helps (128 components).

## Spatial concept distillation (2026-10-06)

Image-level targets can separate "cat" from "couch" only through the images where the two disagree,
and erasing "couch" before predicting "cat" reduces to the residual target that failed (see
`docs/xfit/spatial_concepts_literature.md`). Cat and couch occupy different pixels, though, and the
fusion happens at global average pooling. The twelfth round supervises the concepts where they are.

* `cospro.cli.build_concept_maps` (launcher `scripts/build_concept_maps.sbatch`) reads every training
  image at 448 pixels through 224-pixel windows with frozen CLIP ViT-B/32, takes the last block with
  query-query attention and without residual and feed-forward branch (ClearCLIP), and stores per image a
  14 x 14 softmax over the factors of each patch's cosine with the factor's text direction. MetaShift
  takes 40 seconds on one GPU. The factor band is 2 percent: at 5 percent the MetaShift factors hold no
  scene concept, while at 2 percent its 130 LAION factors include couch, lawn, beds, windows and bench,
  and their maps sit on those regions.
* `--factor_spatial_weight` with `--factor_concept_maps` records each SimCLR view's crop box and flip,
  warps the maps onto the view (`roi_align`), captures the encoder's last feature map and regresses it
  location by location with the cross-fit: a ridge fitted on the locations of half the images predicts
  the other half's. `--factor_spatial_maps shuffled` gives every image another image's maps.

The spatial held-out score starts near 0.4 at epoch 1, since maps are smooth and partly predictable from
position and low-level statistics; only its rise above the shuffled control counts.

## Spatial results and the dense CLIP control (2026-10-05)

Seeds 1 to 4, a fifth held out of the concept loss, mean of the last four probes:

| dataset | arm | WGA | acc | group acc | spatial EV at epoch 500 | unseen EV |
|---|---|---|---|---|---|---|
| MetaShift | SimCLR | 44.8 +- 9.3 | 54.8 | 55 / 49 / 53 / 62 | | |
| MetaShift | spatial, shuffled maps | 45.3 +- 3.4 | 55.6 | 61 / 46 / 48 / 66 | 0.39 to 0.41 | 0.13 |
| MetaShift | spatial | 53.0 +- 5.8 | 62.2 | 65 / 53 / 58 / 73 | 0.46 to 0.48 | 0.22 |
| Waterbirds | SimCLR | 45.8 +- 1.7 | 52.3 | 47 / 54 / 52 / 64 | | |
| Waterbirds | spatial, shuffled maps | 46.7 +- 2.7 | 52.6 | 48 / 56 / 54 / 57 | 0.39 to 0.40 | 0.17 |
| Waterbirds | spatial | 53.0 +- 3.0 | 60.1 | 55 / 64 / 59 / 68 | 0.44 to 0.45 | 0.25 |

The spatial loss is the first method that beats its own control on both datasets over four seeds, by
7.7 and 6.3 points of WGA, and it lifts the minority groups (MetaShift cat outdoors 53 against 46 and
49, dog indoors 58 against 48 and 53; Waterbirds landbird on water 64 against 56 and 54, waterbird on
land 59 against 54 and 52). The student generalizes its concepts (seen and unseen scores agree, 0.23 and
0.22 on MetaShift) where image-level distillation memorized. The image-level decoupling scores move
little: the residual score stays at 0.00 against 0.15 for CLIP, and the MetaShift cat factor still scores
0.47 on unseen cats indoors and 0.06 outdoors (shuffled maps: 0.24 and 0.08). The gain shows on the
minority groups; whether it comes from the concepts or from dense CLIP information is the next control:
`--targets pca` builds as many principal components of the dense patch embeddings as there are factors
(`spatial_pca_w5_holdout`).

## Concept maps against dense CLIP components (2026-10-06)

Same seeds 1 to 4 and hold-out, mean of the last four probes:

| dataset | maps | WGA | acc | group acc |
|---|---|---|---|---|
| MetaShift | concepts | 53.0 +- 5.8 | 62.2 | 65 / 53 / 58 / 73 |
| MetaShift | principal components | 44.4 +- 7.4 | 56.3 | 56 / 45 / 56 / 69 |
| Waterbirds | concepts | 53.0 +- 3.0 | 60.1 | 55 / 64 / 59 / 68 |
| Waterbirds | principal components | 53.6 +- 1.9 | 58.1 | 56 / 58 / 61 / 66 |

On MetaShift the concept maps win on every seed (final probe 61.1 against 44.4, 51.4 against 48.6, 52.8
against 50.0, 47.2 against 34.7) and the components stay at the SimCLR level; cats outdoors reach 53
against 45. On Waterbirds the two tie. Birds against land and water backgrounds are among the strongest
directions of CLIP's patch embeddings, so its leading components carry them; cats against dogs and
couches against lawns are not, and only the concept directions single them out.

## Concept ablation of the maps (2026-10-06)

`--factor_spatial_drop` leaves out every factor whose name lists one of the given concept words. On
MetaShift `spatial_w5_noanimals_holdout` drops the 21 animal factors (cat, dog, the breeds, pet, paws,
fur) and `spatial_w5_noscenes_holdout` the 20 scene factors (couch, lawn, bed, window, desk, bench,
pasture, snow, seaside and others). Whichever set the cat-outdoor gain depends on is the concept that
makes it. The Spur-CIFAR10 map job failed on a gpuidle node whose GPU was busy; map jobs now go to the
main partition whatever `SBATCH_PARTITION` says.

## Which concepts carry the MetaShift gain (2026-10-06)

MetaShift, seeds 1 and 2 for the ablations, a fifth held out, mean of the last four probes:

| maps | WGA | acc | cat-indoor / cat-outdoor / dog-indoor / dog-outdoor |
|---|---|---|---|
| all 130 concepts (seeds 1 to 4) | 53.0 +- 5.8 | 62.2 | 65 / 53 / 58 / 73 |
| without the 20 scene concepts | 50.0 +- 2.5 | 61.9 | 67 / 55 / 58 / 69 |
| without the 21 animal concepts | 45.7 +- 9.6 | 58.9 | 58 / 46 / 62 / 71 |
| shuffled maps (seeds 1 to 4) | 45.3 +- 3.4 | 55.6 | 61 / 46 / 48 / 66 |

Leaving out the animal concepts (cat, dog, the breeds, pet, paws, fur) returns cats outdoors to the
control level of 46; leaving out the scene concepts (couch, lawn, bed, window and others) keeps them at 55.
The cat-outdoor gain comes from localizing the animal concepts. Waterbirds at weight 10 (seeds 1 and 2):
53.5 WGA and 61.8 accuracy, level with weight 5. The MetaShift weight-10 runs and the real Spur-CIFAR10
spatial runs failed at epoch 0 on node gpu003 with an IndexError that does not occur locally; the
Spur-CIFAR10 shuffled runs on gpuidle nodes take 2.05 minutes per epoch against 0.46 on the dgx nodes.

## Four-seed ablation, weight 10 and Spur-CIFAR10 spatial (2026-10-07)

Seeds 3 and 4 weaken the two-seed ablation. Over seeds 1 to 4 (MetaShift, a fifth held out, last four
probes): all concepts 53.0 +- 5.8 WGA, cats outdoors 53; without scene concepts 47.9 +- 2.8, cats outdoors
50; without animal concepts 46.6 +- 5.8, cats outdoors 49; shuffled maps 45.3 +- 3.4, cats outdoors 46.
Both ablations lose part of the gain and differ from each other by less than the seed noise: the gain
needs the full concept set, and the data cannot yet attribute it to the animal concepts alone. MetaShift
at weight 10 (seeds 1 and 2): 48.3 WGA, below weight 5.

Spur-CIFAR10 (seeds 1 and 2, a fifth held out): spatial maps 38.1 +- 0.5 WGA and 69.3 accuracy, the best
Spur-CIFAR10 result of the branch, against 33.6 for the image-level cross-fit, 17.2 for shuffled maps and
15.6 for SimCLR.

## Headline spatial runs without the hold-out (2026-10-08)

Spatial maps at weight 5 with every image in the concept loss, seeds 1 to 4, mean of the last four probes:
MetaShift 48.5 +- 3.7 WGA and 60.7 accuracy (with the hold-out 53.0 and 62.2; SimCLR 44.8 and 54.8),
Waterbirds 51.4 +- 2.2 and 59.2 (with the hold-out 53.0 and 60.1; SimCLR 45.8 and 52.3). Both stay above
SimCLR on accuracy on every seed and on WGA on average; the drop against the hold-out arms lies within
the seed spread, largest on MetaShift seed 1 (61.1 against 51.4).

## Concept-region contrast (2026-10-07)

The spatial regression makes concepts visible: accuracy rises on every seed, but on MetaShift the gain
sits in the majority groups (cats indoors 64 against 55) and the cat factor still scores far lower on cats
outdoors. Nothing in it asks a cat patch on a lawn to resemble a cat patch on a couch, and the probe reads
the average of all patches. `--factor_region_weight` adds that demand (DetCon-style, Hénaff et al. 2021):
each view's feature map is pooled inside its two dominant concept regions (warped map probability as the
weight, mean probability at least 0.1), projected by a small head and contrasted across the batch. The
regions of the same concept in other images are positives, weighted towards pairs whose images' concept
mixes differ; the image's own regions leave the denominator. On MetaShift a batch holds about 335 regions,
94 percent with positives (about 20 each); the dominant regions are cats, puppies, blankets, lawn, fields
and windows, so the generic "pet" concept rarely enters. Arms: `region_w1_holdout` (spatial weight 5 plus
region weight 1) and `region_w1_shuffled_holdout`.

## Region contrast results (2026-10-09)

With the hold-out, mean of the last four probes. MetaShift region uses seeds 1, 3 and 4 (seed 2 died at
epoch 175); the Waterbirds shuffled region arm uses seeds 3 and 4 (seeds 1 and 2 died at start). All three
deaths ran on dgx nodes and left no Python error.

| dataset | arm | WGA | acc | group acc |
|---|---|---|---|---|
| MetaShift | spatial | 53.0 +- 5.8 | 62.2 | 65 / 53 / 58 / 73 |
| MetaShift | region | 49.3 +- 7.8 | 64.6 | 73 / 49 / 63 / 71 |
| MetaShift | region, shuffled maps | 42.4 +- 6.7 | 54.3 | 55 / 43 / 53 / 66 |
| Waterbirds | spatial | 53.0 +- 3.0 | 60.1 | 55 / 64 / 59 / 68 |
| Waterbirds | region | 55.3 +- 3.1 | 62.7 | 62 / 65 / 58 / 62 |
| Waterbirds | region, shuffled maps | 49.7 +- 2.2 | 55.0 | 52 / 59 / 50 / 57 |

The region contrast raises the concept scores on unseen images on every seed (MetaShift 0.22 to 0.29,
Waterbirds 0.25 to 0.32) and the cat factor on unseen cats outdoors from 0.05 to 0.12. The probe gain goes
to the majority groups again. On MetaShift WGA falls on all three paired seeds (by 4.8, 2.5 and 6.3) and
cats outdoors fall from 53 to 49. On Waterbirds accuracy rises on every seed (2.6 on average) and WGA by
2.3, on three of four seeds. On Waterbirds seeds 3 and 4 the shuffled region arm gains as much over the
shuffled spatial arm (6.3 and 3.8) as the real region arm over the real spatial arm (5.3 and 4.1): part of
the Waterbirds gain belongs to region pooling itself. `region_w1_ctx3_holdout` raises the context weight to 3.

## Where decoupling fails: the student inherits the co-occurrence (2026-10-09)

The CLIP teacher separates cat from its surroundings: its cat factor scores 0.92 on unseen cats indoors
and 0.88 outdoors. The student's cat factor scores 0.61 and 0.12 with the region contrast, 0.47 and 0.05
with the spatial loss alone, and 0.18 on dogs indoors. It predicts cat-ness where cat and scene agree
(cats indoors, dogs outdoors) and fails on both minority groups: the student's "cat" tracks indoor scenes.

Every concept loss so far is an expectation over the natural training images, where 88 percent of cats sit
indoors. A position of the ResNet-18 `layer4` map sees most of the image, so the cheapest predictor of "cat
here" combines local evidence with the scene, and the scene is right 88 percent of the time. The spatial
and region losses localize the concept; the joint distribution of concept and context that the student
fits stays the training distribution, and the losses reward the scene shortcut as long as it does.

## Next method: concept compositing (proposal, 2026-10-09)

The concept maps can change that distribution. For a fraction of each batch, cut the object region out of
view A (the warped maps of its object concepts, soft and resized to the view) and paste it onto view B of
another image, with B's own object region blurred away. The composite replaces A's second view in the
SimCLR loss. Its concept maps are composited the same way, so the spatial regression sees cats on lawns and
dogs on couches as often as the batch pairs them. With B drawn at random, object and context are
independent in the composites: they sample the interventional distribution in which the spurious attribute
carries no information about the object, the content and style condition under which contrastive learning
isolates content (von Kügelgen et al., 2021). The object concepts come from the text-only scores of
`cospro/pipeline/concept_type.py`, so no label enters. Composites are built on the GPU from views already
in the batch; the crop boxes and `warp_maps` give the masks at view resolution.

Precedent: background swaps with unsupervised saliency raise ImageNet-9 Mixed-Rand accuracy from 70.7 to
84.1 for MoCo-v2 (Ryali et al., 2021); copy-paste at probability 0.3 helps and always-on copy-paste
collapses (DiLo, Zhao et al., 2021). The concept masks supply the controls that tie the gain to a concept:
random regions of the same area pasted onto the same donors, context concepts pasted in place of object
concepts, and shuffled maps. Object pasting above random and context pasting is the evidence that the
concept "cat" produces the cat-outdoor gain.

## Concept compositing, implemented (2026-10-14)

Which concepts get cut is decided without labels and without naming the spurious pair. The maps give, per
14 x 14 cell, a softmax over the run's factors. `cospro.cli.score_concept_types` scores every concept group by
its words alone (CLIP text similarity to scene prompts minus similarity to object prompts); a factor averages
its groups, and the factors below the median of the run's factor set count as objects. The text encoder rates
almost every word closer to an object than to a scene (MetaShift groups range from -0.27 to +0.10), so the
split is relative. On MetaShift the 65 object factors hold every animal and breed factor and also verbs and
people (jump, editor, owner); lawn, couch, bed, desk, window, beach and snow fall on the context side. On
Waterbirds the birds are objects and bamboo, forest, wetland, loch and landscape are context; "File" (7.7
percent of patches) lands among the objects. The one assumption, stated in `cospro/pipeline/concept_type.py`,
is that the target is an object and the spurious attribute its surroundings; CelebA breaks it and gets no
compositing arms.

Per batch, after the views reach the GPU (`TrainingMethod.compose_views`): an image qualifies when it carries
a concept-loss weight (held-out images never take part, as image or donor) and its object mass covers at
least 5 percent of its second view; 30 percent of the qualifying images (ramped in with the factor schedule)
get a random donor from the batch whose objects cover at most three quarters of its view. The object mass of
the image and of the donor is warped onto the view with its crop box and sharpened (mass 0.25 maps to 0, 0.75
to 1, linear between) so a pasted region keeps a seam half a map cell wide. The donor's objects are painted
over by normalized convolution from their surroundings; the image's object region goes on top. The spatial
targets of a composite are composited with the same masks (`composite_maps`), which is exact for standardized
maps because the mixture is affine. On a local MetaShift batch 23 percent of the images were composited and
the pasted region covered 44 percent of a view: dogs from couches land on lawns, a retriever from a lawn in an
office. Arms (spatial weight 5, hold-out): `comp_w5_holdout`, controls `comp_random_w5_holdout` (another
image's object silhouette), `comp_context_w5_holdout` (the context region is kept, the object painted over)
and `comp_shuffled_w5_holdout` (shuffled maps), and `comp_only_holdout` (compositing without the spatial loss).

## Compositing results on MetaShift (2026-10-16)

Seeds 1 to 4 with the hold-out, mean of the last four probes:

| arm | WGA | acc | cats outdoors | cat factor indoors / outdoors (unseen) |
|---|---|---|---|---|
| spatial maps (same seeds) | 53.0 +- 5.8 | 62.2 | 53.1 | 0.47 / 0.06 |
| compositing, object regions | 52.3 +- 1.9 | 63.3 | 55.2 | 0.55 / 0.12 |
| control: another image's silhouette | 50.6 +- 3.7 | 60.0 | 51.0 | 0.53 / 0.16 |
| control: context region kept | 49.3 +- 1.9 | 60.1 | 49.3 | 0.46 / 0.02 |
| control: shuffled maps | 48.4 +- 3.1 | 56.6 | 49.0 | 0.26 / 0.13 |
| SimCLR (same seeds) | 44.8 | 54.8 | 48.6 | |

Compositing gives the best and steadiest MetaShift WGA of the branch and the most cats outdoors. Its arms
order as the concept hypothesis predicts (object regions above random silhouettes above context regions);
object compositing beats the context control on all four seeds (3.0 WGA on average) and the shuffled
control on three (3.8). Against the random-silhouette control the gain is 1.6 with two of four seeds
ahead, and against spatial maps on the same seeds it is -0.8 (seeds -9.0, -0.3, -1.7, +8.0). The cat
factor on unseen cats outdoors moves from 0.06 to 0.12, and the random-silhouette control reaches 0.16:
the representation still ties the cat to its scene. Waterbirds compositing and `comp_only_holdout` have
not run yet. With seeds 5 and 6 the spatial arms stand at MetaShift 51.1, Waterbirds 52.8 +- 4.7
(SimCLR 46.7, shuffled maps 47.5); Waterbirds shuffled region seed 1 completes that arm at 49.3 +- 1.7.

## Symmetric concept mixing: compositing without the object prior (proposal, 2026-10-19)

Compositing chooses which regions to keep through text prompts that name objects and scenes. That prior
holds for MetaShift and Waterbirds and fails wherever the spurious attribute is something other than the
surroundings (CelebA: hair colour against gender), and it reintroduces knowledge of what is spurious, which
the concept-factor idea set out to avoid: decouple every correlated pair and let the balanced probe pick the
side the task needs. The positive pair is what forces the choice. Pairing a composite with its source as a
hard SimCLR positive asks the representation to ignore whatever came from the donor, so somebody has to
decide what is ignorable.

The symmetric version treats every concept alike and asks for composition instead of invariance:

1. Mask: each concept region of view A (cells its factor wins, factors covering at least three cells) is
   kept with probability one half, kept area clipped to 25 to 75 percent. No concept is typed.
2. Composite: x = M * A + (1 - M) * B with a random donor B; nothing is painted over, since no part counts
   as background.
3. Region-to-source contrast (RegionCL, Xu et al., ECCV 2022): the composite's feature map pooled inside M
   is a positive of A's embedding, pooled inside 1 - M a positive of B's. Each region alone has to identify
   its source with foreign surroundings, which targets the measured failure: the student's cat cells carry
   the indoor scene, because each layer4 cell sees most of the image.
4. Soft global positive (Un-Mix, Shen et al., AAAI 2022): the composite's embedding is a positive of A with
   weight equal to the kept area and of B with the rest, so it keeps both sources.
5. The spatial concept regression runs on the composited maps (`composite_maps`), as now.

The assumption that remains is spatial separability: correlated concepts occupy different pixels. It is
symmetric (no concept is declared class or spurious) and it can be checked without labels by comparing a
pair's image-level correlation with its pixel overlap. Additive decoders identify blocks of latent factors
even when the factors are statistically dependent and extrapolate to unseen combinations (Lachapelle et al.,
NeurIPS 2023), and compositional generalization needs training support that covers each component's
marginal support (Wiedemer et al., NeurIPS 2023); mixing widens the support of (concept, surrounding) from
the training diagonal towards the product.

Check on the MetaShift maps (image-level presence: a factor wins at least three cells; one composite per
image, random donor, no training): the twelve most correlated pairs are mostly the same thing named twice
(cats and cat 0.82, lawn and field 0.48, laptop and computing 0.47). They share pixels and stay correlated
under mixing (0.82 to 0.52), which is the intended behaviour. The twelve strongest animal-by-scene pairs (cat
and field -0.41, desk and cat 0.31, blanket and cat 0.30) fall from a mean |phi| of 0.33 to 0.19 under
symmetric mixing and to 0.17 under object pasting. With group labels used only for this check, symmetric
mixing puts an animal region on a donor of the other scene type in 35 percent of composites (object pasting
48 percent, natural minority share 12 percent). The symmetric version keeps most of the decorrelation
without the prior.

Arms (MetaShift, Waterbirds and CelebA, spatial weight 5, hold-out): `mix_w5_holdout` (steps 1 to 5),
`mix_box_w5_holdout` (random rectangles of the same area: tests whether concept boundaries matter),
`mix_shuffled_w5_holdout` (masks and targets of other images), and ablations without step 3 and with the
hard positive in place of step 4. The existing object-paste arm measures what dropping the prior costs.

Implemented 2026-10-19 (`--factor_mix_fraction`, `--factor_mix_mask concepts|boxes`, `--factor_mix_region`,
`--factor_mix_global soft|hard`, `--factor_mix_weight`). A concept region is the set of cells its factor wins,
counted from three cells; three draws per image look for a kept area between 25 and 75 percent. Donors come
from the same cross-fit half of the batch, so the mixed views join the spatial ridge without crossing halves,
and held-out images take part neither as image nor as donor. The mixed view is a third encoder pass; the
keys of both contrastive terms are the first views' projections, which share no pixels with the mix. Each
term weighs 0.5, so together they weigh about as much as the SimCLR loss (both start near log 128 = 4.85).
On a local MetaShift run with fraction 0.5, a third of every batch got a mixed view with a kept area of 45
percent, and an epoch took about a tenth longer. The arms run on MetaShift and Waterbirds; CelebA needs its
concept maps first, and its 162,770 training images make the maps several times larger than the loader's
float32 copies fit into 40 GB.

## Waterbirds compositing and MetaShift mixing (2026-10-22)

Seeds 1 to 4 with the hold-out, mean of the last four probes; group order landbird-land / landbird-water /
waterbird-land / waterbird-water and cat-in / cat-out / dog-in / dog-out.

| dataset | arm | WGA | acc | group acc |
|---|---|---|---|---|
| Waterbirds | SimCLR | 45.8 | 52.3 | 47 / 54 / 52 / 64 |
| Waterbirds | spatial maps | 53.0 | 60.1 | 55 / 64 / 59 / 68 |
| Waterbirds | compositing, object regions | 62.0 +- 3.9 | 65.9 | 63 / 68 / 68 / 67 |
| Waterbirds | control: another image's silhouette | 57.0 +- 5.5 | 64.6 | 60 / 69 / 63 / 65 |
| Waterbirds | control: context region kept | 52.7 +- 2.1 | 60.0 | 58 / 62 / 57 / 62 |
| Waterbirds | control: shuffled maps | 52.9 +- 1.0 | 57.2 | 57 / 54 / 59 / 67 |
| MetaShift | spatial maps | 53.0 | 62.2 | 65 / 53 / 58 / 73 |
| MetaShift | symmetric mixing | 51.0 +- 8.5 | 61.2 | 59 / 53 / 61 / 72 |
| MetaShift | mixing, rectangles | 51.8 +- 3.4 | 61.4 | 60 / 53 / 59 / 74 |
| MetaShift | mixing, no region contrast | 52.8 +- 6.5 | 62.2 | 65 / 55 / 57 / 70 |
| MetaShift | mixing, hard positive | 51.6 +- 2.8 | 61.2 | 60 / 51 / 60 / 72 |
| MetaShift | mixing, shuffled maps | 47.7 +- 1.5 | 58.9 | 60 / 49 / 55 / 72 |

Compositing on Waterbirds is the strongest result of the branch: it beats spatial maps on all four seeds
(9.0 WGA on average), the context and shuffled controls on all four (9.3 and 9.1) and another image's
silhouette on three (5.0), and it flattens the group accuracies (63 to 68), the minority groups included. It
lies above LateTVG's published Waterbirds figure (55.4 for SimCLR). The object prior carries it: keeping
the context region instead loses everything compositing adds.

Symmetric mixing on MetaShift adds nothing over spatial maps (-2.0, two of four seeds) and ties its
rectangle control (-0.8): concept boundaries make no difference there, and the main arm's seeds spread from
44.4 to 62.2. The cat factor on unseen cats outdoors stays at 0.01 to 0.03 in every mixing arm. The
per-group factor scores rest on few images (the unseen fifth holds about 11 waterbirds on land and 15 to 21
cats outdoors), so on these groups the validation WGA (133 and 72 images) is the reliable measure. Symmetric
mixing on Waterbirds, where compositing works, decides whether the prior is necessary.

## Symmetric mixing on Waterbirds, compositing seeds 5 and 6, and the locked test (2026-10-26)

Waterbirds, mean of the last four probes:

| arm | seeds | WGA | acc | group acc |
|---|---|---|---|---|
| spatial maps | 1 to 4 | 53.0 | 60.1 | 55 / 64 / 59 / 68 |
| symmetric mixing | 1 to 4 | 49.2 +- 2.5 | 56.0 | 49 / 58 / 67 / 61 |
| mixing, rectangles | 1 to 4 | 49.0 +- 4.8 | 56.2 | 49 / 59 / 65 / 62 |
| mixing, shuffled maps | 1 to 4 | 44.8 +- 3.3 | 51.6 | 45 / 52 / 67 / 59 |
| compositing | 1 to 6 | 59.1 +- 7.1 | | |
| compositing, another image's silhouette | 1 to 6 | 54.2 +- 6.1 | | |

Symmetric mixing loses to spatial maps on all four seeds (-3.8 WGA, -4.1 accuracy) and ties its rectangle
control (+0.2), after tying it on MetaShift as well. Typing no concept leaves an augmentation that behaves
like CutMix and weakens the majority group landbird on land (49 against 55). The gain of compositing needs
the asymmetric choice: keep the object, replace its surroundings. The line ends here; the prior stays and
the thesis states it, as the background-augmentation literature does.

Compositing seeds 5 and 6 score 60.2 and 46.1. Over six seeds it beats spatial maps on all six (6.3 on
average) and another image's silhouette on four (4.8). Seed 6 is hard for every method: spatial maps,
compositing and the silhouette control all drift down over the last 150 epochs to 46, with the majority
group landbird on land as the worst group in each, so the seed and its probe, not the method, set that
number.

`FINAL_TEST=1 bash scripts/submit_factor_study.sh <dataset> ...` runs the same arms with `--final_test` (one
probe on the test split after training) under the study `factors_<dataset>_test`; `summarize_study` writes a
locked-test table for it and the results book leaves those runs out. Waterbirds test holds 642 images in
each minority group against 133 in validation.

## Per-group heads: an asymmetry without a concept type (2026-10-26)

Compositing gains through the object prior and symmetric mixing without it does not gain, so the prior-free
method of the thesis is the spatial concept loss. A likely reason symmetric mixing fails: one embedding was
asked to stay the same whatever part was replaced, sometimes the scene and sometimes the cat, and the
contradictory demands cancel. LooC (Xiao et al., ICLR 2021) resolves such conflicts with one embedding space
per invariance over a shared backbone. The per-group heads (`--factor_mix_mask groups --factor_mix_global
group`) follow it:

1. The factors fall into 12 groups by the pixels they share in the maps (`coactivation_groups`: cosine of
   two factors' maps over every cell of the images the concept loss sees, spectral clustering). On MetaShift
   the groups are cats (tabby, cats, kittens, cat), dogs (dog, puppies, spaniel, paws, collie, labrador),
   lawn (pasture, grazing, lawn), desks (workspace, laptop, bookshelf), vehicles, beaches and boats, and one
   group of 55 remaining factors. No label, prompt or concept type enters.
2. For half of the images one group covering 15 to 85 percent of the second view is drawn; its regions are
   kept, the rest comes from a donor of the same cross-fit half whose copy of that group is painted over.
3. The mix is a positive of its image only in that group's own projection head. The cat head learns to
   ignore what surrounds cats, the lawn head what stands on lawns, and the backbone has to carry both, so the
   balanced probe can pick either side without anyone naming the class.

A local MetaShift check mixed 37 percent of each batch, used 8 of the 12 heads per batch at a kept area of
40 percent, and the group loss started at chance. Arms (MetaShift and Waterbirds, spatial weight 5,
hold-out): `heads_w5_holdout`; controls `heads_single_w5_holdout` (every pair in the SimCLR head, the
conflict the heads remove), `heads_random_w5_holdout` (a random head per pair: heads without the binding to
the kept group) and `heads_shuffled_w5_holdout` (shuffled maps). The heads count if they beat the single
head and the spatial maps; compositing (Waterbirds 59.1) marks what a correct prior reaches.

## Seeds 5 and 6, and stalled jobs (2026-10-10)

MetaShift with the hold-out, mean of the last four probes: spatial maps 51.1 +- 5.4 WGA and 61.2 accuracy
over six seeds (cats outdoors 51), SimCLR 44.3 +- 7.3 and 55.2 (cats outdoors 47), shuffled maps 45.6 +- 3.0
and 56.3 over five seeds (cats outdoors 46). The spatial arm beats SimCLR on five of six paired seeds, by
6.8 WGA on average; seeds 5 and 6 sit below seeds 1 to 4 (48.6 and 45.8). Region seed 2 finished at 52.1
against 50.7 for spatial, so the region arm stands at 50.0 +- 6.5 over four seeds, below spatial on three.
The Waterbirds shuffled region arm seed 2 scores 47.7, level with shuffled spatial maps (47.5): over seeds 2
to 4 the region arm beats its control by 7.8 and the spatial arm its control by 6.0.

Six jobs stalled: their records stopped at epochs 275, 300 and 375 (inside the periodic probe) or during
setup before the first epoch (after W&B started, before the dataset and concept factors were recorded).
None wrote an error; Slurm kept them running until they were cancelled. Five of the eight jobs on dgx003
stalled, one of four on dgx004 and none of three on dgx002; the earlier region runs that stopped without an
error also ran on dgx nodes. Jobs submitted with MAX_PARALLEL wait for their lane predecessor to end, so a
stalled job also holds back the jobs behind it.

## Results book

`python -m cospro.cli.build_results_book` writes one page per dataset and method under
`outputs/reports/results/`, with the hyperparameters that tell the arms apart, the seeds and the
validation results, and the dataset's SimCLR arms as reference. `run_training.sbatch` rebuilds it after
every run.

## References

* Kirichenko et al., Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations,
  ICLR 2023. https://arxiv.org/abs/2204.02937
* Chen, Luo and Li, Intriguing Properties of Contrastive Losses, NeurIPS 2021.
  https://arxiv.org/abs/2011.02803
* Tsai et al., Conditional Contrastive Learning for Improving Fairness in Self-Supervised Learning,
  2021. https://arxiv.org/abs/2106.02866
* Bertinetto et al., Meta-learning with differentiable closed-form solvers (R2D2), ICLR 2019.
  https://arxiv.org/abs/1805.08136
* Lee et al., Meta-Learning with Differentiable Convex Optimization (MetaOptNet), CVPR 2019.
  https://arxiv.org/abs/1904.03758
* Zhang et al., Deep Stable Learning for Out-of-Distribution Generalization, CVPR 2021.
  https://arxiv.org/abs/2104.07876
* Locatello et al., Weakly-Supervised Disentanglement Without Compromises, ICML 2020.
  https://arxiv.org/abs/2002.02886
* Yao et al., Multi-View Causal Representation Learning with Partial Observability, ICLR 2024.
  https://arxiv.org/abs/2311.04056
* Yang et al., Identifying Spurious Biases Early in Training through the Lens of Simplicity Bias,
  AISTATS 2024. https://arxiv.org/abs/2305.18761
* von Kügelgen et al., Self-Supervised Learning with Data Augmentations Provably Isolates Content from
  Style, NeurIPS 2021. https://arxiv.org/abs/2106.04619
* Ryali, Schwab and Morcos, Characterizing and Improving the Robustness of Self-Supervised Learning
  through Background Augmentations, 2021. https://arxiv.org/abs/2103.12719
* Zhao et al., Distilling Localization for Self-Supervised Representation Learning (DiLo), AAAI 2021.
  https://arxiv.org/abs/2004.06638
