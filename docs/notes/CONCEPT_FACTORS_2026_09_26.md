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
