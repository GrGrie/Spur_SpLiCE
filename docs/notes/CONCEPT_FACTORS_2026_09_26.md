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

## References

* Kirichenko et al., Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations,
  ICLR 2023. https://arxiv.org/abs/2204.02937
* Chen, Luo and Li, Intriguing Properties of Contrastive Losses, NeurIPS 2021.
  https://arxiv.org/abs/2011.02803
* Tsai et al., Conditional Contrastive Learning for Improving Fairness in Self-Supervised Learning,
  2021. https://arxiv.org/abs/2106.02866
* Yang et al., Identifying Spurious Biases Early in Training through the Lens of Simplicity Bias,
  AISTATS 2024. https://arxiv.org/abs/2305.18761
