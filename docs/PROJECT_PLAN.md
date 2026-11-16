# Project plan: Concept Learning Curves and Early Concept Swap

Version 3, 2026-11-16. Branch `concept-factors`.

This file is the single source of truth for the thesis project. Every person and every coding model
reads it first and follows it. The theory with full proofs lives in
[`paper/ConceptLearningCurves.tex`](../paper/ConceptLearningCurves.tex); when the two files disagree,
this file wins and the `.tex` gets corrected.

Words with a fixed meaning: **must** marks a binding rule, **default** marks a starting value that an
experiment may change through a command-line option, **later** marks work outside the current stage.

---

## 1. Goal and setting

**Thesis goal.** Raise worst-group accuracy (WGA) of self-supervised representations on datasets with a
spurious attribute, small ones included (MetaShift with 1,700 training images, Waterbirds with 4,795),
using patch-wise concepts from CLIP and SpLiCE, in a way that is an explainability method and has a
measurable gain.

**The setting is the one of Spatial Concept Distillation (SCD) and must stay unchanged.** Only the
method idea differs.

| Topic | Fixed value |
|---|---|
| Training | SimCLR on the training split, ResNet-18 from scratch (`resnet18_large` for 224-pixel images) |
| Labels in training | none: no class label and no information about the spurious attribute or the groups |
| Entry point | `spur_splice.py` through `scripts/run_training.sbatch`, preset `matched` (batch 128, temperature 0.05, four workers), 500 epochs |
| Evaluation | the existing linear probe on the group-balanced split `ds_train`, every 25 epochs; WGA on the validation split is the headline metric |
| Reported number | mean of the last four probes of a run, as in the SCD notes |
| Seeds | 1 to 4 for a first table, more before a claim |
| One run per seed | audit, flagging and swap happen inside one training run |
| Studies | `factors_<dataset>` through `scripts/submit_factor_study.sh`, so new arms sit beside the existing SimCLR, spatial and compositing runs |
| Development dataset | MetaShift; Waterbirds only after the kill test passes |
| Tracking | W&B and `run.json` as in every SSL run of the repository |

**What the supervisor asked for.**

1. A clear gain over the baseline.
2. A real link to explainable AI: the concept tool has to explain the model that is trained.
3. Patch-wise concept gathering stays the core ingredient.

**What ended.** SCD regressed the student's feature map onto CLIP concept maps. It is knowledge
distillation from CLIP, and its explanation describes the teacher. The new method has no concept loss
and no CLIP target.

## 2. The idea in five sentences

SimCLR learns the easiest features first, and one direction that fuses an object with its usual
surroundings serves it well; the SCD measurements showed exactly this (the student's cat feature
followed the indoor scene). Patch-wise concept maps tell where each named concept sits in each
training image. Occluding the region of a concept and measuring how far the encoder's embedding moves
gives the **reliance** of the encoder on that concept, and tracking it over the first epochs gives one
**learning curve** per concept. The concepts that lead the curves are **flagged**. From then on the
flagged regions of each view are exchanged with flagged regions of other images, so the two views of
an image agree only on what lies outside the flagged regions, and SimCLR has to build its
representation from the rest.

Roles of the components:

- **SpLiCE and CLIP** supply the names and the locations of concepts. They supply no training target.
- **The audit** (reliance, curves, flags) explains the encoder under training. It is the xAI
  contribution, and its report is readable by a person.
- **The swap** is the intervention that the audit selects. It is the compositing of the SCD line with
  one change: the text prior "objects against scenes" is replaced by the encoder's own learning order.

Relation to the SCD results: compositing with the object prior reached 62.0 WGA on Waterbirds against
53.0 for spatial maps, and every prior-free variant (symmetric mixing, per-group heads) failed. The
missing piece was an asymmetry without a hand-given prior. The audit is the candidate for it.

## 3. Method specification

### 3.1 Inputs

- Training images `x_i`, `i = 1..n`. No label.
- Concept maps: a tensor `[n, F, 14, 14]`, per cell a distribution over `F` concept factors, built by
  `cospro.cli.build_concept_maps` from frozen CLIP ViT-B/32 with ClearCLIP patch embeddings and the
  SpLiCE concept dictionary. The file also holds `sample_ids` (`"<dataset>:<index>"`) and
  `factor_names`. These are the maps the spatial SCD runs used (factor band 2 percent).

### 3.2 Concept units and regions

1. Group the `F` factors into `K` units with `coactivation_groups(maps, K, seed=0)`. Default `K = 12`.
   `K = 0` selects single factors as units.
2. The unit map of an image is the sum of its factors' maps: `[n, K, 14, 14]`.
3. The **hard region** of unit `k` in image `i` is the set of cells where unit `k` has the largest
   unit-map value. The regions of one image partition its 196 cells.
4. Unit `k` is **present** in image `i` when its hard region holds at least 3 cells.
5. A unit is **audited** when it is present in at least 2 percent of the training images.
6. The name of a unit lists its three factors with the largest map mass.

### 3.3 Reliance

- **Audit images**: every training image without augmentation (resize to the dataset's image size,
  normalize), encoder in evaluation mode. The audit must leave the training random stream untouched.
- **Pixel mask**: the hard region (a 14 x 14 binary map) upsampled bilinearly to the image size and
  passed through `sharpen`.
- **Occlusion** `O_k(x)`: the pixels under the mask are replaced by `inpaint`, a normalized
  convolution of the surrounding pixels.
- **Embedding shift** (statistic `shift`, the default):
  `R_k(t) = mean over images that show k of [1 - cos(h_t(x), h_t(O_k(x)))]`, where `h_t` is the
  encoder output (the features the probe reads).
- **Margin drop** (statistic `margin`, recorded in every run): in the projection space of SimCLR, the
  instance-discrimination margin of a query is its similarity to its own image minus its largest
  similarity to another image of the batch. The statistic is the mean drop of this margin when the
  occluded copy is the query.
- **Acquired reliance**: `D_k(t) = R_k(t) - R_k(0)`. A randomly initialized encoder already moves
  under any occlusion; the difference keeps what training added.
- **Share**: `S_k(t) = max(D_k(t), 0) / sum_j max(D_j(t), 0)` over the audited units.
- Area is left in: a large region is easy because it is large (Remark on area in the `.tex`).

### 3.4 Concept learning curves

Audits happen before training (`t = 0`), every `curve_audit_every` epochs inside the early window, at
its last epoch `T0`, afterwards every `curve_late_audit_every` epochs and at the last epoch.
Defaults: `T0 = 50`, audit every 5 epochs early and every 50 epochs late. The curves are the
sequences `t -> S_k(t)`.

### 3.5 Flagging

Flagging happens once, right after the audit of epoch `T0`.

- **Onset score**: `s_k = mean of S_k(t)` over the audits with `1 <= t <= T0`.
- Rule `top` (the method): sort the audited units by decreasing `s_k`; the flagged set is the
  shortest prefix whose onset scores reach the share `tau` of their total. Default `tau = 0.5`.
- **Area cap**: a unit that would lift the expected flagged share of an image (presence times area,
  summed) above 0.75 ends the prefix. One unit is always flagged and one audited unit always stays.
- Control rules, each flagging as many units as `top` does: `bottom` takes them from the end of the
  ranking and `random` draws them uniformly from the audited units with a seed.
- `units` flags an explicit list of unit ids (human-corrected flags).
- `oracle` flags the audited units that the hidden-label typing of Section 7 calls attribute units.
  It reads group labels by design and exists only as the upper-bound control arm.
- `none` flags nothing: the run audits and trains exactly as SimCLR.
- If no unit has a positive acquired reliance in the early window, nothing is flagged and the run
  continues as SimCLR; `flagged_units.json` records the reason.

### 3.6 Early concept swap

From epoch `T0 + 1` on, for every batch on the GPU and **for each of the two views independently**:

1. `phi_i`: the flagged mask of the view, obtained by warping the union of the image's flagged hard
   regions with the view's crop box and flip (`warp_maps`) and `sharpen`. `kappa_i = 1 - phi_i` is the
   kept mask.
2. A view is **eligible** when the mean of `kappa_i` lies in `[0.05, 0.95]`. A view is a **donor**
   when the mean of its `phi` is at least 0.25.
3. Each eligible view is swapped with probability `q`. Default `q = 0.5`, reached by a linear ramp
   over 10 epochs after `T0`.
4. A swapped view draws a donor `j` from the same view tensor of the batch, different from itself, and
   becomes `kappa_i * x_i + (1 - kappa_i) * b_j`, where `b_j` is the donor view with its own kept
   region painted over (`composite_images(views, donors, keep=kappa_i, donor_objects=kappa_j)`).
5. The SimCLR loss, the positives and the negatives are unchanged.

The swap draws from its own seeded random stream. Control `--curve_shuffle_maps true` gives every
image the regions of another image. Both views are swapped because a single swapped view leaves a
share of the flagged concept in the learned direction (Remark on one view in the `.tex`).

### 3.7 Why it should work

Inside a model with two Gaussian concepts and a linear contrastive encoder (proofs in the `.tex`,
formulas verified numerically):

- SimCLR's surrogate objective learns a direction that fuses the core and the spurious concept, and
  learns it before any other direction;
- the acquired reliance of the spurious concept exceeds that of the core concept at every time, so
  the onset score ranks it first;
- swapping the flagged concept in both views with probability `q` turns the learned direction
  monotonically towards the core concept, and the WGA of a balanced linear probe rises monotonically
  to the optimum of a linear classifier at `q = 1`;
- swapping the core concept by mistake lowers WGA monotonically, which is why the audit has to type
  the concepts and why untyped mixing failed;
- resampling images keeps `4 p (1 - p)` of the sample (42 percent on MetaShift), the swap keeps all.

**The assumption that no proof covers**: the spurious concept is the one with the larger second
moment (signal plus noise, growing with its area), which the encoder therefore relies on first.
Without labels the two concepts play symmetric roles, so every label-free method assumes something of
this kind. A large animal in front of a small scene breaks the assumption; stage K1 measures it.

## 4. Architecture

The method is one registered training method of the existing SSL trainer. No new entry point exists.

```text
concept maps (.pt, scratch)        spur_splice.py --splice_mode concept_curves
        |                                        |
        v                                        v
 cospro/curves/core.py  <------  cospro/methods/concept_curves.py  (ConceptCurves)
   build_units, hard_regions        wrap_loader   views + rows + crop boxes, SimCLR batch order kept
   embedding_audit, acquired        audit         reliance of every unit after an epoch
   concept_shares, onset_scores     decide_flags  once, after epoch T0
   flag_units                       compose_views swap in both views from epoch T0 + 1
   unit_typing (evaluation only)            ^
        ^                                   |
 cospro/curves/swap.py              cospro/training/callbacks.py  (ConceptAudit)
   flagged_masks, swap_flagged        calls audit and decide_flags, writes the JSON files
                                            |
                                            v
        <run folder>/concept_curves.json  flagged_units.json      (beside args.json, under outputs/)
                                            |
                                            v
        cospro/cli/summarize_concept_curves.py -> outputs/reports/<study>/concept_curves.md
```

### 4.1 Modules

| File | Content |
|---|---|
| `cospro/curves/core.py` | Pure tensor functions: units, regions, masks, the embedding audit, shares, onset scores, flag rules, unit typing, AUROC, Spearman. Reads no label except in `unit_typing`. |
| `cospro/curves/swap.py` | `flagged_masks` and `swap_flagged`, the batch operation of Section 3.6. |
| `cospro/methods/concept_curves.py` | `ConceptCurves`, the training method: loader, audit, flags, swap schedule. |
| `cospro/training/callbacks.py` | `ConceptAudit`: runs the audits at the due epochs, scores them, triggers flagging, writes and restores the JSON files. |
| `cospro/config/training.py` | Options `--curve_*` (section `ConceptCurveOptions`) and their checks. |
| `cospro/cli/summarize_concept_curves.py` | Collects the audits of a study into one report. |
| `scripts/submit_factor_study.sh` | Arms `curves*` (function `add_curve_arms`). |
| `tests/test_concept_curves.py` | Unit tests and method tests on a synthetic MetaShift tree. |

Reused primitives (must stay unchanged): `warp_maps`, `sharpen`, `inpaint`, `composite_images`,
`coactivation_groups` from `cospro/pipeline/concept_maps.py`; `rows_for_subset`;
`IndexedViewBoxDataset` and `RecordedTwoCropTransform`, which record each view's crop box and flip.

### 4.2 Options

| Option | Default | Meaning |
|---|---|---|
| `--splice_mode concept_curves` | | selects the method |
| `--curve_concept_maps` | required | the `concept_maps.pt` of the dataset |
| `--curve_unit_count` | 12 | concept units; 0 uses single factors |
| `--curve_early_epochs` | 50 | early window `T0` |
| `--curve_audit_every` | 5 | audit interval inside the early window |
| `--curve_late_audit_every` | 50 | audit interval afterwards; 0 disables it |
| `--curve_statistic` | `shift` | reliance statistic for shares and flags: `shift` or `margin` |
| `--curve_flag_rule` | `top` | `none`, `top`, `bottom`, `random`, `units`, `oracle` |
| `--curve_tau` | 0.5 | share of the onset scores the flagged units reach |
| `--curve_max_flagged_area` | 0.75 | area cap of the flagged regions |
| `--curve_flag_units` | | unit ids for the rule `units` |
| `--curve_flag_seed` | 0 | seed of the rule `random` |
| `--curve_swap_probability` | 0.5 | `q` |
| `--curve_swap_warmup_epochs` | 10 | linear ramp of `q` after `T0` |
| `--curve_swap_views` | `both` | `both` or `second` |
| `--curve_shuffle_maps` | false | control: another image's regions |

The `curve_*` options stay out of the storage name of every other method, so existing runs keep their
folders.

### 4.3 File formats

`concept_curves.json`, written after every audit into the run's training folder:

```json
{
  "dataset": "metashift", "seed": 1, "concept_maps": "<path>", "statistic": "shift",
  "unit_count": 12, "early_epochs": 50,
  "unit_of_factor": [0, 3, 3, 7],
  "units": [{"unit": 0, "name": "lawn / pasture / grazing", "factors": ["..."],
             "presence": 0.41, "area": 0.33, "audited": true}],
  "epochs": [0, 5, 10],
  "shift": [[0.02, 0.03]], "margin": [[0.01, 0.02]], "share": [[0.0, 0.0]],
  "onset": [0.45, 0.55],
  "evaluation_only": {"typing": [{"unit": 0, "u_attribute": 0.2, "u_class": 0.01, "type": "attribute"}],
                      "detection_auroc": 0.9}
}
```

`shift`, `margin` and `share` are indexed `[audit position][unit]`; `onset` is indexed by unit. The
block `evaluation_only` comes from hidden class and attribute labels (`cospro.diagnostics.labels`),
as the factor-learnability record of SCD does. Training and flagging must never read it, except for
the `oracle` rule.

`flagged_units.json`, written once after epoch `T0`: `{"rule", "tau", "flagged": [unit ids],
"names", "types", "decided_after_epoch", "mean_flagged_area", "shuffled_maps"}`.

A resumed run reads both files back, so the flags survive a restart.

Training metrics added to the epoch report: `relational_curve_swap_probability` and
`relational_curve_swapped_fraction` (the trainer prefixes method diagnostics with `relational_`),
plus `Concept audit leading share` and `Concept audit detection AUROC` at audit epochs.

## 5. Commands

All training runs on the cluster. Concept maps of MetaShift exist from the spatial runs; the study
launcher builds them first when they are missing.

```bash
# Kill test, both stages in one submission: 5 arms x 4 seeds = 20 jobs, at most 8 at once
ARMS="curves curves_random curves_bottom curves_shuffled curves_oracle" bash scripts/submit_factor_study.sh metashift 1 2 3 4

# Optional: the audit under plain SimCLR for all 500 epochs
ARMS="curves_audit" bash scripts/submit_factor_study.sh metashift 1 2 3 4

# Reports, on any machine after git pull
python -m cospro.cli.summarize_concept_curves --study factors_metashift
```

WGA per arm appears in the study summary `outputs/reports/factors_metashift/summary.md`, beside the
existing `simclr`, `spatial_w5_holdout` and `comp_*` arms.

## 6. Arms

| Arm | Options beyond `--splice_mode concept_curves --curve_concept_maps <maps>` | Purpose |
|---|---|---|
| `curves` | defaults | the method |
| `curves_random` | `--curve_flag_rule random` | tests whether the identity of the flagged units matters |
| `curves_bottom` | `--curve_flag_rule bottom` | tests the direction of the timing signal |
| `curves_shuffled` | `--curve_shuffle_maps true` | tests whether the concept maps matter |
| `curves_oracle` | `--curve_flag_rule oracle` | upper bound with the correct type (reads group labels) |
| `curves_audit` | `--curve_flag_rule none` | audit only; training equals SimCLR |

References already in the study: `simclr` (SimCLR), `spatial_w5_holdout` (SCD) and
`comp_w5_holdout` (compositing with the object prior).

All arms of one seed share the first `T0` epochs, since the swap starts after them.

## 7. Kill test on MetaShift

One submission serves both stages. Stage K1 is readable after epoch 50 of any `curves*` run; stage K2
needs the finished runs.

### Stage K1: does the audit find the spurious concept?

Diagnostics from hidden labels, written into `evaluation_only` and collected by the summary:

- **Unit typing.** For the presence indicator `P_k` of each audited unit:
  `u_attribute = I(P_k; a | y) / H(P_k)` and `u_class = I(P_k; y | a) / H(P_k)`. A unit is an
  *attribute unit* when `u_attribute >= 0.05` and `u_attribute >= 2 * u_class`, a *class unit* in the
  mirrored case, and *untyped* otherwise.
- **Detection AUROC**: how well the onset score separates attribute units (positive) from class
  units (negative).
- **Seed agreement**: mean Spearman correlation of the onset scores between seeds.

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | AUROC at least 0.7 on three of four seeds, and the flagged set holds at least one attribute unit and no class unit on three of four seeds | the timing signal holds |
| Weak | AUROC between 0.55 and 0.7 | try `--curve_statistic margin`, `--curve_unit_count 0`, `T0` of 20 and 100 |
| Fail | AUROC below 0.55, or the class units lead | the timing hypothesis ends; continue with human-corrected flags (rule `units`) or an alternative of Section 10 |

The typing needs both kinds of units among the 12. If the units mix animals and scenes (many untyped
units), use `--curve_unit_count 0`.

### Stage K2: does the swap raise WGA?

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | `curves` beats `simclr` by at least 5 points of mean validation WGA and beats both `simclr` and `curves_random` on three of four seeds | proceed to Section 8 |
| Partial | `curves_oracle` passes and `curves` fails | the swap works and the audit needs repair |
| Fail | `curves_oracle` ties `simclr` | the swap itself fails in this setting; move to an alternative |

Reference values of the branch on MetaShift (validation WGA, seeds 1 to 4): SimCLR 44.8, spatial
maps 53.0, compositing with the object prior 52.3. MetaShift validation groups hold 72 images, so a
single-seed difference below 7 points carries no weight; extend to seeds 5 to 8 before any claim.

## 8. Plan after the kill test (later)

1. Waterbirds (locked test split with 642 images per minority group) and Spur-CIFAR10. CelebA
   violates spatial separability and serves as the stated limit.
2. Baselines of the setting: SimCLR, LateTVG, LA-SSL, SCD spatial maps and compositing with the
   object prior.
3. Ablations: `K`, `T0`, `tau`, `q`, statistic, one view against both views, area-normalized
   reliance, human-corrected flags, swap from epoch 1 with oracle flags (tests whether the early
   window costs anything).
4. Theory checks: WGA as a function of `q`; the curves after the swap starts (the reliance on the
   flagged units has to fall).

Timeline, 14 weeks from 2026-11-16: weeks 1 to 3 kill test; weeks 4 to 6 method and ablations on
MetaShift and Waterbirds; weeks 7 to 9 baselines and further datasets; weeks 10 to 14 writing.

## 9. Prior work and novelty

Searches on 2026-11-14 and 2026-11-16 (about 20 queries) found no work that combines these three
elements: named patch-level concepts, the timing of the model's reliance on them as the signal for
spuriousness and a region-level swap chosen by that signal. None of the related works below operates
in label-free self-supervised training from scratch. The search was broad and remains
non-exhaustive; repeat it before submission.

| Work | Shared element | Difference |
|---|---|---|
| Background augmentations for SSL (Ryali et al., 2021), DiLo (Zhao et al., 2021) | swap or remove backgrounds in contrastive views | a saliency model decides what the foreground is; no named concepts, no audit of the encoder |
| LateTVG (Hamidieh et al., ICLR 2024) | label-free SSL against spurious features, the same benchmark protocol | prunes late layers for one view; no concepts, no explanation |
| LA-SSL (Zhu et al., 2023) | learning speed as the label-free signal | resamples whole images by learning speed |
| Can contrastive learning avoid shortcuts? (Robinson et al., NeurIPS 2021) | modifies what views share to suppress easy features | adversarial perturbation in latent space; unnamed |
| DISC (Wu et al., ICML 2023) | concept sensitivity at every epoch | supervised; image-level concepts from a concept bank; needs environments |
| Decompose-and-Compose (Noohdani et al., CVPR 2024) | attribution-based image parts recombined across images | supervised; unnamed parts; group-labelled validation decides the causal part |
| MaskTune (Asgari et al., NeurIPS 2022) | masks what the model relies on | supervised; after training |
| Gradient probes on concept decompositions (arXiv 2605.28780) | label-free ranking of concepts by a bias signal | after training; suppression at inference |
| SpLiCE (Bhalla et al., NeurIPS 2024) | sparse named concepts | image-level; explains CLIP itself |
| SCD and compositing (this repository) | concept maps, region swap | CLIP regression target, or a text prior for objects |

## 10. Alternative thesis topics

Each alternative keeps the setting of Section 1 and reuses the concept maps and the audit code.

**A. Concept-conditioned negatives.** Keep the audit and change the negatives of SimCLR: for a share
of the batch, the negatives of an image are images that show the same flagged concept, as the F1
batches of this repository did with image-level factors. Inside such a batch the flagged concept
distinguishes no image from another, so the contrast has to use the rest. No composite images, hence
no seams. Closest prior work: conditional contrastive learning (Tsai et al., 2021), which needs an
annotated attribute. Risk: F1 with image-level factors gave no gain; the patch-level audit changes
which concepts condition the batches. Effort: low, the sampler exists.

**B. Reliance penalty.** Keep the audit and add a loss that lowers the encoder's reliance on the
flagged units directly: the embedding of a view and of the same view with its flagged regions
occluded are pulled together. This is an invariance to the flagged concept without donor content.
Closest prior work: background removal augmentations (Ryali et al., 2021) with a saliency model.
Risk: occlusion fills are out of distribution. Effort: low, the occlusion operator exists.

**C. Concept learning curves as a diagnostic study.** No new training method: measure with the audit
which named concepts SimCLR, LateTVG, LA-SSL, SCD and compositing rely on over training, on four
datasets, and relate the curves to WGA. The safe fallback if no intervention yields a gain, since
the audit code exists. Risk: a thesis without a method gain. Effort: low.

## 11. Risks

| Risk | Sign | Response |
|---|---|---|
| The encoder relies on the object first (large animals) | K1 fails, class units lead | human-corrected flags; alternatives |
| Units mix class and scene factors | many untyped units | `--curve_unit_count 0` or a larger count |
| A direction learned before `T0` persists after the swap starts | reliance on flagged units stays high after epoch 60 | smaller `T0`; larger `q` |
| Concept maps at 14 x 14 are too coarse | `curves_oracle` ties `simclr` | NACLIP-style maps or ViT-B/16 |
| Swap seams act as a shortcut | `curves_shuffled` matches `curves` | alternative B |
| MetaShift noise hides the effect | seed spread above the gain | eight seeds; confirm on the Waterbirds test split |
| The shift statistic at random initialization dominates | shares near uniform | statistic `margin` |

## 12. Implementation status

Implemented and tested locally on 2026-11-16: the audit, the flag rules, the swap, the training
method, the callback, the options, the study arms and the summary. `tests/test_concept_curves.py`
holds 23 tests; the whole suite passes. One end-to-end run of `spur_splice.py` on a synthetic
MetaShift tree completed with audits, flagging, the swap and the linear probe. No run on real data
has happened yet; the kill test of Section 5 is the next action.

Not verified yet: that a `curves_audit` run reproduces the `simclr` run of the same seed number for
number. The loader keeps SimCLR's sampler and generator and the recorded transform draws as the plain
one does, so the design aims at equality; the first cluster runs confirm or refute it.

## 13. Glossary

- **Concept factor**: a group of SpLiCE dictionary words from the existing grouping stage; one
  channel of the concept maps.
- **Concept unit**: a group of factors that share pixels; the object the audit measures.
- **Hard region**: the map cells a unit wins in one image.
- **Embedding shift** `R_k(t)`: one minus the cosine between the embeddings of an image and of its
  copy with the region of unit `k` occluded, averaged over the images that show the unit.
- **Acquired reliance** `D_k(t)`: the embedding shift minus its value before training.
- **Share** `S_k(t)`: acquired reliance normalized over the audited units.
- **Onset score** `s_k`: mean share over the early window.
- **Flagged set**: the units the audit marks; their regions are swapped.
- **Swap**: exchange of the flagged regions of a view with those of another image.
- **WGA**: the lowest accuracy of the linear probe over the four groups (class, attribute).
