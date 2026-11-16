# Project plan: Concept Learning Curves and Early Concept Swap

Version 2, 2026-11-16. Branch `concept-factors`.

This file is the single source of truth for the thesis project. Every person and every coding model
reads it first and follows it. The theory with full proofs lives in
[`paper/ConceptLearningCurves.tex`](../paper/ConceptLearningCurves.tex); when the two files disagree,
this file wins and the `.tex` gets corrected.

Words with a fixed meaning: **must** marks a binding rule, **default** marks a starting value that an
experiment may change through a command-line option, **later** marks work outside the current stage.

---

## 1. Goal

**Thesis goal.** Raise worst-group accuracy (WGA) on datasets with a spurious attribute, small ones
included (MetaShift with 1,700 training images, Waterbirds with 4,795), using patch-wise concepts
from CLIP and SpLiCE, in a way that is an explainability method and has a measurable gain.

**What the supervisor asked for.**

1. A clear gain over a standard baseline.
2. A real link to explainable AI: the concept tool has to explain the model that is trained.
3. Patch-wise concept gathering stays the core ingredient.

**What ended.** Spatial Concept Distillation regressed the student onto CLIP concept maps. It is
knowledge distillation from CLIP, and its explanation describes the teacher. Its code stays in the
repository as a reference and as the source of the concept maps and the compositing primitives.

**Success criteria of the thesis.**

| Criterion | Measure |
|---|---|
| The audit names the spurious concept without group labels | detection AUROC of the onset score (Section 7) |
| The swap raises WGA over ERM | validation and test WGA, four seeds or more |
| The gain comes from the flagged concepts | the method beats the random-units control |
| The method competes with published group-label-free methods | JTT and Decompose-and-Compose rows in the final table (later) |

## 2. The idea in five sentences

A network learns the easiest predictive feature first, and on these datasets the easiest feature is
the surroundings of the object. Patch-wise concept maps tell where each named concept sits in each
training image. Occluding the region of a concept and measuring how much the model's margin for the
correct class drops gives the **reliance** of the model on that concept, and tracking it over the
first epochs gives one **learning curve** per concept. The concepts that lead the curves are
**flagged** as spurious. Training then restarts, and the flagged regions of each image are exchanged
with flagged regions of other images, so the flagged concepts stop predicting the label while the
rest of the image keeps deciding it.

Roles of the components:

- **SpLiCE and CLIP** supply the names and the locations of concepts. They supply no training target.
- **The audit** (reliance, curves, flags) explains the model under training. It is the xAI
  contribution, and its report is readable by a person.
- **The swap** is the intervention that the audit selects.

## 3. Fixed decisions

These decisions are binding for all implementation work.

| Topic | Decision |
|---|---|
| Setting | Supervised classification with class labels. The self-supervised variant is later work. |
| Group labels | Used for evaluation and for the kill-test diagnostics. Training and flagging never read them. |
| Development dataset | MetaShift (`metashift`). Waterbirds is touched only after the kill test passes. |
| Backbone for the kill test | `resnet18_large` from scratch. `resnet50_pretrained` is the second arm. |
| Concept unit | A group of concept factors that share pixels (`coactivation_groups`, 12 groups, seed 0). |
| Reliance statistic | Mean margin drop under occlusion (Section 4.3). The squared drop is recorded as a secondary statistic. |
| Audit images | The whole training split, evaluation transform, model in evaluation mode. |
| Training protocol | Two runs per seed: the ERM run records the curves; the swap run starts from fresh weights with the same seed. |
| Reported number | Mean of the last four evaluations of a run (evaluations every 5 epochs). |
| Code location | Package `cospro/curves/`, command-line entry points in `cospro/cli/`, launchers in `scripts/`. |
| Existing SSL trainer | Left untouched. The new code imports only shared primitives. |

## 4. Method specification

### 4.1 Inputs

- Training images `x_i` with class labels `y_i`, `i = 1..n`.
- Concept maps: a tensor `[n, F, 14, 14]`, per cell a distribution over `F` concept factors, built by
  `cospro.cli.build_concept_maps` from frozen CLIP ViT-B/32 with ClearCLIP patch embeddings and the
  SpLiCE concept dictionary. The file also holds `sample_ids` (`"<dataset>:<index>"`) and
  `factor_names`. No label enters the maps.

### 4.2 Concept units and regions

1. Group the `F` factors into `K` units with `coactivation_groups(maps, K, seed=0)`. Default `K = 12`.
   `K = 0` selects single factors as units.
2. The unit map of an image is the sum of its factors' maps: `[n, K, 14, 14]`.
3. The **hard region** of unit `k` in image `i` is the set of cells where unit `k` has the largest
   unit-map value. Regions of one image partition its 196 cells.
4. Unit `k` is **present** in image `i` when its hard region holds at least 3 cells.
5. A unit is **audited** when it is present in at least 2 percent of the training images.
6. The name of a unit is the list of its three factors with the largest patch share.

### 4.3 Reliance

- **Pixel mask.** The hard region (a 14 x 14 binary map) is upsampled bilinearly to the image size
  and passed through `sharpen` (values up to 0.25 become 0, from 0.75 become 1).
- **Occlusion** `O_k(x)`: the pixels under the mask are replaced by `inpaint`, a normalized
  convolution of the surrounding pixels. The image is in normalized pixel space.
- **Margin** of a model `f` on `(x, y)`: `m(x, y) = f(x)_y - max_{c != y} f(x)_c`.
- **Margin drop** of image `i` and unit `k`: `d_ik = m(x_i, y_i) - m(O_k(x_i), y_i)`.
- **Reliance** at epoch `t`: `D_k(t) = mean of d_ik over the images where k is present`.
  A positive value says that the pixels of unit `k` support the correct class.
- **Secondary statistic**: `R_k(t) = mean of d_ik^2`. A randomly initialized network already has
  `R_k > 0` because any occlusion moves its output, while `D_k` is zero in expectation at
  initialization. This is the reason `D_k` is the primary statistic.
- **Share**: `S_k(t) = max(D_k(t), 0) / sum_j max(D_j(t), 0)` over the audited units.
- **Area**: the mean fraction of the image that unit `k` covers where present. It is recorded, and
  the area-normalized reliance `D_k / area_k` is an ablation. The default leaves area in, since
  area is part of what makes a concept easy (Remark on area in the `.tex`).

### 4.4 Concept learning curves

The ERM run audits after every epoch `t = 1..T0` and afterwards every 10 epochs, plus once before
training (`t = 0`). Default `T0 = 20` for `resnet18_large` and `T0 = 5` for `resnet50_pretrained`.
The curves are the sequences `t -> S_k(t)`.

### 4.5 Flagging

- **Onset score**: `s_k = mean of S_k(t) for t = 1..T0`.
- Sort the audited units by decreasing `s_k`. The flagged set is the shortest prefix whose onset
  scores sum to at least `tau`. Default `tau = 0.5`.
- **Area cap**: a unit that would lift the mean flagged area above 0.75 of the image ends the prefix.
  At least one unit is flagged, and at least one audited unit stays unflagged.
- Control rules, each flagging as many units as `top` does: `bottom` takes them from the end of the
  ranking and `random` draws them uniformly from the audited units with a seed.
- `units` flags an explicit list of unit ids (human-corrected flags).
- `oracle` flags the audited units that the hidden-label typing of Section 7 calls attribute units.
  It reads group labels by design and exists only as the upper-bound control arm.

### 4.6 Early concept swap

The swap run reads the unit assignment and the curves of the ERM run with the same seed, computes
the flagged set and trains from fresh weights. For every batch, on the GPU:

1. `phi_i`: the flagged mask of image `i` on its training view, obtained by warping the union of
   the flagged hard regions with the view's crop box and flip (`warp_maps`) and `sharpen`.
   `kappa_i = 1 - phi_i` is the kept mask.
2. An image is **eligible** when the mean of `kappa_i` lies in `[0.05, 0.95]`. A view is a
   **donor** when the mean of its `phi` is at least 0.25.
3. Each eligible image is swapped with probability `q`. Default `q = 0.5`, constant over training.
4. A swapped image draws a donor `j` from the batch, different from itself, and becomes
   `x~_i = kappa_i * x_i + (1 - kappa_i) * b_j`, where `b_j` is the donor view with its own kept
   region painted over (`composite_images(images, donors, keep=kappa_i, donor_objects=kappa_j)`).
5. The label stays `y_i`.

Control `--shuffle-maps` gives every image the regions of another image.

### 4.7 Why it should work

Inside a model with two Gaussian concepts and a linear readout (proofs in the `.tex`):

- the concept with the larger label covariance leads the learning curves in an explicit time window;
- ERM converges to a predictor with a positive weight on the spurious concept, and WGA falls
  strictly as the ratio of the spurious to the core weight grows;
- swapping the spurious concept with probability `q` lowers the correlation to `(1 - q) * rho`, and
  WGA rises monotonically in `q` up to the optimum of a linear classifier;
- a swap that treats both concepts alike can push WGA below ERM, which is why the audit has to type
  the concepts;
- reweighting keeps `4 p (1 - p)` of the sample (42 percent on MetaShift), the swap keeps all of it.

**The assumption that no proof covers**: the concept learned first is the spurious one. Core and
spurious concepts play symmetric roles in the training distribution, so every method without group
labels assumes something of this kind (JTT, LfF and SPARE assume the same). Stage K1 measures it.

## 5. Architecture

```text
concept maps (.pt, scratch)            MetaShift images + class labels
        |                                        |
        v                                        v
 cospro/curves/core.py                 cospro/curves/training.py
   build_units, hard_regions             CurveDataset (view, row, label, box)
   reliance_audit                        build_classifier, train_epoch, evaluate
   concept_shares, onset_scores                  |
   flag_units, unit_typing                       |
        |                                        |
        +--------------> cospro/cli/train_concept_curves.py <------ cospro/curves/swap.py
                               |                                      swap_flagged
                               v
     outputs/seeds/<study>/seed_NN/<arm>/<attempt>/
        run.json  concept_curves.json  flagged_units.json
                               |
                               v
              cospro/cli/summarize_concept_curves.py
                               |
                               v
        outputs/reports/<study>/summary.json and summary.md
```

### 5.1 Modules

| File | Content |
|---|---|
| `cospro/curves/core.py` | Pure tensor functions: units, regions, masks, reliance audit, shares, onset scores, flagging, unit typing, AUROC. No file access except `load_unit_maps`. |
| `cospro/curves/swap.py` | `swap_flagged`: the batch operation of Section 4.6. |
| `cospro/curves/training.py` | Dataset wrappers, the classifier, one training epoch, group evaluation, optimizer defaults per model. |
| `cospro/cli/train_concept_curves.py` | One run: ERM with audit (`--mode erm`) or swap (`--mode swap`). |
| `cospro/cli/summarize_concept_curves.py` | Aggregates a study: WGA table, detection AUROC, seed agreement, the audit report. |
| `scripts/run_concept_curves.sbatch` | One training job. |
| `scripts/submit_concept_curves.sh` | Submits the kill-test arms with at most 8 jobs at once. |
| `tests/test_concept_curves.py` | Unit tests on synthetic fixtures. |

Reused primitives (must stay unchanged): `warp_maps`, `sharpen`, `inpaint`, `composite_images`,
`coactivation_groups` from `cospro/pipeline/concept_maps.py`; `rows_for_subset` from
`cospro/pipeline/concept_factors.py`; `build_resnet_encoder` from `cospro/models/resnet.py`;
`compute_group_metrics` from `cospro/metrics.py`; the dataset registry; `cospro.tracking.artifacts`.

### 5.2 File formats

`concept_curves.json` (written by an ERM run):

```json
{
  "dataset": "metashift", "seed": 1, "model": "resnet18_large",
  "concept_maps": "<path>", "unit_count": 12, "early_epochs": 20,
  "unit_of_factor": [0, 3, 3, 7],
  "units": [{"unit": 0, "name": "lawn / pasture / grazing", "factors": ["..."],
             "presence": 0.41, "area": 0.33, "audited": true}],
  "epochs": [0, 1, 2],
  "reliance": [[0.01, 0.02]],
  "squared": [[0.10, 0.08]],
  "share": [[0.4, 0.6]],
  "onset": [0.45, 0.55],
  "evaluation_only": {"typing": [{"unit": 0, "u_attribute": 0.2, "u_class": 0.01, "type": "attribute"}],
                      "detection_auroc": 0.9, "top_flags": [0], "top_flag_types": ["attribute"]}
}
```

The block `evaluation_only` is computed from the hidden group labels. Training and flagging must
never read it, except for the `oracle` rule.

`reliance`, `squared` and `share` are indexed `[epoch position][unit]`. `onset` is indexed by unit.

`flagged_units.json` (written by every swap run): `{"rule", "tau", "flagged": [unit ids],
"names": [...], "types": [...], "source_curves": "<path>", "mean_flagged_area": 0.4,
"shuffled_maps": false}`.

`run.json`: `{"config", "status", "train_images", "flags", "training": [{"epoch", "loss",
"accuracy", "swapped_fraction"}], "evaluations": [{"epoch", "val": {"wga", "acc", "group_acc",
"group_counts"}}], "final": {"val_wga", "val_acc", "val_group_acc", "test"?}}`.

Every run, ERM or swap, writes `concept_curves.json` with the same audit schedule, so the reliance
on the flagged units under the swap is on record.

### 5.3 Training defaults

| Option | `resnet18_large` (scratch) | `resnet50_pretrained` |
|---|---|---|
| epochs | 100 | 30 |
| batch size | 64 | 64 |
| optimizer | SGD, momentum 0.9, cosine schedule | the same |
| learning rate | 0.05 | 0.001 |
| weight decay | 5e-4 | 1e-4 |
| early window `T0` | 20 | 5 |
| augmentation | `RandomResizedCrop(224, scale=(0.5, 1))`, horizontal flip | the same |
| mixed precision | `torch.float16` autocast with a `GradScaler` | the same |

### 5.4 Repository rules that apply

- All training runs on the cluster through `sbatch`. Launchers source
  `scripts/load_splice_cluster_env.sh`, call `announce_results`, request one V100, at most 5 CPUs and
  8G per CPU. At most 8 jobs run at once.
- Compact JSON goes under `outputs/`. This project writes no checkpoint: the swap run retrains.
- W&B logging with WGA under the stable key `curves/val/wga` at every evaluation, stored in the run
  summary and in `run.json`.
- Text in English, lists as "A, B and C", the term "spurious attribute" or "spurious concept".

## 6. Commands

Concept maps of MetaShift exist on the cluster (130 LAION factors, band 2 percent). The submit script
finds them at the path `submit_factor_study.sh` uses; `MAPS=<path>` overrides it.

```bash
# Stage K1: ERM with the audit, seeds 1 to 4
STAGE=k1 bash scripts/submit_concept_curves.sh metashift 1 2 3 4

# Stage K2: swap arms, each seed reads the curves of its own ERM run
STAGE=k2 bash scripts/submit_concept_curves.sh metashift 1 2 3 4

# Summary (reads JSON only, runs on any machine after git pull)
python -m cospro.cli.summarize_concept_curves --study curves_metashift
```

## 7. Kill test on MetaShift

The test has two stages. A failed stage ends the stage's hypothesis as written below.

### Stage K1: does the audit find the spurious concept?

Runs: arm `erm`, seeds 1 to 4, `resnet18_large`. Second arm `erm_rn50`, the same with
`resnet50_pretrained`.

Diagnostics, computed by each ERM run from the hidden labels (block `evaluation_only`) and collected
by the summary:

- **Unit typing.** For the presence indicator `P_k` of each audited unit:
  `u_attribute = I(P_k; a | y) / H(P_k)` and `u_class = I(P_k; y | a) / H(P_k)`. A unit is an
  *attribute unit* when `u_attribute >= 0.05` and `u_attribute >= 2 * u_class`, a *class unit* in the
  mirrored case, and *untyped* otherwise.
- **Detection AUROC**: how well the onset score separates attribute units (positive) from class
  units (negative).
- **Seed agreement**: mean Spearman correlation of the onset scores between seeds.

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | AUROC at least 0.7 on three of four seeds, and the flagged set holds at least one attribute unit and no class unit on three of four seeds | go to K2 |
| Weak | AUROC between 0.55 and 0.7 | run K2 with the oracle and the method side by side, and try `K = 0` (single factors) and the squared statistic |
| Fail | AUROC below 0.55 or the class units lead | the timing hypothesis ends; continue with human-corrected flags (rule `units`) or an alternative of Section 10 |

### Stage K2: does the swap raise WGA?

Arms, seeds 1 to 4, all with identical training settings:

| Arm | Flag rule | Purpose |
|---|---|---|
| `erm` | none | reference (reused from K1) |
| `swap` | `top` | the method |
| `swap_random` | `random` | tests whether the identity of the flagged units matters |
| `swap_bottom` | `bottom` | tests the direction of the timing signal |
| `swap_shuffled` | `top` with `--shuffle-maps` | tests whether the concept maps matter |
| `swap_oracle` | `oracle`, the attribute units of the typing | upper bound with the correct type |

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | `swap` beats `erm` by at least 5 points of mean validation WGA and beats both `erm` and `swap_random` on three of four seeds | proceed to Section 8 |
| Partial | `swap_oracle` passes and `swap` fails | the swap works and the audit needs repair: tune `T0`, `tau`, `K`, the statistic |
| Fail | `swap_oracle` ties `erm` | the swap itself fails in this setting; move to an alternative of Section 10 |

MetaShift validation groups hold 72 images, so a single-seed difference below 7 points carries no
weight. Extend to seeds 5 to 8 before any claim.

## 8. Plan after the kill test (later)

1. Waterbirds, Spur-CIFAR10 and one dataset with large evaluation groups; locked test splits.
2. Baselines under the same code: JTT, Decompose-and-Compose, DFR (the reference with group labels).
   A last-layer variant of the swap (retrain only the classifier on swapped images) is an ablation.
3. Ablations: `K`, `T0`, `tau`, `q`, area-normalized reliance, squared statistic, donor fill in the
   audit, continue-versus-restart, an early-only schedule for `q`, human-corrected flags.
4. Theory checks: WGA as a function of `q`; a synthetic two-concept dataset that satisfies the
   assumptions, on which the formulas are verified end to end.
5. The self-supervised variant: the swapped image as the second SimCLR view.

Timeline, 14 weeks from 2026-11-16: weeks 1 to 3 kill test; weeks 4 to 6 method and ablations on
MetaShift and Waterbirds; weeks 7 to 9 baselines and further datasets; weeks 10 to 14 writing.

## 9. Prior work and novelty

Searches on 2026-11-14 and 2026-11-16 (about 20 queries) found no work that combines these three
elements: named patch-level concepts, the timing of the model's reliance on them as the signal for
spuriousness and a region-level swap chosen by that signal. The search was broad and remains
non-exhaustive; repeat it before submission.

| Work | Shared element | Difference |
|---|---|---|
| DISC (Wu et al., ICML 2023) | concept sensitivity at every epoch, concept-aware augmentation | image-level concepts from a concept image bank, needs environments, mixes whole concept images |
| Decompose-and-Compose (Noohdani et al., CVPR 2024) | attribution-based image parts recombined across images; last-layer retraining on the composites | unnamed parts from the model's own attribution; a group-labelled validation set decides which part is causal |
| MaskTune (Asgari et al., NeurIPS 2022) | masks what the model relies on | after training, one epoch, masks every salient region |
| JTT, LfF, SPARE | early-training signal | reweight whole examples; limited by the number of minority images |
| GALS (Petryk et al., CVPR 2022) | CLIP attention guides a classifier | a fixed class prompt is the prior |
| RaVL (Varma et al., NeurIPS 2024) | region-level discovery and loss | fine-tuned vision-language models, unnamed region clusters |
| SS-CA (arXiv 2511.12100) | attribution-guided counterfactual augmentation | no named concepts, no training-time order, average accuracy on ImageNet variants |
| Gradient probes on concept decompositions (arXiv 2605.28780) | label-free ranking of concepts by a bias signal | concepts from a factorization of activations, after training, suppression at inference |
| SpLiCE (Bhalla et al., NeurIPS 2024) | sparse named concepts | image-level; explains CLIP itself |
| SEG-MIL-CBM (arXiv 2510.04180) | spatially grounded named concepts, 87 percent WGA on Waterbirds | a concept bottleneck on a frozen backbone; no audit of a trained model |

The closest competitor is Decompose-and-Compose. The differences to state in the thesis: the parts
are named, the choice of the part to exchange comes from the learning order and needs no
group-labelled validation set, and the intervention acts during training of the whole network.

## 10. Alternative thesis topics

Each alternative reuses the cached concept maps and the audit code.

**A. Concept-localized attribution penalty.** Keep the audit and replace the swap by a penalty on the
model's input-gradient or Grad-CAM mass inside the flagged regions ("right for the right reasons").
No composite images, so no seams. In the linear model the penalty is a ridge term on the spurious
weight alone, which lowers the weight ratio monotonically. Closest prior work: GALS, which uses a
CLIP attention map from a class prompt. Risk: a second backward pass per step and a moderate
difference to GALS. Effort: moderate, shares all audit code.

**B. Patch-concept bottleneck classifier.** Classify from patch-wise SpLiCE codes: each patch is a
sparse vector over named concepts, attention pooling aggregates the patches and a linear layer
predicts the class. Removing a concept is an edit of one weight column, and the audit chooses the
concepts to remove. The backbone is frozen CLIP, so 1,700 images suffice. Risk: SEG-MIL-CBM is close
and strong. Effort: low to moderate.

**C. Concept learning curves as a diagnostic study.** No new training method: measure with the audit
which named concepts ERM, JTT, DFR, LateTVG and SimCLR rely on over training, on four datasets, and
relate the curves to WGA. This is the safe fallback if no intervention yields a gain, since the audit
code exists after stage K1. Risk: a thesis without a method gain. Effort: low.

A fourth option, retraining only the last layer on swapped images, is covered by
Decompose-and-Compose and therefore serves as an ablation of the main method.

## 11. Risks

| Risk | Sign | Response |
|---|---|---|
| ERM from scratch learns little on 1,700 images | validation accuracy near 55 percent | use `resnet50_pretrained` as the main backbone |
| The audit flags the class concept | K1 fails | human-corrected flags; alternatives |
| Concept maps at 14 x 14 are too coarse | `swap_oracle` ties `erm` | NACLIP-style maps or ViT-B/16; alternative B |
| Swap seams act as a shortcut | `swap_shuffled` matches `swap` | donor-fill blending; alternative A |
| MetaShift noise hides the effect | seed spread above the gain | eight seeds; confirm on the Waterbirds test split |
| Units mix class and scene factors | untyped units dominate | `K = 0` or a larger `K` |

## 12. Implementation status

Implemented and unit-tested locally on 2026-11-16 (`tests/test_concept_curves.py`, 19 tests, one of
them an end-to-end run of both modes and the summary on a synthetic MetaShift tree): the audit, the
flag rules, the swap, the training entry point, the summary and both launchers. No run on real data
has happened yet; stage K1 is the next action.

Open implementation work, in order: (1) run K1 and K2 on the cluster; (2) baselines JTT and
Decompose-and-Compose; (3) the ablation options of Section 8 that have no flag yet (area-normalized
flagging, squared statistic for flagging, donor fill in the audit, continue-after-audit).

## 13. Glossary

- **Concept factor**: a group of SpLiCE dictionary words from the existing grouping stage; one
  channel of the concept maps.
- **Concept unit**: a group of factors that share pixels; the object the audit measures.
- **Hard region**: the map cells a unit wins in one image.
- **Reliance** `D_k(t)`: mean drop of the correct-class margin when the region of unit `k` is occluded.
- **Share** `S_k(t)`: reliance normalized over the audited units.
- **Onset score** `s_k`: mean share over the early window.
- **Flagged set**: the units the audit marks as spurious.
- **Swap**: exchange of the flagged regions between two training images, label kept.
- **WGA**: the lowest accuracy over the four groups (class, attribute).
