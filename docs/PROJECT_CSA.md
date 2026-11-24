# Project 2: Concept Shortcut Attribution (CSA)

Version 2, 2026-11-24: stage 1 finished with every unit neutral, and the attribution at inference
(Section 6) is implemented as the main measurement. Read [`PROJECT_PLAN.md`](PROJECT_PLAN.md) first: it fixes the setting, the
method specification (Section 3), the code architecture (Section 4) and the file formats. This file
holds what is specific to CSA. The sibling project is [`PROJECT_ECS.md`](PROJECT_ECS.md).

## 1. Goal and claim

**Goal.** Explain which named concepts a SimCLR encoder uses as shortcuts, and back every statement
with an intervention on that concept. This is an explainable-AI project: its product is an
explanation, and WGA serves as the evidence that the explanation is true.

**The sentence CSA produces, per concept:**

> "The encoder starts to rely on *sofa / indoor* within the first 20 epochs (onset rank 1 of 9,
> reliance share 31 percent). Swapping this concept alone during training raises WGA by 2.4 points
> (up in 4 of 4 seeds). It is a shortcut."

and for a core concept:

> "The encoder relies on *fur / paw / whisker* later (onset rank 6 of 9). Swapping it alone lowers
> WGA by 5 points. The encoder needs it."

**Claim to establish.** Three parts, each tested separately (Section 4):

1. *Attribution by intervention.* Swapping a single concept unit changes WGA by a measurable amount,
   and the sign of the change separates shortcut concepts (WGA rises) from needed concepts (WGA
   falls).
2. *Learning order as a label-free marker.* The earlier the encoder relies on a concept, the larger
   the WGA gain from swapping it.
3. *Faithfulness of the reliance curve.* After the swap of a unit starts, the measured reliance on
   that unit falls: the curve reports what the encoder really uses.

**Difference to ECS.** ECS asks "does the automatic selection raise WGA?" and reports one number.
CSA asks "what does each concept do?" and reports one row per concept. CSA makes no claim of a
state-of-the-art WGA. CSA stays valid when the learning order fails as a marker (part 2 becomes a
negative finding) as long as part 1 holds.

## 2. Quantities

All defined per concept unit `k`. Units, regions, reliance and onset are specified in Plan, Sections
3.2 to 3.5.

| Quantity | Meaning | Needs labels |
|---|---|---|
| Name | the three factors with the largest map mass | no |
| Presence, area | share of images that show the unit, mean share of an image it covers | no |
| Curve `S_k(t)` | share of the acquired reliance at audit epoch `t` | no |
| Onset score `s_k`, onset rank | mean share over epochs 1 to 50; rank 1 is the earliest concept | no |
| **WGA effect `E_k`** | `WGA(curves_unit_k) - WGA(simclr)`, the mean over seeds of the paired difference, in points | group labels of the probe and validation splits, as every WGA of the study |
| Accuracy effect | the same difference for mean accuracy | class labels of the probe |
| Hidden-label type | attribute, class or untyped unit (Plan, Section 7) | yes, evaluation only |

The run `curves_unit_k` trains as SimCLR for 50 epochs and from epoch 51 swaps the regions of unit
`k` alone (rule `units`, Plan, Section 3.6). All arms of one seed share the first 50 epochs, so `E_k`
isolates the effect of the swap. Training of every arm stays label-free; labels enter only when WGA
is read, exactly as for the SimCLR baseline.

**Shortcut verdict of a unit:** *shortcut* when `E_k > 0` beyond noise, *needed* when `E_k < 0`
beyond noise, *neutral* otherwise. "Beyond noise" means: the same sign on at least three of four
seeds and `|E_k|` at least twice its standard error (`sd / sqrt(seeds)`).

## 3. Experiment on MetaShift

```bash
# Stage 1 (done 2026-11-24): the nine audited unit arms x 4 seeds = 36 jobs, at most 8 at once
ARMS="curves_unit_0 curves_unit_1 curves_unit_2 curves_unit_3 curves_unit_4 curves_unit_5 curves_unit_7 curves_unit_10 curves_unit_11" \
  bash scripts/submit_factor_study.sh metashift 1 2 3 4

# Stage 3, attribution at inference (Section 6): eight audit-only SimCLR runs that keep their encoder, then one
# attribution job per seed after its training; at most 8 jobs run at once
TRAIN=1 MAX_PARALLEL=8 bash scripts/submit_concept_attribution.sh metashift 1 2 3 4 5 6 7 8

# Report, on any machine after git pull
python -m cospro.cli.summarize_concept_curves --study factors_metashift
```

**Result of stage 1 (2026-11-24).** Outcome *weak*: every unit is neutral. The unit-by-seed F test gives
p = 0.64 at a run-to-run sd of 4.1 points, so which unit is swapped explains no more WGA than the noise between
runs does. H3 holds on 9 of 9 units. Stage 2, more seeds for the training-time arms, is dropped: a 2-point effect
would need about 30 seeds. Details: [`notes/ECS_CSA_RESULTS_2026_11_24.md`](notes/ECS_CSA_RESULTS_2026_11_24.md).

The reference arm `simclr` exists for seeds 1 to 4 in `factors_metashift`; submit it for every
further seed that unit arms use (`ARMS="simclr"`). When `simclr` is missing, the report falls back to
`curves_audit`.

The report `outputs/reports/factors_metashift/concept_curves.md` holds the section **Concept
attribution**. It opens with the outcome and the verdicts of H1 to H3, then the unit-by-seed F test. Then
comes one row per audited unit:
- onset rank, words, onset score, area and hidden-label type;
- WGA effect, its standard deviation over seeds, the number of seeds with a gain and the verdict;
- the change against the mean of all unit arms of the same seed (*vs unit arms*);
- the accuracy effect;
- the unit's own share at epoch 50 and at the last audit, against its reference.

The Spearman correlation of onset score and WGA effect across units also appears. The same data sits under
the key `attribution` of `concept_curves.json`. Units that fewer than 2 percent of the images show are left
out. The H3 reference is the share in the audit-only arm `csa_audit` of the same seed when that arm exists,
otherwise the unit arm's own share at epoch 50.

Before submitting, check which units are audited: the log of any finished `curves*` run lists every
unit with presence and area (`[INFO] Concept unit ...`). Unit arms of unaudited units train as near
SimCLR and are skipped by the report, so leave them out of later stages.

## 4. Hypotheses and criteria

| Id | Hypothesis | Measured by | Holds when |
|---|---|---|---|
| H1 | Attribution by intervention | `E_k` per unit and its sign against the hidden-label type | at least one attribute unit is a *shortcut* and at least one class unit is *needed* by the verdict of Section 2 |
| H2 | Learning order marks shortcuts | Spearman correlation of `s_k` and `E_k` across audited units | at least 0.5 |
| H3 | The reliance curve is faithful | share `S_k` at the late audits of `curves_unit_k` against `curves_audit` or the pre-swap value at epoch 50 | the share of the swapped unit falls after epoch 60 in at least three of four seeds |

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | H1 and H3 hold | CSA stands. H2 decides whether the learning order is reported as a marker (holds) or as a negative finding (fails) |
| Weak | effects have the predicted signs and stay inside noise at four seeds | seeds 5 to 8 for the top units; repeat on Waterbirds, whose test groups are larger |
| Fail | every `E_k` is neutral at eight seeds | the training-time swap of one unit is too weak a lever; move to attribution at inference (Section 6) |

Noise budget: the standard deviation of WGA over seeds is about 4 points on MetaShift. With four
seeds the standard error of a paired difference is about 2 points when pairing removes little
variance, so an effect of 2 points needs eight seeds or the Waterbirds test split. Effects of 4
points and more are readable at four seeds.

## 5. Theory

The two-concept model of `paper/ConceptLearningCurves.tex` predicts the pattern CSA tests:

- Theorem "Early concept swap": swapping the spurious concept raises the WGA of the balanced probe,
  monotonically in `q`. Prediction: `E_k > 0` for attribute units.
- Proposition "Swapping the core concept": swapping the core concept lowers WGA. Prediction: `E_k < 0` for class
  units.
- Theorems "Learning order" and "Order of the learning curves": the concept with the larger second moment is learned first. Prediction
  for H2, valid under the assumption that this concept is the spurious one.

So the sign of `E_k` types a concept, and this typing holds without the assumption that H2 needs.
The `.tex` treats one flagged concept; a section on `E_k` as a function of the swapped concept, with
several spurious units sharing the attribute, is open theory work (later).

## 6. Attribution at inference (implemented 2026-11-24)

**Why.** A training-time swap compares two runs that share only their first 50 epochs. The 450 epochs
after that decorrelate them, so the effect of one unit drowns in the 4-point run-to-run noise (stage 1).
Occlusion at inference compares clean and occluded copies of the same validation images inside one encoder
and one probe, which removes that noise.

**Measurement**, per encoder (one `csa_audit` run per seed) and per audited unit `k`:

1. Fit the run's logistic probe on the clean features of the probe training split. It uses the same split,
   solver, seed and RNG order as training (`evaluate_probe`).
2. Occlude the hard region of unit `k` in every validation image that shows the unit. The regions and the
   inpainting are those of the audit, and the units come from the run's own `concept_curves.json`, so
   unit `k` is the unit of `curves_unit_<k>`.
3. **Control:** occlude the same images with the region that unit `k` holds in another validation image
   that shows it. That region has the same kind of shape and size and sits where the concept does not.
4. Classify clean, occluded and control features with the one fitted probe. Record WGA, mean accuracy and
   the accuracy of each group.

**Quantities**, in points of validation accuracy:
- `change`: occluded minus clean.
- `control_change`: control minus clean.
- **`net_change`**: occluded minus control, the effect of the concept itself.
- `embedding_shift`: the mean cosine shift of the encoder features, with its control.

**Verdicts** use the rule of Section 2 on the net WGA change over the seeds: *shortcut* when occluding
the concept raises WGA, *needed* when it lowers it.

| Id | Hypothesis | Holds when |
|---|---|---|
| H1-inf | Attribution by occlusion | at least one attribute unit is a *shortcut* and at least one class unit is *needed* |
| H2-inf | Learning order marks the units the probe uses | Spearman of onset score and net WGA change at least 0.5 |
| Agreement | Both attributions name the same units | Spearman of the net change at inference and the training-time `E_k` (reported, no threshold) |

**Code:**
- `cospro/curves/attribution.py`: `occlusion_features` and `control_donors`.
- `cospro/cli/attribute_concepts.py`: one run, writes `concept_attribution_val.json` next to `run.json`.
- `scripts/attribute_concepts.sbatch` and `scripts/submit_concept_attribution.sh`: one job per seed, run one
  after another. With `TRAIN=1` each job waits for its seed's training.
- `cospro.cli.build_concept_maps --like <training maps> --split val`: the validation concept maps, which
  the first attribution job builds when they are missing.
- The section **Concept attribution at inference** of the report.
- The arm `csa_audit` of `scripts/submit_factor_study.sh`: audit-only SimCLR, so training equals `simclr`.
  It keeps `last.pth` on scratch through `--keep_checkpoints --save_freq 500`.

The test `test_attribution_at_inference_end_to_end` trains such a run on a synthetic MetaShift tree,
attributes it and summarizes it.

## 7. Later work (not implemented)

1. **Attribution at inference over training time.** The same occlusion on the encoder at epoch 50,
   which needs that checkpoint kept, to tell "learned early" and "used by the probe" apart in one
   encoder.
2. **Audit per layer.** The embedding shift read after each residual stage of the ResNet-18, which
   answers where in depth a concept is used, beside when in training.
3. **Leave-one-out.** Swap the ECS flagged set without unit `k`, for the interaction between units.
4. **Waterbirds**, then the comparison of the attribution tables of SimCLR, SCD and compositing.

## 8. Prior work

Searched on 2026-11-16 (three queries on top of the ECS searches; non-exhaustive, repeat before
submission). No work found that reports, for a label-free self-supervised encoder trained from
scratch, per named patch-level concept both the learning order and the WGA effect of removing that
concept during training.

| Work | Shared element | Difference |
|---|---|---|
| ROAR, remove and retrain (Hooker et al., NeurIPS 2019) | validates an attribution by retraining without the attributed input | pixels of a supervised model; no named concepts, no spurious attribute |
| Debugging concept bottleneck models through removal and retraining (ICLR 2026) | removes spurious concepts and retrains; WGA on Waterbirds and MetaShift | supervised concept bottleneck models; experts pick the concepts |
| SpLiCE (Bhalla et al., NeurIPS 2024) | named concepts; removing background concepts on Waterbirds | image-level, explains CLIP, concepts picked by hand |
| DISC (Wu et al., ICML 2023) | concept sensitivity tracked over epochs | supervised; image-level concept bank; needs environments |
| Bias Leaves a Gradient Trail (arXiv 2605.28780) | label-free ranking of concepts by a bias signal | frozen supervised models; after training |
| Representation Is Not Reliance (ICML 2026) | concept-based causal diagnostics of shortcut learning | known from its title and abstract only; **read it before any novelty claim** |
| LA-SSL (Zhu et al., 2023) | learning speed in SSL as the signal for spurious correlation | per example, unnamed; resampling |

## 9. Risks specific to CSA

| Risk | Sign | Response |
|---|---|---|
| Single-unit effects stay inside noise | all verdicts neutral | more seeds for top units; Waterbirds; inference attribution |
| The attribute is spread over several units, each swap leaves the others | small `E_k` for every attribute unit, large gain for `curves_oracle` | fewer units (`CURVE_UNITS=6`), or report effects of unit pairs |
| Units mix class and scene factors | many untyped units | `--curve_unit_count 0` or a larger count |
| The swap changes WGA through its seams | `curves_shuffled` of ECS shows the same effect as the unit arms | compare every `E_k` with the effect of `curves_shuffled` |
| 48 runs exceed the cluster budget | queue time | stage 1 with two seeds; drop unaudited and tiny units |

## 10. Status and next steps

Implemented and tested locally (branch `ECS`, 2026-11-24):
- the unit arms `curves_unit_<k>` with the alias `curves_units`;
- the attribution table with verdicts, unit contrast, F test, H1 to H3 and outcome
  (`attribution`, `effect_verdict`, `unit_contrast`, `csa_hypotheses` and `unit_share_change` in
  `cospro/cli/summarize_concept_curves.py`);
- the attribution at inference of Section 6 with its launchers and the arm `csa_audit`.

Stage 1 ran on real data (36 runs). The attribution at inference has run only on the synthetic tree.

Next steps, in order:

1. Submit stage 3 of Section 3: `TRAIN=1 MAX_PARALLEL=8 bash scripts/submit_concept_attribution.sh metashift 1 2 3 4 5 6 7 8`.
2. Check the first `concept_attribution_val.json`: `probe.reference_wga` should match `probe.run_last_wga`, and
   `clean_feature_max_difference` should be near 0.
3. Read the section **Concept attribution at inference** and judge H1-inf and H2-inf. Record the verdicts in a
   dated note under `docs/notes/`.
4. On H1-inf: repeat on Waterbirds (`add_curve_arms` holds `csa_audit` there too) and write the per-concept
   sentences of Section 1 from the net changes.
