# Project 2: Concept Shortcut Attribution (CSA)

Version 1, 2026-11-16. Read [`PROJECT_PLAN.md`](PROJECT_PLAN.md) first: it fixes the setting, the
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
# Stage 1: 12 unit arms x 2 seeds = 24 jobs, at most 8 at once
ARMS="curves_units" bash scripts/submit_factor_study.sh metashift 1 2

# Stage 2: seeds 3 and 4 for all units, then seeds 5 to 8 for the units with the largest |E_k|
ARMS="curves_units" bash scripts/submit_factor_study.sh metashift 3 4
ARMS="curves_unit_3 curves_unit_7" bash scripts/submit_factor_study.sh metashift 5 6 7 8

# Report, on any machine after git pull
python -m cospro.cli.summarize_concept_curves --study factors_metashift
```

The reference arm `simclr` exists for seeds 1 to 4 in `factors_metashift`; submit it for every
further seed that unit arms use (`ARMS="simclr"`). When `simclr` is missing, the report falls back to
`curves_audit`.

The report `outputs/reports/factors_metashift/concept_curves.md` opens with the section **Concept
attribution**: one row per audited unit with onset rank, words, onset score, area, hidden-label type,
WGA effect, its standard deviation over seeds, the number of seeds with a gain and the accuracy
effect, plus the Spearman correlation of onset score and WGA effect across units. The same data sits
under the key `attribution` of `concept_curves.json`. Units that fewer than 2 percent of the images
show are left out.

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
| Fail | every `E_k` is neutral at eight seeds | the training-time swap of one unit is too weak a lever; move to attribution at inference (Section 6, step 1) |

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

## 6. Later work (not implemented)

1. **Attribution at inference.** Occlude unit `k` in the validation images, feed them to the frozen
   encoder and probe and read the change of each group's accuracy. Needs concept maps of the
   validation split and a probe evaluation on occluded images. No training, so it costs minutes per
   run and gives a second, independent attribution for the same units.
2. **Audit per layer.** The embedding shift read after each residual stage of the ResNet-18, which
   answers where in depth a concept is used, beside when in training.
3. **Leave-one-out.** Swap the ECS flagged set without unit `k`, for the interaction between units.
4. **Waterbirds**, then the comparison of the attribution tables of SimCLR, SCD and compositing.

## 7. Prior work

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

## 8. Risks specific to CSA

| Risk | Sign | Response |
|---|---|---|
| Single-unit effects stay inside noise | all verdicts neutral | more seeds for top units; Waterbirds; inference attribution |
| The attribute is spread over several units, each swap leaves the others | small `E_k` for every attribute unit, large gain for `curves_oracle` | fewer units (`CURVE_UNITS=6`), or report effects of unit pairs |
| Units mix class and scene factors | many untyped units | `--curve_unit_count 0` or a larger count |
| The swap changes WGA through its seams | `curves_shuffled` of ECS shows the same effect as the unit arms | compare every `E_k` with the effect of `curves_shuffled` |
| 48 runs exceed the cluster budget | queue time | stage 1 with two seeds; drop unaudited and tiny units |

## 9. Status and next steps

Implemented and tested locally: the unit arms `curves_unit_<k>` with the alias `curves_units` in
`scripts/submit_factor_study.sh`, and the attribution table in
`cospro/cli/summarize_concept_curves.py` (functions `probe_results`, `attribution`,
`render_attribution`; test `test_attribution_pairs_single_unit_arms_with_simclr_by_seed`). No run on
real data yet.

Next steps, in order:

1. Submit stage 1 of Section 3 together with the ECS kill test.
2. Read the audit after epoch 50: list of audited units with names, onset ranks and types.
3. After the runs finish, read the attribution table and judge H1 and H2.
4. Implement the H3 check in the summary: per unit arm, the share of the swapped unit at epoch 50
   against its share at the late audits (data already in `concept_curves.json`).
5. Record the verdicts in a dated note under `docs/notes/` and follow the outcome table of Section 4.
