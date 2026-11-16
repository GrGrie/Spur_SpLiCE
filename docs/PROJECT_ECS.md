# Project 1: Early Concept Swap (ECS)

Version 1, 2026-11-16. Read [`PROJECT_PLAN.md`](PROJECT_PLAN.md) first: it fixes the setting, the
method specification (Section 3), the code architecture (Section 4) and the file formats. This file
holds what is specific to ECS. The sibling project is [`PROJECT_CSA.md`](PROJECT_CSA.md).

## 1. Goal and claim

**Goal.** Raise the worst-group accuracy (WGA) of a SimCLR encoder trained without any label, on
datasets with a spurious attribute.

**Claim to establish.** The concepts that SimCLR relies on first are the spurious ones. Swapping the
regions of these concepts between images, from the end of an early window on, raises WGA over SimCLR.
The selection uses the encoder's own learning order and no label.

**What counts as the result.** One number per dataset and arm: validation WGA of the linear probe
(mean of the last four probes), ECS against SimCLR and against the control arms.

**Role of explainability.** The audit explains the selection ("these concepts were flagged because
the encoder leaned on them first"). In ECS this explanation is a by-product; the headline is the gain.

## 2. Method in one run

1. Epochs 1 to 50 train as plain SimCLR. The audit measures the reliance on every concept unit at
   epoch 0 and every 5 epochs (Plan, Sections 3.3 and 3.4).
2. After epoch 50 the rule `top` flags the units that led the curves (Plan, Section 3.5).
3. From epoch 51 the flagged regions are swapped between images in both views with probability 0.5
   after a 10-epoch ramp (Plan, Section 3.6).
4. The linear probe and the WGA report stay as in every run of the study.

Arm name: `curves`. Options: the defaults of `--splice_mode concept_curves`.

## 3. Kill test on MetaShift

```bash
ARMS="curves curves_random curves_bottom curves_shuffled curves_oracle" bash scripts/submit_factor_study.sh metashift 1 2 3 4
python -m cospro.cli.summarize_concept_curves --study factors_metashift
```

20 jobs, at most 8 at once. Stage K1 is readable after epoch 50 of any `curves*` run; stage K2 needs
the finished runs.

### Stage K1: does the audit find the spurious concept?

Read from `outputs/reports/factors_metashift/concept_curves.md` (diagnostics defined in Plan,
Section 7).

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | detection AUROC at least 0.7 on three of four seeds, and the flagged set holds at least one attribute unit and no class unit on three of four seeds | the timing signal holds |
| Weak | AUROC between 0.55 and 0.7 | try `--curve_statistic margin`, `--curve_unit_count 0`, `T0` of 20 and 100 |
| Fail | AUROC below 0.55, or the class units lead | the timing hypothesis ends; ECS continues only with human-corrected flags (rule `units`), and the effort moves to CSA |

### Stage K2: does the swap raise WGA?

Read from `outputs/reports/factors_metashift/summary.md`.

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | `curves` beats `simclr` by at least 5 points of mean validation WGA and beats both `simclr` and `curves_random` on three of four seeds | proceed to Section 5 |
| Partial | `curves_oracle` passes and `curves` fails | the swap works and the selection needs repair; the per-unit effects of CSA show which units to flag |
| Fail | `curves_oracle` ties `simclr` | the swap itself fails in this setting; ECS ends |

What each control arm tells:

| Arm | If it matches `curves` |
|---|---|
| `curves_random` | the identity of the flagged units carries no information |
| `curves_bottom` | the direction of the timing signal carries no information |
| `curves_shuffled` | the concept maps carry no information; the gain comes from the swap as an augmentation |
| `curves_oracle` | upper bound: the audit already selects as well as hidden labels do |

## 4. Theory

Proofs in `paper/ConceptLearningCurves.tex`, summarized in Plan, Section 3.7. The statements ECS
relies on: SimCLR's surrogate learns the fused direction first; the audit ranks the spurious concept
first when its second moment is the larger one; the swap in both views turns the learned direction
towards the core concept and raises WGA monotonically in `q`; swapping the core concept lowers WGA.

The assumption without proof is that the spurious concept has the larger second moment. Stage K1
measures it.

## 5. Plan after the kill test (later)

1. Waterbirds (locked test split with 642 images per minority group) and Spur-CIFAR10. CelebA
   violates spatial separability and serves as the stated limit.
2. Baselines of the setting: SimCLR, LateTVG, LA-SSL, SCD spatial maps and compositing with the
   object prior.
3. Ablations: `K`, `T0`, `tau`, `q`, statistic, one view against both views, area-normalized
   reliance, human-corrected flags, swap from epoch 1 with oracle flags (tests whether the early
   window costs anything).
4. Theory checks: WGA as a function of `q`; the curves after the swap starts (the reliance on the
   flagged units has to fall).

## 6. Risks specific to ECS

| Risk | Sign | Response |
|---|---|---|
| The encoder relies on the object first (large animals) | K1 fails, class units lead | human-corrected flags; CSA |
| A direction learned before `T0` persists after the swap starts | reliance on flagged units stays high after epoch 60 | smaller `T0`; larger `q` |
| Swap seams act as a shortcut | `curves_shuffled` matches `curves` | alternative B of the Plan, Section 10 |
| MetaShift noise hides the effect | seed spread above the gain | eight seeds; confirm on the Waterbirds test split |

Prior work: Plan, Section 9.

## 7. Status and next steps

Implemented and tested locally: everything ECS needs. No run on real data yet.

Next steps, in order:

1. Submit the kill test of Section 3 on the cluster.
2. After epoch 50 of the first runs, read stage K1.
3. After the runs finish, read stage K2 and record both verdicts in a dated note under `docs/notes/`.
4. On a pass, continue with Section 5. On a fail, follow the consequence column.
