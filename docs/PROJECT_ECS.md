# Project 1: Early Concept Swap (ECS)

Version 2, 2026-11-20. Implementation branch `ECS`. Read [`PROJECT_PLAN.md`](PROJECT_PLAN.md) first: it fixes the setting, the
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

Read from the section **Early Concept Swap kill test** at the top of
`outputs/reports/factors_metashift/concept_curves.md` (diagnostics defined in Plan, Section 7). The
summary computes the verdict (function `k1_verdict` in `cospro/cli/summarize_concept_curves.py`) by
these rules, with "three of four seeds" generalized to three quarters of the seeds:

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | detection AUROC at least 0.7 on three of four seeds, and the flagged set holds at least one attribute unit and no class unit on three of four seeds | the timing signal holds |
| Weak | every other case with scores (mean AUROC at least 0.55 and one pass condition missed) | try `--curve_statistic margin`, `--curve_unit_count 0`, `T0` of 20 and 100 |
| Fail | mean AUROC below 0.55, or a class unit has the highest onset score on more than half of the seeds | the timing hypothesis ends; ECS continues only with human-corrected flags (rule `units`), and the effort moves to CSA |

### Stage K2: does the swap raise WGA?

Read from the same section of `concept_curves.md` (function `k2_verdict`). All numbers are paired
by seed: validation WGA of the arm minus validation WGA of `simclr` on the seeds both finished, each
WGA the mean of the last four probes. The verdict is `pending` while a needed arm is missing.

| Outcome | Condition | Consequence |
|---|---|---|
| Pass | `curves` beats `simclr` by at least 5 points of mean validation WGA and beats both `simclr` and `curves_random` on three of four seeds | proceed to Section 5 |
| Partial | `curves` misses the pass and `curves_oracle` gains at least 5 points over `simclr` with a gain on three of four seeds | the swap works and the selection needs repair; the per-unit effects of CSA show which units to flag |
| Fail | `curves` misses the pass and `curves_oracle` misses the same rule against `simclr` | the swap itself fails in this setting; ECS ends |

What each control arm tells:

| Arm | If it matches `curves` |
|---|---|
| `curves_random` | the identity of the flagged units carries no information |
| `curves_bottom` | the direction of the timing signal carries no information |
| `curves_shuffled` | the concept maps carry no information; the gain comes from the swap as an augmentation |
| `curves_oracle` | upper bound: the audit already selects as well as hidden labels do |

The same section lists every swapping ECS arm with its summed share of the flagged units at the end
of the early window and at the last audit, and the number of seeds where it fell (theory check of
Section 5, item 4).

**Verified locally (2026-11-20):** on a synthetic MetaShift tree, an audit-only run (`curves_audit`)
reproduces the SimCLR loss of the same seed exactly at every epoch, with 0 and with 2 loader workers,
and a swapping run equals SimCLR through epoch `T0` and departs from it afterwards. The test
`test_audit_only_run_reproduces_simclr_and_the_swap_starts_after_the_window` keeps this. So any
difference between `curves*` arms and `simclr` comes from the swap alone.

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
3. Ablations, each one arm that changes one option of `curves`; `ARMS="curves_ablations"` submits
   all eleven (44 jobs for four seeds):

   | Arm | Change | Question |
   |---|---|---|
   | `curves_t20`, `curves_t100` | `--curve_early_epochs 20` or `100` | does an earlier or later flag decision help? |
   | `curves_tau30`, `curves_tau70` | `--curve_tau 0.3` or `0.7` | fewer or more flagged units |
   | `curves_q25`, `curves_q100` | `--curve_swap_probability 0.25` or `1.0` | WGA as a function of `q` (the theory predicts a monotone rise) |
   | `curves_margin` | `--curve_statistic margin` | the second reliance statistic |
   | `curves_area` | `--curve_area_normalize true` | reliance per unit of area: is the ranking more than region size? |
   | `curves_factors` | `--curve_unit_count 0` | single factors as units |
   | `curves_second` | `--curve_swap_views second` | one swapped view leaves a share of the spurious direction (Remark in the `.tex`) |
   | `curves_oracle_t1` | `--curve_flag_rule oracle --curve_early_epochs 1` | swap from epoch 2 with hidden-label flags: what the early window costs |

   Human-corrected flags use `--curve_flag_rule units --curve_flag_units <ids>` (the arms
   `curves_unit_<k>` of CSA for one unit) and need no arm of their own.
4. Theory checks: WGA as a function of `q` (arms `curves_q25`, `curves`, `curves_q100`); the curves
   after the swap starts (the reliance on the flagged units has to fall), reported automatically by
   the summary (Section 3).

## 6. Risks specific to ECS

| Risk | Sign | Response |
|---|---|---|
| The encoder relies on the object first (large animals) | K1 fails, class units lead | human-corrected flags; CSA |
| A direction learned before `T0` persists after the swap starts | reliance on flagged units stays high after epoch 60 | smaller `T0`; larger `q` |
| Swap seams act as a shortcut | `curves_shuffled` matches `curves` | alternative B of the Plan, Section 10 |
| MetaShift noise hides the effect | seed spread above the gain | eight seeds; confirm on the Waterbirds test split |

Prior work: Plan, Section 9.

## 7. Status and next steps

Implemented on branch `ECS` and tested locally (2026-11-20):

| Piece | Where |
|---|---|
| Audit, flag rules, swap | `cospro/curves/core.py`, `cospro/curves/swap.py` |
| Training method `concept_curves` | `cospro/methods/concept_curves.py` |
| Audit callback, JSON files | `ConceptAudit` in `cospro/training/callbacks.py` |
| Options, including `--curve_area_normalize` | `ConceptCurveOptions` in `cospro/config/training.py` |
| Kill-test arms and ablation arms | `add_curve_arms` in `scripts/submit_factor_study.sh` |
| K1 and K2 verdicts, post-swap reliance check | `cospro/cli/summarize_concept_curves.py` |
| Tests, including the SimCLR equality run | `tests/test_concept_curves.py` |

**Result on MetaShift (2026-11-24, seeds 1 to 4): K1 fail, K2 fail, so ECS ends.** The onset score follows
region area (Pearson 0.99), so detection AUROC is 0.50. `curves` gains +5.0 and `curves_oracle` +2.9 over
`simclr`, both carried by one low `simclr` seed. Every swap lowers the measured reliance on its units on 4 of 4
seeds. Details and numbers: [`notes/ECS_CSA_RESULTS_2026_11_24.md`](notes/ECS_CSA_RESULTS_2026_11_24.md). Under
the consequence column of Section 3 the effort moves to CSA. The ablation arms of Section 5 stay unrun.
