# Spur_SpLiCE — summary for the next chat

> **Historical hand-off:** this file records the earlier SpLiCE-IU phase. The
> canonical current proposal is SpLiCE-CRP v2 in
> [`PROJECT_GROUND_TRUTH.md`](PROJECT_GROUND_TRUTH.md). If the files conflict,
> follow `PROJECT_GROUND_TRUTH.md`.

Updated: 2026-08-18

## User goal

Build a scientifically credible SpLiCE-based method for automatically finding useful concepts and mitigating spurious correlations in a separately trained SSL encoder. The method must not assume that the nuisance is always a background: intended datasets are Waterbirds, CelebA and SpurCIFAR10. Currently only Waterbirds is available locally.

The user is competing with the paper “Bias Leaves a Gradient Trail”; that paper is a comparison point, not a method to copy.

## Repository

Workspace: `F:\Programming\Spur_SpLiCE`

Important existing files:

- `spur_splice.py` — main SSL training entry point.
- `experiments/spurious_eval/splice_cbm.py` — frozen SpLiCE-CBM discovery/evaluation.
- `scripts/tools/discover_splice_spurious_concepts.py` — concept discovery methods.
- `splice/ssl_regularization.py` — SpLiCE interventions and synthesis-distillation targets.
- `Spur_SpLiCE.tex` — project report.

Unrelated existing untracked directory `.claude/` belongs to the user and was not modified.

## Implemented method: SpLiCE Intervention Utility (SpLiCE-IU)

The proposed selector is group-free with respect to the spurious attribute, but it uses target labels. It should be described as target-label-only or group-free, not fully unsupervised.

For sparse SpLiCE code `c_i` and target label `y_i`:

1. Split the discovery set into stratified folds.
2. Fit an L1 sparse linear target probe on the complement of each fold.
3. On held-out examples, exactly ablate candidate coordinates in probe logit space.
4. Reward increases in true-class probability on probe errors.
5. Penalize decreases in true-class probability on already-correct examples.
6. Balance repair and damage by target class.
7. Screen individual coordinates, then greedily select a jointly useful non-redundant set.

For a set `S`, the conceptual score is:

```text
U(S) = class_balanced_mean_errors[max(p_true(delete S) - p_true, 0)]
       - class_balanced_mean_correct[max(p_true - p_true(delete S), 0)]
```

The implementation also records wrong-to-correct repairs, correct-to-wrong damage and a repair/damage ratio. `top_k` is an upper bound; greedy selection stops when marginal utility is non-positive.

Main implementation locations:

- `scripts/tools/discover_splice_spurious_concepts.py:285` — cross-fitted probes.
- `scripts/tools/discover_splice_spurious_concepts.py:487` — utility ranking and greedy selection.
- `experiments/spurious_eval/splice_cbm.py` — automatic discovery default is `intervention_utility`.
- `spur_splice.py:112` — automatic SSL concept discovery integration; default selector is group-free utility.

## Downstream application

The proposed training configuration uses selected coordinates in class-conditional neutralization:

- within each target class, selected SpLiCE coordinates are replaced by the target-class median;
- the remaining CLIP residual is preserved;
- the edited CLIP embedding is used as a stop-gradient target;
- a dedicated CLIP distillation head is trained jointly with the normal SimCLR/NT-Xent head.

Controls:

- `OrigCLIP` — unedited CLIP distillation;
- `UtilityNeutralize` — proposed selected coordinates + class median neutralization;
- `UtilityZeroOut` — selected coordinates zeroed;
- `UtilityRandCoords` — same number of random non-selected coordinates neutralized;
- baseline SimCLR.

Important caveat: discovery utility currently scores full zero-ablation in a probe, while the proposed downstream method uses class-neutralization. This mismatch is scientifically relevant and should be checked in the training results.

## Slurm launchers

The new arrays now target one dataset per submission. `DATASET` is an environment parameter; default is `waterbirds`.

Discovery array:

```bash
sbatch --export=ALL,DATASET=waterbirds \
  scripts/SpLiCE_intervention_utility_discovery_array.sbatch
```

- `#SBATCH --array=0-3`
- task 0: `intervention_utility`
- task 1: `error_contrast`
- task 2: `gradient_probe`
- task 3: `conditional_group` metadata oracle

Training array:

```bash
sbatch --export=ALL,DATASET=waterbirds,SEED=0 \
  scripts/SpLiCE_intervention_utility_training_array.sbatch
```

- `#SBATCH --array=0-4`
- task 0: baseline SimCLR
- task 1: unedited CLIP distillation
- task 2: `UtilityNeutralize`
- task 3: `UtilityZeroOut`
- task 4: `UtilityRandCoords`

Later datasets can be selected with `DATASET=celeba` or `DATASET=spur_cifar10`.

Both scripts validate the dataset name. Discovery results are cached and training tasks use a file lock so concurrent tasks share one concept list.

## Validation already completed

- Python files compile successfully.
- Full Python test suite passed: 28 tests.
- CLI help smoke checks passed for discovery, CBM and SSL entry points.
- Both new `.sbatch` files pass `bash -n` using Git Bash.
- `git diff --check` passes (apart from normal Git LF/CRLF warnings).
- No actual full GPU SSL training was run locally.

## Report changes

`Spur_SpLiCE.tex` was updated to describe SpLiCE-IU as the proposed method:

- title changed to automatic intervention-utility concept discovery;
- abstract and introduction now distinguish legacy metadata-assisted baselines from group-free discovery;
- new method subsection with utility equations and greedy selection;
- downstream synthesis-distillation section now uses IU-selected coordinates;
- experimental settings include 5 folds, 20k audit samples, candidate pool 100, minimum one repair and `K=5`;
- results explicitly state that IU experiments are pending and no positive result is claimed;
- limitations clarify that target labels are still required and group metadata remains evaluation/oracle-only;
- appendix and reproduction notes reference the new one-dataset Slurm arrays.

The LaTeX environment pairs and diff checks were verified. No local `pdflatex`, `latexmk`, `xelatex`, `lualatex` or `tectonic` executable was available, so PDF compilation was not performed.

## Waterbirds discovery results received from the user

Input files were copied to `F:\Transfered` by the user:

- `SpLiCE_utility_discovery_23368248_0.out` through `_3.out`;
- `waterbirds_discovery_utility_cbm_seed0.json`;
- `waterbirds_discovery_errcontrast_cbm_seed0.json`;
- `waterbirds_discovery_gradprobe_cbm_seed0.json`;
- `waterbirds_discovery_oracle_cbm_seed0.json`.

All four discovery jobs completed successfully on `dataset=waterbirds`, `train=train`, `eval=val`, `seed=0`.

Frozen SpLiCE-CBM results:

| Method | Selected concepts | Avg | WG | Delta WG |
|---|---|---:|---:|---:|
| Baseline | — | 83.90% | 39.85% | — |
| Intervention utility | `crow, heron, birds` | 83.40% | 40.60% | +0.75 pp |
| Error contrast | `seagull, sparrow, heron, duck, bamboo` | 81.07% | 18.80% | -21.05 pp |
| Gradient probe | `seagull, sparrow, freshwater, duck, birds` | 80.82% | 18.05% | -21.80 pp |
| Conditional-group oracle | `raven, forests, rainforest, whale, bamboo` | 84.74% | 48.87% | +9.02 pp |

Group accuracy order is `(background, y)` encoded as `background + 2*y`:

- group 0: land background / landbird;
- group 1: water background / landbird;
- group 2: land background / waterbird;
- group 3: water background / waterbird.

The baseline worst group is group 2 at 39.85%. IU changes groups approximately to `[98.72, 79.83, 40.60, 84.96]%`: it improves the small worst group but hurts group 1 and lowers average accuracy.

The IU log shows greedy selection in this order:

```text
birds: score 0.002804, repair 13, damage 8
heron: marginal +0.002519, repair 24, damage 13
crow:  marginal +0.002061, repair 45, damage 19
```

The attached CBM JSON files do not include the raw discovery diagnostics (cross-fitted probe accuracy, class supports, positive candidate count). Those are in the raw discovery JSON under the cluster output directory, e.g.:

```text
outputs/waterbirds_discovery_utility_top5_seed0/waterbirds_splice_concepts.json
```

The `nonzero_probe_concepts` value is 23 for all methods: the final CBM L1 probe uses only 23 of the 10,000 SpLiCE coordinates.

## Current scientific interpretation

The result is not an infrastructure failure. It shows:

1. SpLiCE contains useful nuisance-related concepts: the privileged oracle gives +9 pp WG.
2. Zero-out intervention can improve frozen CBM WG when the selected concepts are correct.
3. Error-contrast and gradient-probe selections are actively harmful in this setup.
4. The current group-free IU selector is technically functional but selects content/target-looking concepts (`birds`, `heron`, `crow`) rather than obvious background concepts.
5. Positive error-repair utility is not equivalent to nuisance specificity. This is the main scientific warning.

The IU frozen-CBM result is weakly positive (`+0.75 pp WG`) but not convincing and has lower average accuracy. The oracle gap indicates that concept discovery, not the intervention code, is currently the bottleneck.

## Training interpretation / next action

The user started `SpLiCE_intervention_utility_training_array.sbatch` for Waterbirds, seed 0. No training `.out` or W&B JSON was provided yet, so do not claim a final SSL result.

Recommended next step:

1. Let seed 0 finish because it is a useful matched pilot and includes the necessary controls.
2. Do not launch seeds 1/2 until comparing `UtilityNeutralize` against `OrigCLIP`, `UtilityZeroOut` and `UtilityRandCoords`.
3. Inspect WG, average accuracy, target accuracy and spurious-probe accuracy from the training logs.
4. If `UtilityNeutralize` does not beat both `OrigCLIP` and random coordinates, treat the current IU version as a negative result.
5. If needed, improve the selector before multi-seed training. The two most plausible changes are:
   - score the actual class-neutralization counterfactual rather than only zero-ablation;
   - add a target-preservation / target-separation constraint so the selector cannot prefer core class concepts.

Class-neutralization may still outperform zero-out because it replaces selected coordinates with a target-class median and can preserve class-level signal. Therefore the discovery CBM result is a warning, not proof that the SSL training will fail.

For the next chat, ask the assistant to read this file first, then attach training `.out`/W&B result files if available.
