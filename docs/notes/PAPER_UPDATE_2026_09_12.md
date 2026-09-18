# Manuscript update and submission priorities — 12 September 2026

`CoSpRo.tex` now separates the four-seed held-out test result from the expanded validation controls. Training code was not changed and no experiments were launched.

## What changed

- Rewrote the abstract around the problem, method and main finding; rewrote Method as four stages with intuition before equations.
- Added ten verified references covering negative selection, false-negative cancellation, relational SSL, concept discovery/bottlenecks and shortcut learning. Removed the unrelated, uncited SCLIP entry. The bibliography links primary sources; LA-SSL is pinned to the implemented arXiv v2.
- Kept the matched test comparison central. Integrated semantic and reconstruction seeds 2/4 into four-seed validation comparisons. Moved frozen-feature and two-seed target diagnostics to the appendix; included LA-SSL seed 1 explicitly as preliminary.
- Removed descriptions implying that qualitative image panels were already shown. The repository has a selection manifest, but the manuscript contains no such rendered panels.
- Stated that all 12 selected groups are singletons, that projection can increase wrong-target mass, and that null filtering does not establish causal nuisance identification.

## Evidence snapshot

The read-only W&B query found five completed runs since the previous manuscript update, all Waterbirds, all epoch 500 with converged probes:

| Run ID | Arm / seed | Validation Avg. | Validation WGA |
|---|---|---:|---:|
| dc6lr4cp | Semantic graph / 2 | 54.13 | 51.88 |
| yax3mt1q | Semantic graph / 4 | 54.96 | 51.88 |
| ixma3dec | Reconstruction / 2 | 49.79 | 43.47 |
| lo48apo0 | Reconstruction / 4 | 53.21 | 42.11 |
| emic8s8c | LA-SSL / 1 | 50.38 | 45.61 |

Full configurations and endpoint history are in `outputs/reports/paper_update_2026_09_12/wandb_snapshot.json`. `wandb_context.json` is the compact inventory of the 113 project runs created since 5 September. `validation_results.json` records inputs and sample-standard-deviation aggregates. It supplements the older `paper_results.json`; the historical registry was not overwritten.

Core CoSpRo seeds 1/3 match their graph-ablation validation endpoints exactly. Relevant architecture, optimizer, augmentation, temperature, graph-loss and probe settings match. New semantic runs add CUDA/cuDNN version fields; shared recorded dependency versions are unchanged. All five new runs are validation-only (`final_test=False`).

The main test result remains WGA 50.64 ± 1.97 versus SimCLR 46.82 ± 3.65 and raw-CLIP KL 48.48 ± 3.44. Expanded validation changes the mechanism evidence: CoSpRo versus semantic graph is +1.99 points, with paired differences +7.25, −1.88, +1.60, +1.01. Direct reconstruction averages 46.61 WGA, only +1.07 over SimCLR. The initial two-seed advantages should not be used as the primary conclusions.

## What is already strong

The matched raw-teacher and sampler-only controls address two central confounds: generic external semantic supervision and batch composition. Four-seed test results favor the complete method on average. Graph construction is inspectable, the teacher is training-only, and the updated description agrees with the implementation. The paper already has per-seed/per-group results and converged probes; repeating these checks or the completed semantic/reconstruction seeds is unnecessary.

## Prioritized research plan

### MUST DO before submission

1. **Put the decisive completed controls on the same held-out split.** Evaluate the saved semantic-graph and reconstruction encoders for seeds 1–4 with the existing final-test probe protocol; preserve preprocessing, probe IDs, epoch and settings. Produce a single test table with paired CoSpRo differences and per-group outcomes. This closes the present gap between the main test claim and validation-only mechanism controls. **No new SSL training:** reuse retained checkpoints; feature extraction and probe fitting may be required. Some checkpoints are cluster-only. The manifest `--locked-test` route starts another training attempt, so it is not the checkpoint-only route to use. Record the current decision prospectively; do not reconstruct a missing historical lock timestamp.

2. **Complete the external SSL robustness comparison.** Finish LA-SSL seeds 2–4 with the declared adaptation, then apply the same held-out evaluation. Report its extra scoring cost alongside optimizer-step budget. One seed cannot answer whether CoSpRo improves on a relevant SSL mitigation method. **Three new SSL runs**, unless already running elsewhere. Keep the adaptation fixed; do not launch a broad tuning sweep.

3. **Close the ongoing dataset work with the decisive controls.** Additional datasets are already running. Incorporate their completed results with SimCLR, matched raw-CLIP KL and semantic-SpLiCE KL, using the same reporting and selection discipline. These distinguish generic teacher transfer from the proposed relation construction. **Use ongoing runs first**; commission only a missing essential control. If the gain is dataset-dependent, delimit the claim rather than search for favorable datasets.

4. **Turn the graph audit into a compact mechanism figure.** Reuse the graph, post-hoc composition report and `outputs/reports/paper_evidence/visual/concept_panels.json`. Show a prespecified small set of successful and failed pairs, actual concept names, raw/projected similarity and target/context labels; accompany it with per-source-group cross-context and wrong-target mass. The manuscript currently explains an interpretable operation without showing image-level evidence, and some minority-group relations are poor. **No new training.** Preserve the current qualified claim: a positive relation score does not prove background removal. All selected groups are singletons, so do not claim demonstrated benefits of grouping.

### SHOULD DO if time permits

The larger-probe-budget suggestion was withdrawn at the author's request. No change to the existing probe protocol is planned.

5. **If a stronger projection-specific claim is essential, test one matched random-subspace graph.** The existing null audit tests graph scores, not downstream robustness. A fixed random-subspace graph with matched graph budgets and the same student protocol would test whether concept alignment matters beyond generic geometric perturbation. **New training required.** This is conditional, not a prerequisite for the paper's current qualified empirical claim; settle the existing held-out semantic comparison first.

### NOT NECESSARY / diminishing returns

Do not add new architectures, larger teachers, broad hyperparameter sweeps, a replacement concept dictionary, prototype compression or additional method variants. Do not extend every exploratory raw/shuffled direct-transfer arm merely to fill a matrix. Do not compare supervised or zero-shot literature scores as if they were matched SSL baselines. Do not claim statistical certainty from the final ten probe optimization epochs.

Execution order: completed checkpoint evaluation → remaining LA-SSL seeds → integrate ongoing datasets → mechanism figure. The conditional random-subspace experiment comes only after these results clarify whether a stronger mechanism claim is worth the compute.

## Manuscript verification

Compiled with portable Tectonic 0.17.0: 12 pages (7 main text, 1 references, 4 appendix). All final pages were rendered and visually inspected. All citation keys and cross-references resolve; no overfull boxes or font-substitution warnings remain. Nonfatal underfull-box warnings remain. `git diff --check` passed. Output hashes and checks are recorded in `outputs/reports/paper_update_2026_09_12/verification.json`.
