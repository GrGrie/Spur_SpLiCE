# Refactoring plan

Status: proposal, 2026-09-18. Builds on [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md)
(candidates C1–C8) and the project rules in [`AGENTS.md`](../AGENTS.md).

## Goals

1. Adding a dataset, a concept vocabulary, a training method or an ablation touches one file plus a registry line.
2. Every hyperparameter has one default, defined in one place.
3. Concept groups and teacher graphs come with quantitative metrics and visual evidence, comparable across hyperparameter settings.
4. Paper numbers stay reproducible through every step.

## Constraints every phase respects

- **SLURM.** Each new cluster entry point ships with its launcher in the same change. Resources: one V100,
  at most 5 CPUs, at most 8G memory per CPU.
- **Storage.** Binaries go to `/scratch/xar68reb/CoSpRo/`, compact JSON goes to `outputs/` and syncs through Git.
  Intermediate epoch checkpoints are deleted during training to protect the `/home` quota.
- **W&B.** WGA stays visible per probe epoch in W&B across all refactors.
- **English** for code, comments and docs, following the style rules in `AGENTS.md`.
- **Timing.** Phases that change command lines (3 onward) land between studies, after the current run matrix completes.
  `experiments.runner` resumes an attempt only when its command matches exactly.

## Phase order at a glance

| Phase | Content | Risk | Value | Cluster work |
|---|---|---|---|---|
| 0 | Safety net: tests, golden snapshots, CI | none | enables everything | one smoke job |
| 1 | Conventions: SLURM lint, launcher template, quick wins | low | high | none |
| 2 | Concept-group and teacher-graph diagnostics | low (new code only) | very high | diagnostics jobs |
| 3 | Typed configuration (C2) | medium | high | re-run golden checks |
| 4 | `TrainingMethod` seam (C1) + W&B metric contract | medium | very high | smoke job per method |
| 5 | Trainer, callbacks, storage policy (C4) | medium | high | smoke job |
| 6 | Dataset adapters (C3) | low | high for new datasets | cache check per dataset |
| 7 | Linear probe as a library function (C5) | low | medium | none |
| 8 | `cospro/` package and `ConceptDictionary` (C6, C7) | medium | high for vocab ablations | graph rebuild check |
| 9 | Package layout move (C8) | low with shims | navigation | none |

Phase 2 depends only on existing artifacts, so it runs as a parallel track right after phase 1.

---

## Phase 0 — Safety net

**Status: done (2026-09-18).** 117 tests pass locally; the Linux training snapshot appears with the first
`run_golden_smoke.sbatch` run. Manifests moved to YAML in the same change.

**Steps**

1. Add `pytest` to a dev extra in `pyproject.toml` and move project metadata there from `setup.py`.
   `scripts/freeze_environment.sbatch` records the exact cluster package versions in `environment/`.
2. Golden command snapshot: for every manifest, seed, arm and `locked_test` variant, store the output of
   `experiments.runner.command_for` in `tests/golden/commands.json`.
3. Golden numeric smoke test: a synthetic 32×32 dataset with 64 images and 4 groups, two SSL epochs on CPU for each
   training mode. Store loss values and probe metrics with a tolerance.
4. Golden graph test: `build_concept_groups` and `build_teacher_graph` on the fixture cache produce a stored
   `graph_fingerprint`.
5. GitHub Actions workflow running `pytest` on CPU.
6. `scripts/run_golden_smoke.sbatch`: runs the suite on the cluster and every training mode on a V100 with AMP.
   AMP changes the numbers, so the GPU run checks completion, finite metrics and a recorded WGA.

**Done when** `pytest` passes locally and in CI, the three golden files are committed and one cluster smoke job has completed.

## Phase 1 — Conventions and quick wins

**Status: done (2026-09-18)** on branch `refactor`, one commit per step. Notes:

- `splice/settings.py` and `scripts/load_splice_cluster_env.sh` are the single sources of cluster paths
  and the W&B entity. `run_cospro_pipeline.sh` forwards only the variables that are set.
- `splice/compat.py` holds every CRP-era name. Storage names hash the configuration under the historical
  option names (pinned by `tests/test_storage_identity.py`), so resumed runs keep their checkpoint folders.
  The `waterbirds_crp` study, `crp_graph.json` and the `--crp_*` options stay as historical records.
- Kept on purpose: `--cudnn_benchmark` and `indegree_factor` feed stored identities (storage names,
  W&B run names, graph configs).
- Moved to phase 3: the CoSpRo student defaults that `run_training.sbatch` injects in automatic graph mode.
  They belong in a named preset of the typed configuration.
- Follow-up: a style pass over older comments and docs for the rules in `AGENTS.md`.

**Steps**

1. **SLURM lint test** (`tests/test_slurm_launchers.py`): parse every `#SBATCH` header and assert
   `cpus-per-task ≤ 5`, `mem ≤ 8G × cpus`, `gres=gpu:1` on GPU jobs and a log path under `outputs/SLURM/`.
   The rule becomes executable and every future launcher is checked automatically.
2. **Launcher template** `scripts/_template.sbatch`: resource header, `set -euo pipefail`,
   `load_splice_cluster_env.sh`, one Python call with `"$@"` passthrough. New launchers copy it.
3. Move hyperparameter defaults out of `run_training.sbatch`, `run_cospro_pipeline.sh` and siblings.
   The manifest carries the values and the launcher forwards arguments.
4. Collect cluster paths and identities (`/scratch/xar68reb/CoSpRo`, `/home/xar68reb/Datasets`, W&B entity) in
   `scripts/load_splice_cluster_env.sh` as environment variables with the current values as defaults.
5. Rename internal `crp_*` names to `cospro_*` (C7). Readers of old artifacts keep working through one
   `compat.upgrade_graph_artifact` function.
6. Move `CoSpRo.tex`, `iclr2027_conference.sty` and `paper_results.json` to `paper/`, dated notes to `docs/notes/`.
7. Split `scripts/tools/` into `tools/maintenance/` (migration, archive, cleanup) and `tools/paper/` (figures,
   submission evaluation, paper registry). Pipeline stages stay in place until phase 8.
8. Delete the dead code listed in section 8 of the review.

**Done when** the SLURM lint test passes, golden snapshots are unchanged and `grep -rn "crp_" --include=*.py`
lists only `compat/` and serialized-field readers.

## Phase 2 — Diagnostics for concept groups and teacher graphs

This phase answers three questions: how good is a grouping, how good is a graph and how two hyperparameter
settings compare. It adds a new module that only reads existing artifacts, so trained results stay untouched.

### 2.1 Two tiers of metrics

- **Label-free metrics** use only the SpLiCE cache and the artifacts. They are safe for choosing hyperparameters
  inside the label-free protocol.
- **Post-hoc metrics** also read the hidden labels `y` and the spurious attribute `a` from dataset metadata. They explain
  and validate the method. Each post-hoc metric reads labels from one module, `cospro/diagnostics/labels.py`,
  which keeps the separation auditable. Open decision for the paper protocol: whether post-hoc metrics on the validation
  split may guide hyperparameter choice, or serve for explanation only.

### 2.2 Concept-group metrics

Computed per grouping configuration.

| Metric | Definition | Reading |
|---|---|---|
| Text coherence | mean pairwise cosine of concept text embeddings inside each composite group | high means the group is one meaning |
| Co-activation coherence (NPMI) | mean over concept pairs in a group of `log(p(i,j) / (p(i)p(j))) / -log p(i,j)`, with `p` from binary SpLiCE activations over images | standard topic-coherence score in `[-1, 1]`; high means the concepts fire on the same images |
| Separation | silhouette score of composite groups on text embeddings with cosine distance | high means groups are distinct from their neighbours |
| Image coverage | fraction of images with positive activation for at least one composite group | how much of the dataset the grouping explains |
| Structure | group count, singleton fraction, maximum size, size entropy | detects over-merging (one giant group) or under-merging (all singletons) |
| Bootstrap stability | mean adjusted Rand index (ARI) between the grouping on the full cache and groupings on five 80% image subsamples | high means the grouping reflects the data robustly |
| Cross-config agreement | ARI and variation of information between two configurations on their shared active concepts | how much a hyperparameter change reshapes the partition |
| Audit yield | fraction of groups that pass the null test in the graph stage | links grouping to its downstream use |
| Spurious selectivity (post-hoc) | `abs(AUC_a − 0.5) − abs(AUC_y − 0.5)` where `AUC_a` and `AUC_y` measure how well group activation predicts `a` and `y` | high marks groups that carry the spurious attribute and leave the class intact |
| Spurious fragmentation (post-hoc) | number of groups needed to reach 80% of the total positive spurious selectivity | low means synonyms such as "water", "lake" and "ocean" landed in one group |

The paper update of 2026-09-12 records that all 12 selected groups are singletons. Fragmentation and NPMI turn that
observation into a number and show which thresholds produce multi-concept spurious groups.

### 2.3 Teacher-graph metrics

| Metric | Definition | Reading |
|---|---|---|
| Coverage, edge count, effective donor count, indegree Gini | existing `degree_stats` | graph shape and hubness |
| Novelty over raw CLIP | fraction of teacher edges absent from the raw CLIP kNN at the same `k` | near zero means the graph repeats CLIP kNN |
| Concept contrast | mean absolute difference of the source group activation across an edge, divided by that activation's standard deviation | high means edges connect images that differ in the removed concept |
| Residual agreement | mean residual SpLiCE similarity over edges | high means the remaining content matches |
| Null margin | observed score divided by null threshold, per selected group | strength of evidence per group |
| Seed stability | Jaccard overlap of edge sets built with two audit seeds | robustness of the graph |
| Group transition matrix (post-hoc) | edge-weight mass from each `(y, a)` group to each `(y', a')` group | the central picture of what the graph teaches |
| Counterfactual edge rate (post-hoc) | weighted fraction of edges with `y_i = y_j` and `a_i ≠ a_j` | headline graph metric |
| Class consistency (post-hoc) | weighted fraction of edges with `y_i = y_j` | edges that preserve the class |
| Lift (post-hoc) | counterfactual edge rate divided by the rate of the matched raw-CLIP graph and of a degree-matched random graph | improvement over baselines |
| Minority reach (post-hoc) | fraction of minority-group anchors with at least one counterfactual edge | whether the graph reaches the groups WGA depends on |
| Confidence calibration (post-hoc) | counterfactual edge rate per edge-confidence decile | a rising curve validates the confidence weights used in the relational KL |

The matched baselines already exist (`raw_clip_graph.json`, `semantic_splice_graph.json` from
`build_cospro_baseline_graphs.py`), so every graph metric appears next to its two baselines.

### 2.4 Link to training outcomes

Every SSL run records the teacher-graph metrics in its W&B config and `run.json`. A W&B scatter of counterfactual edge
rate against final WGA across graph variants then shows whether the graph metric predicts the training outcome.
A predictive metric allows cheap graph selection before any 500-epoch run.

### 2.5 Visual dashboards

One HTML dashboard per stage, rendered locally from synchronized JSON. Thumbnails come from the local dataset copy.
The dashboard renders the metric panels alone when the dataset is absent.

**Concept-group dashboard**

1. Sweep heatmaps over `text_similarity_threshold × coactivation_threshold`: NPMI, silhouette, singleton fraction,
   fragmentation and ARI to a reference setting. One glance shows the useful region of the grid.
2. Concept map: 2D UMAP of active concept text embeddings, coloured by composite group, point size by frequency,
   hover shows concept names.
3. Selectivity scatter (post-hoc): `AUC_y` on the x axis, `AUC_a` on the y axis, one point per group, selected groups
   highlighted. Correct behaviour places background groups in the upper-left region.
4. Group cards sorted by spurious selectivity, each with concepts, pairwise evidence and top-activating thumbnails
   (reusing the existing card code).

**Teacher-graph dashboard**

1. Summary table: every metric for the teacher graph and its two baselines side by side.
2. Transition matrices: three `(y, a) × (y', a')` heatmaps for raw CLIP, semantic SpLiCE and CoSpRo.
   Mass on same-class, flipped-attribute cells is the visual signature of a correct graph.
3. Edge gallery: stratified edge sample by source group and confidence decile, shown as anchor → neighbour image pairs
   with the group concepts and edge confidence. Post-hoc view marks counterfactual edges with a coloured frame.
   Builds on `select_graph_panels.py` and `render_concept_panels.py`.
4. Null-test strip plot: per group, the null score distribution with the observed score marked.
5. Calibration curve and degree histograms.

### 2.6 Implementation

```text
cospro/diagnostics/            # new module; moves with the phase 8 package
  labels.py                    # sole reader of y and a
  group_metrics.py             # label-free and post-hoc group metrics
  graph_metrics.py             # label-free and post-hoc graph metrics
  sweep.py                     # evaluates a list of grouping configs on one cache
  dashboard.py                 # renders HTML from metrics JSON
scripts/run_group_diagnostics.sbatch   # 5 CPU, 40G; reads cache from scratch
scripts/run_graph_diagnostics.sbatch   # 5 CPU, 40G, V100 for neighbour search
tools/render_cospro_dashboard.py       # local PC entry point
```

Outputs: `outputs/reports/cospro_diagnostics/<dataset>/<artifact_id>/metrics.json` plus `edge_sample.json`
(a few hundred sampled edges with sample IDs). Both are compact and synchronize through Git. HTML stays local.

**Done when** both dashboards render for Waterbirds, every metric has a unit test on the fixture cache with a
hand-computed expected value and the sweep job has produced a heatmap over at least a 4 × 4 threshold grid.

### 2.7 Choosing between two groupings

A grouping is an intermediate artifact: its value is the teacher graph it produces and the WGA that graph
earns. "Best" therefore follows a funnel, cheap to expensive:

1. **Validity filters (label-free, automatic).** Keep configurations with maximum group size under a cap,
   mean NPMI above a floor, bootstrap ARI of at least 0.8 and a positive audit yield.
2. **Graph proxy (post-hoc, manual).** Build the graph for every survivor and rank by the group-balanced
   counterfactual rate: the mean over the four `(y, a)` anchor groups of the fraction of edge weight that keeps
   `y` and flips `a`. Plot it against coverage and keep the Pareto front.
3. **Short training.** 100-epoch single-seed runs for the top five configurations, compared on validation WGA.
4. **Full training.** 500 epochs over four seeds for the winner.
5. **Proxy check.** Spearman correlation between step 2 and step 3 across the short runs. A strong correlation
   makes step 2 the default selector for later sweeps.

Step 2 reads training-split labels. A cleaner variant builds the cache and graph for the validation split with
the same hyperparameters and scores the proxy there, keeping selection on the split that already selects models.

**First reading of the current Waterbirds graphs (2026-09-18, post-hoc).** Group-balanced counterfactual
rate: CoSpRo 0.23, semantic SpLiCE 0.20, raw CLIP 0.20. Minority anchors with a counterfactual edge:
CoSpRo 59%, semantic SpLiCE 54%, raw CLIP 52%. Per concept group, the background groups {Bamboo}, {Sunset}
and {Autumn} produce counterfactual edge rates of 5–7%, while the owl groups produce 1–3% and mostly link
images that share both class and background. A grouping that merges background synonyms into a few strong
groups is the direction the proxy rewards. The global counterfactual rate is dominated by majority anchors,
so the group-balanced form is the one to report.

## Phase 3 — Typed configuration (C2)

**Steps**

1. Define frozen dataclasses: `DataConfig`, `ModelConfig`, `SSLConfig`, `ProbeConfig`, `RunIdentity`,
   `StorageConfig`, `TrackingConfig`. Each validates itself in `__post_init__`.
2. `load_config(manifest, arm, seed, overrides)` merges `common`, arm args and dotted overrides such as
   `ssl.temperature=0.05`.
3. The CLI becomes a thin adapter: `--manifest`, `--arm`, `--seed`, `--set key=value`.
4. Runner writes `command.json` schema `experiment-command-v2` and records the resolved config.
5. Manifest `sweep` block: `{"method.weight": [0, 0.25, 0.5]}` expands into arms automatically.

**Done when** golden numeric results match phase 0, every default lives in one dataclass field and launchers carry
resources and paths only.

## Phase 4 — `TrainingMethod` seam (C1) and W&B metric contract

**Steps**

1. Protocol `TrainingMethod` with `Config`, `wrap_loader`, `extra_loss`, `provenance`, `input_artifacts`.
2. Adapters: `SimCLROnly`, `CoSpRoRelational`, `FrozenConceptDistill`, `LaSSL`, registered by name.
3. `ssl_loop` calls `method.extra_loss` through one path. Dataset adapters lose all knowledge of the training mode.
4. W&B metric contract in `tracking/metrics.py`: stable keys such as `probe/val/wga`, `probe/val/avg_acc`,
   `probe/val/group_acc/<g>`, `train/loss/*`, `method/*`. `wandb.define_metric("probe/val/wga", summary="max")` plus a
   `last` value. `run.json` stores the same keys. A mapping from old keys keeps `export_wandb_runs.py` working.
5. One smoke launcher per method: `scripts/smoke_methods.sbatch` runs two epochs of each method on the V100.

**Done when** `grep splice_mode` returns only the CLI adapter and `compat/`, a new method needs one file plus a
registry line (proved by a test that registers a dummy method) and W&B shows `probe/val/wga` for each smoke run.

## Phase 5 — Trainer, callbacks and storage policy (C4)

**Steps**

1. `Trainer.fit()` owns the epoch loop and emits `on_epoch_end`, `on_train_end` and `on_failure`.
2. Callbacks: `RankMetrics`, `PeriodicProbe`, `WandbLogger`, `RunRecordLogger`, `CheckpointPolicy`.
3. `StoragePolicy` in one module: rolling epoch checkpoints in the run folder, deletion as training proceeds, the final
   checkpoint moved to scratch with a SHA-256 attestation in `run.json`. Tests assert the peak number of checkpoints
   in the run folder, which protects the `/home` quota.
4. `TrainingState` bundles model, optimizer, scaler, loader generator and method state for save and resume.
5. `spur_splice.py` shrinks to a short entry point.

**Done when** golden results match, a unit test simulates 10 epochs with a fake model and checks the files left in
`/home` and scratch. A cluster smoke job resumes from a checkpoint with identical metrics.

## Phase 6 — Dataset adapters (C3)

**Steps**

1. `SpuriousDataset` base class with `read_metadata()` returning columns `path, y, a, split`, plus `load_image()`.
2. One generic `build_loader(dataset, role, config)` for the roles `ssl`, `rank`, `probe_train` and `probe_eval`.
3. `@register_dataset(name, aliases=...)` replaces the three registry dictionaries and the bash `case` statement.
4. Image size and model compatibility become dataset attributes.
5. Dataset onboarding checklist in `docs/ADDING_A_DATASET.md`: adapter file, cache launcher run, diagnostics run,
   manifest.

**Done when** Waterbirds, CelebA and Spur-CIFAR10 each fit in about 50 lines and a shared test suite runs against every
registered dataset.

## Phase 7 — Linear probe as a library function (C5)

**Steps**

1. `evaluate_probe(encoder, dataset, ProbeConfig) -> ProbeResult` as a pure computation.
2. `persist_probe_result` writes features to scratch and JSON to `outputs/`.
3. The trainer callback, `evaluate_submission_checkpoints.py` and the standalone CLI all call `evaluate_probe`.

**Done when** the 35-field `Namespace` bridge is gone and probe tests run on synthetic features.

## Phase 8 — `cospro/` package and concept dictionaries (C6, C7)

**Steps**

1. Split `splice/cospro.py` into `cache.py`, `grouping.py`, `neighbors.py`, `audit.py`, `graph.py` and `pipeline.py`.
2. `NeighborIndex` with `ExactNeighbors` and `LshNeighbors`, tested against each other on the fixture.
3. `ConceptDictionary` (words, text embeddings, provenance) with adapters for LAION, Open Images V7 and any text file.
   Manifest entry: `dictionary: {kind: file, path: ..., order: frequency | file, size: N}`.
4. Pipeline stages move from `scripts/tools/` to `cospro/cli/`. Launchers keep their names.
5. Vocabulary ablation manifest: the same grouping and graph pipeline over two or three dictionaries, compared through
   the phase 2 dashboards.

**Done when** the golden graph fingerprint matches and a new vocabulary needs a text file plus a manifest entry.

## Phase 9 — Package layout (C8)

**Steps**

1. `git mv` into the target layout from the review, section C8, one package per commit.
2. Re-export shims at old import paths for one release, each emitting a `DeprecationWarning`.
3. Vendored SpLiCE moves to `third_party/splice/` with a NOTICE describing local modifications.
4. Update `PROJECT_MAP.md`, `docs/REPO_STRUCTURE.md` and `scripts/README.md`.

**Done when** all tests pass, every launcher runs its `--help` path in the SLURM lint test and the shims are the
only references to old paths.

---

## Decisions for the author

Resolved on 2026-09-18:

1. Post-hoc metrics guide manual selection; automatic tuning uses label-free metrics only.
2. Manifests are YAML (`experiments/manifests/*.yaml`); the runner still reads legacy JSON.
3. The method and package name is CoSpRo; phase 1 renames every `crp_*` identifier.
4. `ARCHITECTURE_REVIEW.md` stays in Russian as a personal working note.
