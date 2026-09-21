# CoSpRo refactoring: plan, decisions and handoff log

Last updated: 2026-09-20. Branch `refactor` (worktree `E:\Programming\Spur_SpLiCE-refactor`).

This document is the single source of truth for the refactor. It records what the code looked like, what hurt,
which designs were weighed, which were kept and why, what is done and what comes next. It is written so that a
different engineer or model can continue the work from here without the original conversation.

`ARCHITECTURE_REVIEW.md` (Russian, personal working note) holds the first review. Everything needed to continue is
restated here in English.

---

## 0. How to continue this work

### 0.1 Where things are

| Item | Location |
|---|---|
| Refactor branch | `refactor` on `origin` (GitHub `GrGrie/Spur_SpLiCE`) |
| Refactor checkout on the author's PC | `E:\Programming\Spur_SpLiCE-refactor` (a `git worktree` of the main checkout) |
| Main checkout (author keeps working on `main`) | `E:\Programming\Spur_SpLiCE` |
| Local Python | conda env `grgrie-train`: `C:\Users\GrGrie\miniconda3\envs\grgrie-train\python.exe` (torch 2.11 nightly) |
| Cluster Python | conda env `grgrie-train` on the SLURM cluster (torch 2.5.1+cu118), loaded by `scripts/load_splice_cluster_env.sh` |
| Local Waterbirds copy | `E:\Datasets\waterbirds` (used for post-hoc labels and dashboard thumbnails) |
| Cluster scratch | `/scratch/xar68reb/CoSpRo/` (checkpoints, feature caches, SpLiCE caches) |

The refactor merges into `main` after the last phase. Every phase lands as small commits on `refactor` and is
pushed. The author pulls `refactor` on the cluster to run cluster checks.

### 0.2 Working rules

- Read `AGENTS.md` first. It holds the language and style rules (English, no comma before "and", no word
  "nuisance", affirmative statements), the SLURM rules (V100, at most 5 CPUs, at most 8G per CPU, one launcher per
  cluster task, `announce_results` banner) and the storage rules (`/home` quota, binaries to scratch, JSON to
  `outputs/`).
- Commits and PR descriptions carry **no** Claude co-author trailer and no "Generated with" line.
- One commit per logical step. Run the full test suite before every commit.
- Every change that touches command lines, stored configuration or artifacts first adds or checks a pinning test
  (see section 3), then changes the code.

### 0.3 Commands

```bash
# local tests (the Windows sandbox needs a writable basetemp)
python -m pytest -q --basetemp=<writable temp dir>

# simulate a Slurm job environment for the launcher tests
SLURM_JOB_ID=1 python -m pytest -q

# regenerate golden snapshots after an intended behaviour change, then review the diff in tests/golden/
SPUR_SPLICE_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_*.py

# cluster: full suite plus all training modes on the GPU
sbatch scripts/run_golden_smoke.sbatch

# diagnostics (cluster computes, any machine renders)
sbatch scripts/run_cospro_diagnostics.sbatch
python -m cospro.diagnostics dashboard outputs/reports/cospro_diagnostics/waterbirds/diagnostics.json \
    --data-folder E:/Datasets --gallery-graph cospro --output tmp/diagnostics.html
```

### 0.4 Tool gotchas met during this refactor

| Gotcha | Symptom | Remedy |
|---|---|---|
| Backslashes inside `bash` heredocs sent through the agent's Bash tool | `\n` and `\0` arrive mangled; string replacements silently miss | Write edit scripts to a file with the file-writing tool, then run them. Every scripted replacement asserts that its anchor exists. |
| `core.autocrlf=true` on the author's Windows machine | Worktree files carry CRLF; anchors with `\n` stop matching | The edit helper normalizes to `\n`, edits, restores the original newline style. Fixtures whose bytes enter a fingerprint are marked `-text` in `.gitattributes`. |
| `bash` on Windows resolves to the WSL relay in System32 | Launcher tests fail with `execvpe(/bin/bash) failed` | `tests/test_training_launchers.py` uses Git Bash on Windows. |
| Tests running inside a Slurm job | `SLURM_JOB_ID` switches launchers to their cluster branch; real training starts on a busy GPU | Launcher tests drop every `SLURM_*` variable and use an empty `SPUR_SPLICE_OUTPUT_ROOT`. |
| Two scratch variables | `SPUR_SPLICE_SCRATCH_ROOT` overrides the legacy `SPUR_SPLICE_ARTIFACT_ROOT` | Tests set `SPUR_SPLICE_SCRATCH_ROOT`. |
| `pytest` `--basetemp` | pytest creates only the last path component | Launchers create the parent directory; `run_golden_smoke.sbatch` uses node-local `$TMPDIR`. |
| `torch.backends.cudnn.version()` on a CUDA build without a visible GPU | `ValueError: min() arg is an empty sequence` | Fixed in `spur_splice._cudnn_version`. |
| `reportlab` | Only the paper figure tools need it | `pytest.importorskip` in `tests/test_submission_figure.py`; extra `paper`. |
| Browser pane and `file://` URLs | The pane refuses local files | Serve `tmp/` with `python -m http.server` (see `.claude/launch.json` in the main checkout). |
| Float bytes across platforms | Graph JSON hashes differ between Windows and Linux | Golden graph test compares structure with a tolerance; training snapshots are per operating system. |

---

## 1. Goals and constraints

### 1.1 Goals

1. Adding a dataset, a concept vocabulary, a training method or an ablation touches one file plus a registry line.
2. Every hyperparameter has one default in one place; named presets cover the recurring non-default setups.
3. Concept groups and teacher graphs come with quantitative metrics and visual evidence, comparable across settings.
4. Paper numbers stay reproducible through every step; runs started before a step resume after it.

### 1.2 Constraints

- **SLURM.** All training, probing, caching, grouping and graph building runs on the cluster through `sbatch`.
  Each new cluster entry point ships with its launcher in the same commit. Resources: one V100 (an A40 also works),
  at most 5 CPUs, at most 8G per CPU. Every launcher prints `Results expected in: ...` at the top of its `.out` file.
- **Storage.** `/home` has a 100 GB quota (about 70 GB used). Checkpoints and tensors go to
  `/scratch/xar68reb/CoSpRo/` (slow HDD, write once per stage). Compact JSON results go under `outputs/` and
  synchronize through Git between the cluster, the laptop and the PC.
- **W&B.** Every SSL run logs to W&B; worst-group accuracy (WGA) is the headline metric.
- **Label-free protocol.** Concept grouping, graph building and SSL use no labels. Post-hoc label metrics guide
  manual selection only; automatic tuning uses label-free metrics.

---

## 2. Status at a glance

| Phase | Content | Status | Main commits |
|---|---|---|---|
| 0 | Safety net: golden tests, packaging, CI, cluster smoke launcher | done, cluster-verified (jobs 23823481, 23823905) | `0c97c92`, `37d6096`, `08739b9` |
| 1 | Conventions: SLURM lint, template, single sources for defaults and cluster identity, CRP→CoSpRo rename, layout of paper and tools, dead code | done | `567d70c` … `af0a346` |
| 2 | Concept-group and teacher-graph diagnostics with dashboard | done; cache tier awaits a cluster run | `177fced` |
| 3 | Typed configuration, presets, sweeps | done | `aa8e738` … (see phase 3 notes) |
| 4 | `TrainingMethod` seam and W&B metric contract | done | `c0efd8b`, metric contract commit |
| 5 | Trainer, callbacks, storage policy | done | trainer seam commit |
| 6 | Dataset adapters | done | dataset seam commit |
| 7 | Linear probe as a library function | done | probe seam commit |
| 8 | `cospro/` pipeline package, neighbour index, selection rule | done; dictionaries deferred | pipeline split commit |
| 8b | Concept dictionaries, dataset downloads | done; cluster check `verify_concept_dictionary.sbatch` pending | dictionary commit |
| 9 | Package layout move | planned | – |

Other commits on the branch: `b51575f` AGENTS.md rules, `00a4c8b` paper commas, `3f2a5fb` YAML manifests,
`8ae09d5` review and plan.

---

## 3. Invariants and how they are pinned

These are the stored identities that link code to past results. Breaking one silently orphans checkpoints,
breaks resume or changes paper provenance. Each has a test.

| Invariant | Where it lives | Why it matters | Pinned by |
|---|---|---|---|
| Runner command identity | `experiments/runner.py` compares a new command with `command.json` before `reuse` or `resume` | A changed command blocks resuming an interrupted 500-epoch run | `tests/golden/commands.json` (92 commands); `_comparable` compares `--manifest_path` by file stem |
| Manifest fingerprint | `manifest_fingerprint` (SHA-256 of the parsed manifest) in `execution.json` | Identifies which manifest produced a run | YAML files parse to the same mapping as the JSON originals (checked at conversion) |
| Run storage name | `spur_splice.format_storage_name` hashes `vars(args)` minus a list of excluded keys | Selects the checkpoint folder; a changed hash makes a resumed run write into a new folder and leaves old checkpoints undeleted in `/home` | `tests/test_storage_identity.py`; historical option names through `splice.compat.with_legacy_option_names` |
| Flat training namespace | `vars(args)` feeds the storage hash, `args.json`, `run.json` config, the W&B config and the checkpoint `opt` payload | Every key rename or addition changes all four | Phase 3 adds `tests/golden/resolved_configs.json` |
| Checkpoint graph fingerprint | `opt["cospro_graph_fingerprint"]` or the legacy `opt["crp_graph_fingerprint"]` | Resume refuses a changed teacher graph | `tests/test_reproducibility.py` covers both generations |
| Artifact type strings | `splice_crp_v2_teacher_graph`, `splice_crp_v3_teacher_graph`, `splice_crp_concept_groups`, `cospro_teacher_graph_v3` | Validators accept old graphs and groups | `splice/compat.py` constants |
| Sample IDs | `<dataset>:<metadata row>` | Align caches, groups, graphs and labels | `validate_teacher_graph`, diagnostics label tests |
| Numerical behaviour | training loop, probe, graph builder | Paper numbers | `tests/golden/training_runs.<os>.json`, `tests/golden/teacher_graph.json` |
| Launcher contract | `#SBATCH` headers, `announce_results` | Cluster rules and discoverable results | `tests/test_slurm_launchers.py` |

Known consequence already accepted: the W&B config of runs after phase 1 carries `cospro_temperature` and other
`cospro_*` keys; runs before carry `crp_*`. W&B filters on these fields split the two generations.

---

## 4. Problem catalogue (what this refactor fights)

Each entry: evidence, consequence, phase that addresses it. File references point at the state before the refactor
unless marked.

**P1. One string decides the training method everywhere.** `splice_mode` appeared 64 times in 10 files:
`spur_splice.py` (choices, validation, loader building, regularizer building, config recording), `ssl_loop.py`
(branches on `requires_crp_indices` / `requires_concept_transfer` attributes), each dataset adapter (whitelists of
modes and a frozen-distillation branch inside `make_waterbirds_ssl_loader`), `run_cospro_pipeline.py`,
`export_wandb_runs.py`. Adding a method or an ablation variant meant editing all of them. The two regularizers had
different call signatures, so the loop had to know which one it held (a Liskov violation). → phase 4.

**P2. Three sources for the same default.** SimCLR temperature was 0.5 in argparse, 0.05 in manifests, 0.05 in
`run_training.sbatch` and 0.05 in `run_cospro_pipeline.sh`; batch size 256 / 128; CoSpRo temperature 0.1 / 0.25.
→ phase 1 removed the shell copies; phase 3 turns the remaining two meanings into a base default plus a named preset.

**P3. The flat `argparse.Namespace` is the interface of every function.** About 110 fields; each function reads a
handful; `parse_args` and `build_ssl_loader` mutate it (`relational_graph_empty`, `teacher_graph_*`,
`run_recorder_instance`), so call order became part of the interface. `build_linear_probe_args` copies 35 fields into a
fake namespace to call a CLI entry point. → phases 3 (typed sections), 5, 7.

**P4. `spur_splice.py` held eight responsibilities** (CLI, validation, naming, loader building, model building, epoch
loop, retention and cleanup, W&B and run records) in 1243 lines. → phases 3, 4, 5.

**P5. Dataset adapters are shallow copies.** Waterbirds, CelebA and Spur-CIFAR10 each carry about 250 lines with
three near-identical loader factories. Registry entries are untyped dictionaries plus alias tables in Python and a
`case` statement in bash. → phase 6.

**P6. `splice/cospro.py` was a 1528-line module** with cache validation, grouping, neighbour search (exact and LSH),
auditing, null controls, graph assembly and a CLI; `build_teacher_graph` alone spans about 300 lines. Vocabularies are
special-cased by name. → phase 8.

**P7. Folder names contradicted roles.** Library code under `experiments/spurious_eval/`, the vendored SpLiCE library
mixed with the CoSpRo method and with infrastructure in `splice/`, pipeline stages next to one-off migrations in
`scripts/tools/`, the paper in the repository root. → phase 1 moved the paper and the tools; phase 9 finishes.

**P8. Hard-coded personal paths and identities** (`/scratch/xar68reb/CoSpRo`, `/home/xar68reb/Datasets`,
`gsgrechkin-rptu`) in library code. → phase 1 (`splice/settings.py`, `load_splice_cluster_env.sh`).

**P9. CRP-era names mixed with CoSpRo names** (`CrpAuditConfig`, `args.crp_temperature`, shim modules, shim
scripts). → phase 1 (`splice/compat.py` is now the only place that knows the old names).

**P10. No executable safety net.** No pytest in the environment, launcher tests broke on Windows, no CI, no numerical
regression tests. → phase 0.

**P11. Concept groups and graphs could only be judged by reading HTML pages.** No metric compared two grouping
configurations or a graph with its baselines. → phase 2.

**P12. Cluster logs did not say where results land.** → phase 1 follow-up (`announce_results`).

---

## 5. Principles adopted

1. **Strangler fig.** New typed modules grow next to the old code; the old surface becomes a thin adapter over them
   and disappears only when nothing stored depends on it. A big-bang rewrite was rejected because the flat namespace
   is part of four stored identities (section 3).
2. **Pin before change.** Every step that could move a stored identity first adds a snapshot or identity test.
3. **The CLI surface is a public contract.** Existing flags keep their names and meanings; new behaviour arrives as new
   options. This keeps runner command identity and manifests stable.
4. **One compatibility module.** `splice/compat.py` holds every historical name. New code uses current names only.
5. **Pure validation, separate resolution.** Configuration checks that need no I/O live with the configuration types;
   checks that touch the filesystem (graph exists, target bank loads) live in a resolution step.
6. **Two adapters make a seam.** A plug-in interface is introduced where two or more implementations exist today
   (training methods, neighbour search, datasets, dictionaries).
7. **Small commits, pushed often,** so the author can pull and run cluster checks at any point.

---

## 6. Phases

### Phase 0 — Safety net (done)

**Delivered**

- `pyproject.toml` with project metadata (package `cospro`), extras `dev`, `paper`, `diagnostics`; `setup.py` removed;
  `requirements.txt` gained `pyyaml`.
- Golden tests (`tests/golden_support.py` builds every fixture deterministically):
  - `test_golden_commands.py`: all runner commands for every manifest, seed, arm and locked-test variant.
  - `test_golden_graph.py`: grouping and graph structure on a synthetic cache (neighbour indices exact, floats with
    tolerance).
  - `test_golden_training.py`: two CPU epochs of each training mode (SimCLR, CoSpRo, CoSpRo KL-only, frozen concept
    distillation, LA-SSL) on a synthetic Waterbirds (80 images, 4 groups, background colour as the spurious
    attribute). Snapshots per operating system: `training_runs.windows.json`, `training_runs.linux.json`.
    `SPUR_SPLICE_GOLDEN_DEVICE=cuda` runs the same modes on a GPU with AMP; the GPU run checks completion only.
- `.github/workflows/tests.yml` (CPU, triggers on `main` and pull requests).
- `scripts/run_golden_smoke.sbatch`, `scripts/freeze_environment.sbatch`.

**Decisions and rejected alternatives**

- Graph golden by file hash: rejected, float serialization differs across platforms. Adopted: structural comparison
  with tolerance.
- One training snapshot for all platforms: rejected after the first cluster run; BLAS differences exceed tolerance.
  Adopted: per-OS snapshots, created on first run and committed.
- GPU numeric comparison: rejected, AMP and cuDNN are nondeterministic; the GPU run proves the modes run.
- The synthetic dataset first gave 100% WGA everywhere, which pins nothing. The class mark was made small and weak
  so modes produce different WGA values.

**Follow-ups**: `scripts/freeze_environment.sbatch` has not run yet; the cluster (torch 2.5.1) and the local PC
(torch 2.11 nightly) use different torch versions.

### Phase 1 — Conventions and quick wins (done)

**Delivered**

1. `tests/test_slurm_launchers.py` enforces resources, log paths and the `announce_results` banner;
   `scripts/_template.sbatch`; `scripts/announce_results.sh`.
2. Defaults: `run_cospro_pipeline.sh` forwards only environment variables that are set; the Python CLI owns every
   pipeline default (equivalence checked for three scenarios). `cache_splice_dataset.sh` lost its SpLiCE literals.
3. Cluster identity: `splice/settings.py` (Python) and `scripts/load_splice_cluster_env.sh` (bash) hold the scratch
   root, the dataset root and the W&B entity, each overridable by environment variable.
4. CRP→CoSpRo rename of classes, functions, options and tests; `splice/compat.py`; shim modules and scripts removed.
5. `paper/` (manuscript, style, `paper_results.json`), `docs/notes/` (dated notes), `tools/maintenance/`,
   `tools/paper/`.
6. `experiments/spurious_eval/training/reproducibility.py` shares `seed_worker`, `make_dataloader_kwargs` and
   `preserve_rng_state`; dead branches removed.

**Decisions and rejected alternatives**

- Replacing the pipeline's environment-variable interface by a YAML pipeline config: postponed to phase 3/8. The
  author uses `EPOCHS=10 bash scripts/run_cospro_pipeline.sh`; the forwarding table keeps that habit with one source of
  defaults.
- Renaming `crp_graph.json`, the `waterbirds_crp` study or arm names: rejected, they are provenance for paper results.
- Removing `--cudnn_benchmark` (it only forbids `true`) and `CoSpRoAuditConfig.indegree_factor`: rejected, both feed
  stored identities (storage hash, W&B run name, graph configs).
- Merging the trainer's and the probe's `set_seed`: rejected, the trainer also disables cuDNN and enables
  deterministic algorithms; merging would change probe behaviour when the probe runs standalone.
- Command identity by manifest path: replaced by comparison by file stem, since every manifest value is expanded into
  its own flag.
- Storage names: adopted `with_legacy_option_names` so the hash stays computed over the historical option names.
  Rejected: hashing the new names (orphans checkpoint folders of resumed runs).
- `cleanup_local_outputs.py` keeps a literal scratch path because it runs as a standalone file without the package.

**Deferred from phase 1 to phase 3**: the CoSpRo student values injected by `run_training.sbatch` in automatic graph
mode. **Open follow-up**: a style pass over older comments and documents for the `AGENTS.md` rules.

### Phase 2 — Diagnostics (done, cache tier pending on the cluster)

**Delivered**: package `cospro/diagnostics/`.

| Module | Role |
|---|---|
| `labels.py` | Only reader of `y` and `a`; maps `<dataset>:<row>` to labels and display names from the dataset metadata |
| `group_metrics.py` | Structure, text coherence, NPMI, silhouette, image coverage, bootstrap ARI, cross-configuration ARI and VI, post-hoc AUC selectivity and fragmentation |
| `graph_metrics.py` | Structure, edge Jaccard, null-test selection; post-hoc transition matrices, balanced counterfactual and same-class rates, minority reach, confidence calibration, per-group edge semantics, stratified edge samples |
| `evaluate.py` | Builds one JSON record (schema `cospro-diagnostics-v1`) for a sweep directory plus named graphs |
| `dashboard.py` | Self-contained HTML (matplotlib PNGs, JPEG thumbnails) |
| `__main__.py` | `python -m cospro.diagnostics evaluate|dashboard` |

Launcher `scripts/run_cospro_diagnostics.sbatch` (CPU, finds the single SpLiCE cache under scratch).
Record in Git: `outputs/reports/cospro_diagnostics/waterbirds/diagnostics.json` (label and graph tiers).

**Decisions and rejected alternatives**

- Lift over a degree-matched random graph as the headline: rejected. Random neighbours of a minority anchor mostly come
  from the majority group of the same class, so the random balanced counterfactual rate (0.25) beats every real graph
  while the random balanced same-class rate collapses to 0.50. Adopted: balanced counterfactual read together with
  balanced same class, compared with the raw-CLIP graph at the same degree.
- Minority groups as "below the mean group size": rejected, it marked waterbird/water (1057 of 4795) as minority.
  Adopted: below half of the mean.
- Random edge samples for the gallery: rejected, they show majority-to-majority edges. Adopted: per concept group the
  two highest-confidence edges plus two edges from minority anchors.
- Two launchers (grouping, graph) and a separate local rendering tool from the first plan: merged into one launcher and
  one module entry point.
- Not built yet: the UMAP concept map, null-test strip plots, group cards with top-activating thumbnails (the older HTML
  reports still render group cards).

**Findings on Waterbirds (post-hoc, 2026-09-18)**

- The 31 grouping configurations barely change the partition: at most 20 composite groups of two or three concepts
  (44 at text 0.5 / coactivation 0.1, which has no graph).
- The null test passes 425–450 of about 500 groups. Sweep graphs (no selection cap) share 6% of edges with raw CLIP and
  score balanced counterfactual 0.18–0.19, below raw CLIP (0.20). The paper graph keeps the top 12 groups and scores
  0.23 with minority reach 59% (raw CLIP 52%, semantic SpLiCE 54%). **Group selection is the decisive lever.**
- Confidence is calibrated in the paper graph: counterfactual rate 1.4% in the lowest decile, 8% in the highest;
  baselines stay flat.
- Background groups ({Bamboo}, {Sunset}, {Autumn}) produce most counterfactual edges; owl groups link images that
  share class and background. The gallery shows Bamboo edges joining the same bird species on water and on bamboo.

**Choosing between groupings (protocol agreed with the author)**

1. Label-free validity filters: maximum group size cap, NPMI floor, bootstrap ARI ≥ 0.8, positive audit yield.
2. Graph proxy (post-hoc, manual): balanced counterfactual rate with balanced same class and minority reach, against
   raw CLIP. A cleaner variant scores the proxy on a validation-split cache and graph.
3. 100-epoch single-seed runs for the top five, compared on validation WGA.
4. 500 epochs over four seeds for the winner.
5. Spearman correlation between steps 2 and 3 decides whether step 2 becomes the default selector.

Given the findings, the next sweep should vary selection (`max_selected_groups`, `null_quantile`) rather than the
grouping thresholds.

### Phase 3 — Typed configuration, presets and sweeps (done)

**Delivered (2026-09-18)**

- `aa8e738` `.gitattributes` keeps every JSON file LF. Found while pinning: with `core.autocrlf` the
  Windows checkout held CRLF JSON, so `graph_fingerprint` of the same teacher graph differed between
  Windows (3 381 149 bytes for `crp_graph.json`) and the cluster (3 198 492 bytes). Resume across
  machines would have refused the graph. After pulling this change on Windows, re-checkout JSON
  files once: `git checkout -- '*.json'` in a clean tree.
- `1b2e4bf` `tests/test_config_resolution.py` + `tests/golden/resolved_configs.json` (pinning first).
- `6c5ab7a` `cospro/config/` (`options.py`, `training.py`, `presets.py`): fourteen sections, generated
  parser, `normalize_training_options` (filesystem-free checks and derived values),
  `TrainingConfig.from_namespace`, `--preset`. `spur_splice.parse_args(argv)` keeps the filesystem
  checks and naming; `spur_splice.training_config(args)` returns the typed view.
- `2527bc2` shared defaults: pipeline (grouping and audit from `CoSpRoAuditConfig`, student from the
  preset), linear probe (`LINEAR_PROBE_DEFAULTS` derived from `ProbeOptions`), `run_training.sbatch`
  (`--preset cospro_student`; the five `COSPRO_*` environment knobs are gone).
- Runner `sweeps` block (`expand_sweeps` in `experiments/runner.py`) with tests.

**Behaviour notes**

- Filesystem checks (target bank, teacher graph) now run after all filesystem-free checks. With a
  single error the message is identical; with two errors of different kinds the reported one can
  differ from before.
- `--preset` stays out of the resolved namespace, so a preset run shares its storage name with the
  spelled-out equivalent (tested).
- Sweep arm names come from option values; values that cannot form an arm name (paths) need explicit
  arms. Unknown option names in a grid fail at manifest load.

**Original design (kept for reference)**

**Problem addressed**: P2, P3 (first half), P4 (CLI and validation part).

**Design**

1. `cospro/config/training.py` declares the training configuration as frozen dataclass sections. Every field name equals
   today's argparse `dest`, so the flat projection is identical to today's namespace:
   `ExperimentIdentity` (study, arm, attempt, run record, manifest path), `StorageOptions` (checkpoint and artifact
   directories, retention), `DataOptions`, `ModelOptions`, `OptimizerOptions`, `SSLOptions`, `RuntimeOptions` (device,
   AMP, channels-last, cuDNN), `ProbeOptions`, `TrackingOptions`, `CoSpRoOptions`, `ConceptTransferOptions`,
   `LaSSLOptions`. Field metadata carries the CLI flags (including aliases such as `--crp_temperature` and
   `--ssl-crop-min`), help text, choices and the argument style (value, store-true flag, optional boolean).
2. `build_parser()` generates the argparse parser from the sections, in today's order. `spur_splice.parse_args` becomes:
   parse → `TrainingConfig.from_namespace` → pure validation (`ConfigError` with today's messages, reported through
   `parser.error`) → resolution (filesystem checks, graph fingerprint, target bank, derived fields such as warm-up,
   `n_cls`, run names and storage name) → flat namespace.
3. `cospro/config/presets.py` defines named presets. `cospro_student` = batch 128, 4 workers, SimCLR temperature 0.05,
   `cospro_relational`, relation weight 0.5, relation temperature 0.25. `spur_splice.py --preset NAME` applies a preset
   before explicit flags; the preset name stays out of the namespace, so a preset run and the equivalent explicit flags
   share one storage name. `run_training.sbatch` uses `--preset cospro_student` instead of numbers;
   `run_cospro_pipeline.py` takes its student defaults from the same preset and its grouping and audit defaults from
   `CoSpRoAuditConfig`.
4. `ProbeOptions` defaults feed the trainer's `linear_*` options and `linear_probe.py` (its parser and
   `normalize_args`, which today repeat the same defaults twice).
5. Manifest sweeps: an optional `sweeps` block expands into arms inside `load_manifest`:
   ```yaml
   sweeps:
     weight:                   # prefix of the generated arm names
       base: cospro            # arm whose arguments the grid overrides
       grid:
         splice_weight: [0.25, 0.5, 1.0]
         cospro_temperature: [0.1, 0.25]
   ```
   Generated arm names are `weight__splice_weight-0.25__cospro_temperature-0.1`. Existing manifests have no `sweeps`
   block, so their commands and fingerprints stay unchanged.

**Pinning first**: `tests/test_config_resolution.py` with `tests/golden/resolved_configs.json`, captured from the code
before the change: the flat namespace for every runner command of every manifest (graphs copied to a temporary output
root, a fake target bank under a temporary scratch root), plus standalone variants (defaults, LA-SSL, Spur-CIFAR10,
final test, legacy `--crp_*` flags, boolean spellings) and a list of invalid argument sets with their expected error
messages. Volatile entries (paths, runtime versions, storage name) are normalized; the storage name stays pinned by
`test_storage_identity.py`.

**Alternatives considered**

| Alternative | Verdict | Reason |
|---|---|---|
| Hydra / OmegaConf | rejected | New CLI syntax and run-directory conventions would change runner commands and `run.json`; interpolation hides values; YAML manifests plus dataclasses cover the need |
| pydantic models | rejected for now | Extra dependency on the cluster env; frozen dataclasses with explicit checks suffice. Revisit if validation grows. |
| Nested configuration in `vars(args)` | rejected | Changes the storage hash, `run.json`, W&B config and checkpoint `opt` of every run (section 3) |
| `command.json` schema v2 with `--set section.key=value` overrides | rejected | The CLI surface stays unchanged, so command identity needs no new schema |
| Presets as manifest includes (`extends:`) | deferred | Presets serve standalone runs; manifests stay explicit records of every value |
| Declarative option table without dataclasses | rejected | Loses typed sections that phases 4, 5 and 7 consume |

**Done when**: the resolution snapshot and all golden tests pass unchanged; `run_training.sbatch`,
`run_cospro_pipeline.py` and `linear_probe.py` carry no student or probe literals; a sweep test expands a grid into
the expected arms and commands.

### Phase 4 — `TrainingMethod` seam and W&B metric contract (done)

**Delivered (2026-09-20)**

- `cospro/methods/`: `base.py` (interface, registry, `LoaderContext`, `LossTerms`), `simclr.py`,
  `relational.py`, `concept_transfer.py`, `la_ssl.py`. A method owns `wrap_loader`, `extra_loss`,
  `diagnostics`, `provenance`, `input_artifacts`, `sampling_state` and `refresh_sampling`.
- Selection is data-driven: each class declares the `splice_mode` values it serves (`modes`) or
  `requires_la_ssl`; `method_for` resolves them and registration rejects two claims on one mode.
- Dataset adapters lost their mode whitelists and the Waterbirds frozen-transfer branch. The
  frozen method rebuilds the loader around `FrozenConceptTransferSubset` using the same subset,
  transform and generator, so sampling and augmentation stay identical.
- `ssl_loop` has one call path and creates a meter per diagnostic the method reports, keeping the
  historical `relational_<name>` metric names.
- `cospro/tracking/metrics.py`: canonical W&B keys logged next to the historical ones, with
  `probe/<split>/wga` as the run summary metric.

**Decisions**

- Graph provenance keeps its historical key names (`teacher_graph_artifact`, `relational_graph_empty`,
  ...); `spur_splice` copies `method.provenance()` onto the namespace, so `run.json` and the W&B
  config look exactly as before.
- Canonical metric keys go to W&B only. Adding them to `run.json` would change every metric event
  and the golden training snapshots for no reader today; phase 5 revisits this when logging moves
  into callbacks.
- The three run-name formatters still branch on `splice_mode`. Their strings (`cospro-relational`,
  `crp-v2-relational`, `la-ssl`, `concept-transfer`) are part of stored identities, so they stay.
- Rejected: dispatching on capability flags (`requires_graph_indices`), subclassing `SimCLRLoss`
  per method and moving LA-SSL into a sampler option (it stays a method because it wraps the
  loader and saves state).

**Follow-up**: put the phase 2 graph metrics (balanced counterfactual, minority reach) into the
W&B config of relational runs, so W&B can scatter graph quality against final WGA.

**Original design (kept for reference)**

**Design**

- Protocol `TrainingMethod`: `name`, `Options` (its dataclass section from phase 3), `wrap_loader(loader, dataset)`,
  `extra_loss(batch, embeddings, model, epoch) -> LossTerms`, `provenance() -> dict`, `input_artifacts() -> list[Path]`,
  `state_dict()` / `load_state_dict()` for resume.
- Adapters: `SimCLROnly`, `CoSpRoRelational` (graph sampler and relational KL), `FrozenConceptDistill` (target bank and
  cosine head), `LaSSL` (learning-speed sampler). Registry by name with a decorator.
- `ssl_loop` calls `method.extra_loss` through one path. Dataset adapters stop receiving `splice_mode`.
- W&B metric contract (`cospro/tracking/metrics.py`): stable keys `probe/val/wga`, `probe/val/avg_acc`,
  `probe/val/group_acc/<group>`, `train/loss/<term>`, `method/<name>/<diagnostic>`;
  `wandb.define_metric("probe/val/wga", summary="max")` plus the last value. `run.json` stores the same keys. A key map
  keeps the current human-readable keys ("Last linear val worst-group acc") for `export_wandb_runs.py`,
  `collect_results.py` and `build_paper_results.py`.
- Teacher-graph diagnostics from phase 2 (balanced counterfactual, minority reach) go into the W&B config of relational
  runs, so W&B can scatter graph quality against final WGA.

**Open questions**: keep logging the old W&B keys next to the new ones for one study (recommended) or switch at once;
whether LA-SSL is a method or a sampler option (it keeps the plain SimCLR loss).

**Rejected**: subclassing `SimCLRLoss` per method (couples loss and sampling); keeping capability flags
(`requires_graph_indices`) as the dispatch mechanism.

### Phase 5 — Trainer, callbacks, storage policy (done)

**Delivered (2026-09-20)**

- `cospro/training/`: `trainer.py` (`Trainer.fit`, the epoch loop and the two RNG rules),
  `state.py` (`TrainingState`), `storage.py` (`StoragePolicy`), `callbacks.py` (`Callback`,
  `EpochReport`, `RankMetrics`, `RunRecordLogger`, `WandbLogger`, `PeriodicProbe`,
  `CheckpointPolicy`).
- `spur_splice.py` went from 778 to 490 lines and now parses, builds and fits. `main` is 45 lines:
  open the run record, build the storage policy and the W&B run, build the state, build the
  callbacks, `fit`, then finish W&B, clean up and write the status.
- A callback holds what it needs from construction, so the hooks carry only the epoch report or the
  failure and the trainer never passes itself around.
- `EpochReport` carries the raw epoch metrics, the learning rate and whatever diagnostics the
  callbacks add; `cospro.tracking.epoch_payload` turns it into the historical metric event, which
  both the run record and W&B store. `ssl_loop.log_rank_metrics` disappeared into this split.
- `tests/test_trainer.py` runs ten epochs with a stubbed training step and asserts the peak number
  of live checkpoint files: `--checkpoint_keep_count` recovery checkpoints, one more while a probe
  reads its temporary one. It also pins that an observing callback cannot move the training RNG
  stream and that a failing epoch reaches `on_failure` and the caller.
- Golden training snapshots, storage names and `run_status.json` are unchanged.

**Decisions**

- The callbacks run inside `preserve_rng_state`, so the whole dispatch is observational by
  construction. Before, each observer had to remember to preserve state on its own.
- W&B closes after `fit` returns instead of in `on_train_end`, so the final linear probe still logs
  into the run. `WandbLogger` owns the run, its identity and its finish state, and it writes into
  the status dictionary `run_status.json` persists.
- `StoragePolicy` and `CheckpointPolicy` stay separate: the policy answers where a file goes and how
  long it stays, the callback answers when the trainer writes one.
- The probe deletes its temporary checkpoint in a `finally`, so a failed probe no longer leaves one
  behind. `save_freq = 0` no longer divides by zero.
- Removed as dead: `get_probe_score`, `maybe_run_periodic_probe`, `maybe_run_final_probe`.
- **Rejected**: PyTorch Lightning or Accelerate. They own RNG handling, checkpoint layout and logging; this project needs
  exact RNG isolation for observational probes, a custom storage split between `/home` and scratch and record
  attestations.

**Follow-up**: the linear probe still travels through the 35-field namespace bridge
(`build_linear_probe_args`); phase 7 replaces it with `evaluate_probe` and `PeriodicProbe` then
calls that directly.

### Phase 6 — Dataset adapters (done)

**Delivered (2026-09-20)**

- `experiments/spurious_eval/datasets/base.py`: `SpuriousDataset` (the contract plus the shared
  group report), `DatasetConfig`, `register_dataset`, `dataset_transforms` and one `build_loader`
  for the roles `ssl`, `rank`, `probe_train` and `probe_eval`. `build_probe_loaders` reads the
  probe's two splits from one built dataset.
- The three adapters keep only what differs: metadata reading, `get_input` and their attributes.
  Together they went from 793 lines to 371 plus 211 shared; the three near-identical transform
  functions, the three triplicated `eval` methods and the nine loader factories are gone.
- `registry.py` is a facade over the decorator: importing it registers the adapters, and
  `dataset_names`, `canonical_dataset_name` and `dataset_class` read the registry. The spec
  dictionaries, the alias table and the second `DATASET_REGISTRY` with its CelebA spellings are
  gone.
- Image size decides the model: `default_model()` and `model_error()` replace
  `if dataset == "spur_cifar10"` in `cospro/config/training.py` and `run_cospro_pipeline.py`, with
  the historical message rebuilt from the name and the size.
- `run_training.sbatch` stops listing dataset names: it matches the requested spelling against the
  directories under `outputs/shared/`, which already carry the canonical names.
  `cache_splice_dataset.sh` points at the Python help instead of repeating the list.
- `docs/ADDING_A_DATASET.md` covers the adapter, the cache launcher, the diagnostics run and the
  manifest. `wilds_compat.py` now says in its docstring that it is vendored WILDS.
- `tests/test_dataset_adapters.py` pins the registry, the alias resolution, the atomic
  registration, the model rules and the split, transform and order of every role for every
  registered dataset. Golden training snapshots are unchanged.

**Decisions**

- `@register_dataset` takes no arguments: `name` and `aliases` are class attributes next to
  `num_classes` and `image_size`, so a decorator cannot disagree with the class it decorates.
- The registry is a dictionary of classes rather than of spec objects. A `DatasetSpec` beside the
  class would be a second place to look for the same facts.
- `read_metadata()` returning `path, y, a, split` stayed on paper. The adapters set the WILDS
  metadata attributes directly, which is what `get_subset` and the groupers already read; a second
  metadata representation would have to be kept in step with the first.
- The adapters stay in `experiments/spurious_eval/datasets/`; phase 9 moves the package to
  `cospro/data/` in one `git mv`.

### Phase 7 — Linear probe as a library function (done)

**Delivered (2026-09-20)**

- `cospro/evaluation/probe.py`: `evaluate_probe(encoder, dataset, ProbeOptions) -> ProbeResult`
  writes no file, opens no W&B run and reads no run identity. `persist_probe_result` writes the
  feature tensors to scratch and the result JSON beside the run and attests both;
  `log_probe_result` sends one probe to W&B; `probe_checkpoint` composes the three around an
  encoder loaded from a checkpoint.
- `ProbeOptions` holds what the measurement needs and `ProbeArtifacts` holds where its outputs go,
  so the split between measuring and storing is visible in the signatures.
- `linear_probe.py` went from 745 to 225 lines: it parses the command line into those two objects
  and dispatches. `build_linear_probe_args`, the 35-field namespace bridge, is gone;
  `spur_splice.probe_options` builds a typed `ProbeOptions` with 20 named fields instead.
- `tools/paper/evaluate_submission_checkpoints.py` calls the library directly and maps the frozen
  `PROTOCOL` onto `ProbeOptions`, leaving the stored protocol unchanged. Its `code_hash` now covers
  `cospro/` as well, so a lock still cannot survive a change to any file the evaluation runs
  through.
- A probe inside a training run logs into that run and never opens a second one; the standalone CLI
  still creates its own run when none is active.
- `tests/test_probe_library.py` pins that measuring writes nothing (torch.save and the JSON writer
  are made to raise), that persisting writes both artifacts with the right retention and emits the
  per-epoch events, and that logging sends both key sets. Golden training snapshots, which compare
  the final probe metrics, are unchanged.

**Decisions**

- The per-epoch probe events reach the run record from `persist_probe_result` rather than from the
  loop, which keeps `evaluate_probe` pure. Their order inside `run.json` is unchanged: the feature
  artifact, the epoch events, the result artifact, then the final event.
- The features are written after the measurement instead of before it, so a crashed probe leaves no
  half-attested tensor. A crash now loses the extraction work, which costs one forward pass.
- `evaluate_probe` keeps its prints, including the closing summary. A silent measurement would make
  cluster logs unreadable, and printing is not a side effect a caller has to undo.

### Phase 8 — Pipeline package, neighbour index, selection rule (done)

**Delivered (2026-09-20)**

- `cospro/pipeline/`: `config.py`, `cache.py`, `grouping.py`, `neighbors.py`, `audit.py`,
  `selection.py` and `graph.py`, each reading the artifact the previous stage wrote. The code moved
  verbatim; `splice/cospro.py` is a 66-line re-export shim for scripts written against the old path.
- The duplicate teacher-graph CLI at the bottom of the old module is gone. Nothing called it, and
  `scripts/tools/build_cospro_teacher_graphs.py` is the stage that writes canonical paths and
  provenance.
- `NeighborIndex` with `ExactNeighbors` and `LshNeighbors`, registered by the `--neighbor-backend`
  value that selects them, each reporting its own provenance. `tests/test_neighbor_index.py` holds
  the approximate index to the exact one: the same contract, true cosine similarities, determinism
  and full recovery of the exact neighbourhood on clustered features.
- `SelectionRule` splits the two decisions phase 2 showed matter: `accepts` judges one group against
  its null controls, `retain` ranks the survivors and applies the cap. `NullQuantilePass` is the
  paper protocol and the default, and the graph records which rule produced it.
- Golden graph and golden training snapshots are unchanged.

**Decisions**

- A selection rule is an argument to `build_teacher_graph`, not a `CoSpRoAuditConfig` field. That
  configuration is hashed into the group-checkpoint identity, so a new field would invalidate every
  audit checkpoint on the cluster. Naming rules on the command line is a deliberate later step,
  taken together with that cost.
- The two tests that reach into pipeline internals patch the module the function lives in. Patching
  the shim would have silently stopped biting.

### Phase 8b — Concept dictionaries and dataset downloads (done, cluster check pending)

**Delivered (2026-09-21)**, after a discussion with the author about what a public reproduction
needs. The author wants file dictionaries for ablations, will run one cluster check and wants the
datasets downloadable by a command.

- `cospro/pipeline/dictionary.py`: `ConceptDictionary` with the bundled `laion` and
  `openimages_v7` kinds plus `file`, any text file with one concept per line. A file dictionary is
  named by the SHA-256 of its selected words, which enters its cache directory name and its
  embedding cache, so an edited file never reuses stale embeddings. The bundled kinds keep their
  historical names, cache directories and cache provenance, so every existing cache stays valid.
- The vendored `splice.splice.load` gained `words` and `dictionary_id`. The embedding loop moved
  verbatim into `embed_concepts`, which the named and the explicit paths share. With a
  deterministic stand-in for the CLIP text tower, the old and the new loader produce equal
  dictionary tensors for Open Images (full and 300) and LAION (500), and a file holding the Open
  Images words produces the bundled tensor.
- `--splice-vocab file --splice-vocab-file PATH --splice-vocab-order head|tail` on the cache stage
  and the pipeline driver; `SPLICE_VOCAB_FILE` and `SPLICE_VOCAB_ORDER` on the pipeline launcher.
  `cache_provenance` is now the one definition the cache writes and the pipeline expects.
- `scripts/verify_concept_dictionary.sbatch`: builds the Waterbirds cache with the code before this
  phase (a git worktree), with this code and from a file holding the Open Images words, then
  compares them field by field through `scripts/tools/compare_splice_caches.py`. It prints
  `[verify] PASS` or `[verify] FAIL`.
- `scripts/tools/download_datasets.py` and `scripts/download_datasets.sbatch`: Waterbirds from the
  Stanford Group DRO archive, CelebA from the official release. The Waterbirds archive
  (SHA-256 `56c51b77...`) and its `metadata.csv` (`2f023b9d...`) were downloaded and checked here;
  the metadata hash equals the one the paper's cluster runs recorded. The CelebA annotation
  converter writes the author's CSV files byte for byte from the official format (checked by
  rebuilding the official format from those CSVs). The CelebA download itself needs `gdown` and
  Google Drive's goodwill, so it was not run here; `--celeba-archive-dir` covers a manual download.
- `tests/test_concept_dictionary.py`, `tests/test_download_datasets.py`. The pipeline-defaults
  section of `tests/golden/resolved_configs.json` gained the two new options, both `None`.

**Decisions**

- A word file has no comment syntax. LAION contains the concepts `#`, `##`, `#@` and `#...`, so a
  comment rule would silently drop real concepts; with one concept per non-blank line, a copy of
  either bundled vocabulary reads back exactly.
- Orders are named `head` and `tail` by what they keep. LAION's file runs from rare to frequent, so
  its natural order is `tail`; the bundled kinds refuse any other.
- A file dictionary records its file name and its SHA-256, never the machine-local path, so caches
  built on different machines from the same words compare equal.
- The pipeline stages stay in `scripts/tools/`; moving them to `cospro/cli/` is part of phase 9.

**Pending**: `sbatch scripts/verify_concept_dictionary.sbatch` on the cluster. Until it prints
PASS, treat file dictionaries as unverified against the real CLIP encoder.

### Phase 9 — Package layout (planned)

Target layout (package name `cospro`):

```text
cospro/            config/ data/ models/ methods/ training/ evaluation/ pipeline stages/ diagnostics/ tracking/ cli/
splice/            vendored SpLiCE (moves to third_party/splice/ with a NOTICE of local changes)
experiments/       manifests/ runner.py
scripts/           Slurm launchers
tools/             maintenance/ paper/
paper/ docs/ tests/ outputs/
```

`git mv` one package per commit with re-export shims at old import paths for one release; update `PROJECT_MAP.md`,
`docs/REPO_STRUCTURE.md` and `scripts/README.md`.

---

## 7. Decisions taken with the author

1. Post-hoc label metrics guide manual selection only; automatic tuning uses label-free metrics.
2. Manifests are YAML; the runner still reads JSON.
3. The method and package name is CoSpRo; historical CRP names live only in `splice/compat.py`.
4. `ARCHITECTURE_REVIEW.md` stays in Russian; this document is the English source of truth.
5. The refactor lives on `refactor` in a separate worktree and merges into `main` at the end.
6. Cluster checks can run on an A40 when no V100 is free.

## 8. Open questions

- W&B key migration strategy (phase 4).
- Whether the validation-split graph proxy (phase 2.7, step 2) becomes part of the paper protocol.
- Whether to run `scripts/freeze_environment.sbatch` and pin the cluster environment in `environment/`.
- When to merge `refactor` into `main` relative to the next training study.
