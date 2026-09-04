# Spur SpLiCE

> **Current project source of truth:** read
> [`PROJECT_GROUND_TRUTH.md`](PROJECT_GROUND_TRUTH.md) before changing the method,
> experiments, or paper. It defines the proposed label-free SpLiCE-CRP v2
> architecture and supersedes older SpLiCE-IU proposal text when they conflict.
> The CRP v2 frozen audit and relational SSL training path are implemented. There
> is still no reported CRP v2 result; matched real-data runs and controls remain
> necessary before making an empirical claim.

Spur SpLiCE studies whether sparse, language-aligned concepts from a frozen
OpenCLIP/SpLiCE model can identify and mitigate spurious correlations during
SimCLR training. The current experiments focus on Waterbirds and SpurCIFAR10.

The repository contains the training pipeline used by `project_report.tex`,
automatic concept discovery, routed-augmentation controls, correlation
regularization, reproducible checkpoint/resume support, and guarded validation
versus final-test evaluation.

## Environment

The cluster and home machine can use the same Conda environment:

```bash
conda activate grgrie-train
pip install -e .
```
RTX 50-series cards require a current Blackwell-capable PyTorch/CUDA build and
an up-to-date NVIDIA driver. Full training is run on the cluster through the
Slurm entry points below.

Dataset directories are intentionally excluded from Git. Pass their location
with `--data_folder` or edit the relevant Slurm `.conf` file.

## Main entry points

- `spur_splice.py` — SimCLR training and periodic/final linear probing.
- `linear_probe.py` — standalone target/spurious linear probing.
- `splice_cbm.py` — sparse concept-bottleneck baseline.
- `scripts/tools/discover_splice_spurious_concepts.py` — automatic concept discovery.
- `scripts/download_openimages_vocabulary.py` — download the Open Images V7 class-label dictionary only.
- `python -m scripts.tools.cache_crp_features` — aligned OpenCLIP/SpLiCE cache construction.
- `python -m splice.crp` — label-free CRP v2 frozen audit and teacher-graph export.
- `splice/crp_training.py` — graph-aware batching and confidence-weighted relational distillation.
- `scripts/tools/summarize_splice_scores.py` — selected-concept score summaries.
- `scripts/tools/render_report_figure.py` — report figure generation.
- `scripts/train_crp.sbatch` — CRP training entry point for Slurm.
- `CoBalT/scripts/prepare_crpv4_spatial.sbatch` — four-way CRPv4 spatial evidence preparation.
- `scripts/cache_openimages_crp.sbatch` — Slurm-only Open Images V7 cache preparation.
- `scripts/*.conf` — editable Slurm configurations for training, sweeps, cache,
  and CoBalT concept discovery.

### Open Images V7 vocabulary

The default SpLiCE vocabulary is the complete `openimages_v7` vocabulary
(`splice_vocab_size=-1`). The loader downloads only the official class description
CSV and writes the cleaned display names to `data/vocab/openimages_v7.txt`
or the selected cache root, and removes the temporary CSV. No Open Images pixels
are downloaded. A local repository copy can be created with:

```bash
python scripts/download_openimages_vocabulary.py --download-root ./data
```

No vocabulary flags are needed when building a new SpLiCE/CRP cache. Pass an
explicit vocabulary and size only for a deliberate ablation. Existing LAION caches
and graphs must not be reused with the default dictionary; rebuild them with a
distinct experiment variant. For a graph-only
Open Images CRP sweep, first build the cache once, then submit the 15-configuration
array. The cache step is intentionally separate so array jobs do not rebuild the
same frozen features concurrently:

Edit `scripts/cache_openimages_crp.conf` and
`scripts/prepare_crp_group_sweep.conf` for the dataset, cache path, vocabulary,
and graph prefix. Then submit the two jobs in order:

```bash
sbatch scripts/cache_openimages_crp.sbatch
# after the cache job finishes successfully:
sbatch scripts/prepare_crp_group_sweep.sbatch
```

The sweep keeps the original twelve audit settings and adds three semantic-family
settings. In those three settings, `coactivation_threshold=0` intentionally removes
the coactivation gate (SpLiCE codes are non-negative), `min_group_size=1` permits
both singleton and variable-size components, and `max_selected_groups=0` disables
the post-audit cap. They test whether mutually exclusive names such as different
bird species collapse into one semantic factor while leaving room for background
families; every factor must still pass the label-free null and semantic gates.
The `oi_v7` prefix makes every graph path distinct from historical LAION runs.
Each task writes its own graph directory and an English-only `graph_audit.html`.
The three CoBalT semantic-family variants also have a dedicated seed-0
launcher, so they can be prepared without rerunning the twelve conventional
grouping variants:

```bash
sbatch scripts/prepare_crp_semantic_family_sweep.sbatch
```

A separate pure-SpLiCE graph array builds the same two CRP settings without
loading CoBalT assignments or applying CoBalT-derived sample weights. It reuses the
same frozen Open Images SpLiCE cache as the CoBalT comparison:

```bash
sbatch scripts/prepare_crp_splice_only_sweep.sbatch
```

The five-task, 100-epoch screening array then runs a matched pure SimCLR control,
two pure-SpLiCE CRP graphs, and the corresponding two CoBalT-balanced CRP graphs
for `g3_t065_c015_k8` and `g2_t070_c020_k12`. It expects all four graph artifacts
to already exist and deliberately disables automatic rebuilding:

```bash
sbatch scripts/train_ssl_graph_shortlist.sbatch
```

This array is a mechanism screen, not a final comparison. Promote a candidate to
the full epoch and multi-seed protocol only after inspecting its W&B loss scale,
supported-anchor fraction, average accuracy, WGA, and per-group accuracy against
the matched SimCLR task.

For a one-command Windows/RTX quick screen of SpLiCE-only CRP against matched
SimCLR, run from a fresh repository clone in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_splice_only_ablation_windows.ps1
```

The script uses the existing `grgrie-train` Conda environment, checks CUDA,
downloads and materializes Waterbirds plus the Open Images vocabulary, builds a
no-CoBalT `g2_t070_c020_k12` graph, trains both 50-epoch SSL arms, performs
one 30-epoch final validation probe per arm, and writes `results.csv`, logs,
checkpoints, an HTML graph audit, and offline W&B records under
`outputs/windows_splice_only_ablation`. These defaults are a local mechanism
screen and are not directly comparable with the 500-epoch protocol.

After that run, add the matched current CRPv3/CoBalT arm with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_splice_only_ablation_windows.ps1 -IncludeCobalt
```

Completed SimCLR and SpLiCE-only logs are reused. The additional path trains the
current 50-epoch CoBalT discovery model, extracts fixed memberships, rebuilds the
same `g2_t070_c020_k12` configuration with CoBalT weighting enabled, and trains a
third SSL student with the same quick-screen settings. CoBalT changes graph
construction; applying it to an already fixed graph would be a no-op.

For a single CRP training run with the same dictionary, use the regular Slurm
entry point:

```bash
# edit the matching values in scripts/train_crp.conf first
sbatch scripts/train_crp.sbatch
```

### CRPv4 spatial SpLiCE balancing

CRPv4 keeps the CRP projection and simplified student loss, but replaces the earlier
anonymous CoBalT concept balance with image-specific evidence in the exact SpLiCE
vocabulary. Frozen CLIP patch tokens are tested in four controlled variants:
vanilla patchwise, vanilla plus slots, SCLIP patchwise, and SCLIP plus slots. Slot
aggregation happens in CLIP's native visual width; CLIP's frozen visual projection
is applied only afterward. The original SpLiCE cache is never overwritten.

Prepare all four spatial artifacts after the matching CRP cache exists:

```bash
sbatch CoBalT/scripts/prepare_crpv4_spatial.sbatch
```

Then select one artifact in `scripts/train_crp.conf` by setting
`CRP_SPATIAL_BALANCE=true`, `CRP_SPATIAL_BALANCE_VARIANT`, and
`CRP_SPATIAL_BALANCE_PATH`, and run the normal CRP entry point. The resulting
graph is versioned separately as CRPv4; legacy CoBalT balancing remains available
only as a distinct compatibility ablation.

The CRP student objective is ordinary SimCLR plus the confidence-weighted graph
KL. Graph-linked examples are not additional SimCLR positives; that behavior is
Projected kNN is an efficient candidate search; the residual SpLiCE gate is the
only optional semantic relation check.

## Training length and learning-rate schedules

Both SSL and linear-probe milestone schedules accept either explicit epochs or
`auto`. Automatic schedules scale with the requested training length:

- SSL: 70%, 80%, and 90% of `--epochs`;
- linear probe: 60%, 75%, and 90% of `--linear_probe_epochs`.

The current protocol fixes SSL training at 500 epochs, which resolves the
automatic milestones to `350,400,450`.

```bash
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --epochs 500 \
  --lr_decay_epochs auto \
  --linear_lr_decay_epochs auto \
  --splice_mode none
```

Runs with a different epoch budget are not directly comparable with the current
500-epoch protocol.

## Experiment families

### Intervention-utility discovery (group-label free)

The `intervention_utility` selector fits L1 probes by cross-validation on sparse
SpLiCE codes and evaluates exact held-out concept deletion. It rewards increases
in the true-class probability on probe errors, penalizes decreases on correct
examples, and greedily selects a jointly useful non-redundant concept set. It
uses target labels but never spurious/group metadata.

The simplified Slurm launcher intentionally exposes only current CRPv2 and the
older augmentation/correlation methods. Historical synthesis experiments remain
available through the direct Python interface documented below.

The proposed downstream configuration is `UtilityNeutralize`: automatically
selected coordinates are replaced by their target-class median in the frozen
SpLiCE code, the residual-preserving CLIP target is synthesized, and the target
is distilled through the dedicated CLIP head. This edit is agnostic to whether
the nuisance is a background, demographic attribute, colour, texture, or other
language-aligned concept.

### Hyperparameter sweep

Tasks `0..8` are baseline, augmentation quantiles
`.50/.75/.90/.95`, and correlation weights `.001/.01/.1/1.0`.

Cluster sweeps use the method-specific entry point. Edit the corresponding
`.conf` file beside the launcher; W&B names include the relevant values.

```bash
sbatch scripts/train_crp.sbatch
```

### SpLiCE synthesis and distillation

The proposed method edits only selected SpLiCE sparse-code coordinates and
keeps the frozen CLIP residual:

`z_cf = normalize(z + alpha * (c'_S - c_S) @ D_S)`.

The synthesized embedding is cached and used as a stop-gradient cosine target
for a separate `g_clip` MLP on the SimCLR encoder. The ordinary SimCLR
projection head remains responsible for NT-Xent, so both objectives share the
ResNet encoder without forcing the two target spaces through one head.

```powershell
python spur_splice.py `
  --dataset waterbirds `
  --data_folder "D:\Datasets\waterbirds" `
  --epochs 1 `
  --linear_probe_epochs 1 `
  --rank_eval_freq 0 `
  --splice_mode synthesis_distill `
  --splice_concepts "bamboo,forest,hiking,rainforest,raven" `
  --splice_intervention class_neutralize `
  --splice_intervention_strength 1.0 `
  --splice_weight 0.1
```

Available interventions are `original`, `class_neutralize` (primary),
`random_coords`, `shuffled_donor`, `same_class_random_donor`, `zero_out`, and
`core_matched_swap` (metadata-oracle). The oracle uses a donor with the same
class and a different spurious attribute, selected by nearest residual/core
CLIP representation. Synthesized targets and CLIP embeddings are cached under
`--splice_score_cache_dir`; the first run therefore performs a one-time
precomputation before SSL training starts.

## Reproducibility and evaluation

- Matched seeds use the same SSL initialization.
- Frozen SpLiCE/OpenCLIP loading, W&B initialization, and periodic probes are
  RNG-isolated.
- Persistent checkpoints contain Python, NumPy, CPU/CUDA Torch, DataLoader, and
  AMP scaler state.
- Development runs evaluate on validation.
- Test evaluation requires the explicit `--final_test` flag.
- W&B display names are concise; complete parameters and runtime versions
  remain in W&B config and detailed checkpoint directory names.

See `experiments/spurious_eval/README.md` for implementation-level details and
`scripts/README.md` for cluster launch and recovery instructions.

## SpLiCE-CRP v2 frozen audit

The first canonical CRP v2 implementation is deliberately separated from SSL
training. It consumes one `.pt` cache with dataset-ordered `sample_ids`, normalized
`clip_embeddings`, `image_mean`, dense non-negative `splice_codes`, `dictionary`,
`vocabulary`, and `cache_version: 1`. An optional
`provenance` mapping may record checkpoint and dataset identifiers; users do not
need to calculate hashes manually. Annotation keys such as `labels`,
`targets`, `metadata`, or `groups` are rejected at the cache boundary.

```bash
python -m splice.crp \
  --cache outputs/crp/waterbirds_train_features.pt \
  --output outputs/crp/waterbirds_teacher_graph.json
```

The command clusters active concepts, projects the full centered CLIP embedding,
uses projected kNN only as a candidate search, calibrates group selection against
matched random-subspace and shuffled-code nulls, caps donor indegree, and writes one
complete, readable JSON teacher graph. Selecting no group
is valid and produces an empty graph, in which case training automatically reduces
to SimCLR.

The CRP graph construction path uses the frozen SpLiCE cache and the residual
semantic gate. Set `COBALT=true`
after running `CoBalT/scripts/prepare_concepts.sbatch` to reweight concept
frequency/coactivation by fixed label-free CoBalT memberships. This second mode
implements only the label-free concept-balancing marginal during group discovery;
it does not import CoBalT's supervised classifier-balancing stage. Variant caches,
graphs, run names, and W&B tags record these choices separately.

## SpLiCE-CRP v2 student training

The student path uses `--splice_mode crp_relational`. Every sample remains an SSL
anchor; for supported graph rows, the batch sampler adds a neighbour drawn from the
row-stochastic teacher distribution. The backbone relation distribution is matched
with confidence-weighted KL while the ordinary SimCLR projection-head loss remains
active. Target labels, spurious attributes, and groups are not passed to this loss.

The CRP entry point builds or reuses its teacher graph automatically:

```bash
sbatch scripts/train_crp.sbatch
```

CRP training and graph hyperparameters are in `scripts/train_crp.conf`. Graph
sweep variants are edited in `scripts/prepare_crp_group_sweep.conf`.

For CRP graph sweeps, edit the variant records in
`scripts/prepare_crp_group_sweep.conf` and submit the pipeline again. The fixed
feature file is reused without submitting another cache job. ResNet and CRP-loss
settings live in `scripts/train_crp.conf`.

The audit still constructs the graph. If it contains no accepted edges, training
runs but the CRP loss is exactly zero, so the result is the SimCLR fallback. The
ungated run must not be reported as having passed the frozen audit.

Both training scripts enable Weights & Biases by default. CRP v2 runs use
project `Spur_SpLiCE` and group by dataset/seed/protocol. A persistent W&B login
must already exist on the cluster account; API keys are deliberately not stored
in the repository.

Two dedicated sanity-check entry points are also available. The first produces a
self-contained post-hoc HTML with four deterministic median-case Waterbirds pairs,
aggregate graph metrics, exact cosine changes, and actual retained teacher edges
for every shown CRP group. Each CRP graph path contains its complete
frozen-audit configuration and the HTML is stored beside that graph, so different
configurations cannot overwrite one another. The second disables
the SimCLR term and trains with relational KL alone while retaining the normal
linear evaluation:

```bash
sbatch scripts/concept_ablation_examples.sbatch
sbatch scripts/train_kl_only.sbatch
```
