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
- `python -m splice.crp_group_screen` — fast group-prototype reconstruction and sampled intervention audit.
- `scripts/run_crpv4_group_screen.ps1` / `scripts/crpv4_group_screen.sbatch` — Windows and Slurm group-screen launchers.
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
explicit vocabulary and size only for a deliberate ablation. Existing caches and
graphs built with another vocabulary must not be mixed with the Open Images V7
dictionary.

The current CRPv4 workflow screens concept groups before building a full teacher
graph. Edit the three experiment records and shared thresholds in
`scripts/crpv4_group_screen.conf`, then run:

```bash
sbatch scripts/cache_openimages_crp.sbatch
sbatch scripts/crpv4_group_screen.sbatch
```

The screen first collapses each flat CRP group into one text prototype and asks
whether the prototype mixture preserves the original SpLiCE reconstruction. After
that gate passes, it runs a small deterministic intervention audit over a
rank-spaced sample of the reconstruction-selected groups, including the most- and
least-active selected groups. Every experiment writes `group_screen.json` and a self-contained
`group_screen.html` with the decision, coverage curve, configuration, class slices,
poor/median/good reconstruction examples, group contact sheets, the audited/selected
group count, and raw-versus-projected nearest-neighbour triplets. Dataset annotations
appear only in the post-hoc HTML section and
never enter grouping or the mini audit.

On Windows the same default three-arm story is launched with:

```powershell
.\scripts\run_crpv4_group_screen.ps1
```

The default records compare the current coactivation-gated grouping against
semantic-only thresholds 0.70 and 0.65. Spatial variants can be added after their
artifacts have been prepared; the configuration contains commented examples.

This is a mechanism screen, not a final comparison. A failed report should be
discarded before graph construction. A visually coherent `PROVISIONAL_GO`
candidate proceeds to the full frozen audit and only then to matched W&B-tracked
student training.

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

CRP training and full-graph hyperparameters remain in `scripts/train_crp.conf`.
Group-screen variants and their lightweight audit settings live in
`scripts/crpv4_group_screen.conf`. The frozen feature file is reused across every
screen record.

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
