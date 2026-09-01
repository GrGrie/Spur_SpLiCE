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
an up-to-date NVIDIA driver. A Windows setup and smoke test are also available:

```powershell
.\scripts\Setup-HomeTraining.ps1
.\scripts\Test-HomeTraining.ps1 -DataFolder "D:\Datasets\waterbirds"
```

Dataset directories are intentionally excluded from Git. Pass their location
with `--data_folder` or `-DataFolder`.

## Main entry points

- `spur_splice.py` — SimCLR training and periodic/final linear probing.
- `linear_probe.py` — standalone target/spurious linear probing.
- `splice_cbm.py` — sparse concept-bottleneck baseline.
- `scripts/tools/discover_splice_spurious_concepts.py` — automatic concept discovery.
- `python -m scripts.tools.cache_crp_features` — aligned OpenCLIP/SpLiCE/DINOv3 cache construction.
- `python -m splice.crp` — label-free CRP v2 frozen audit and teacher-graph export.
- `splice/crp_training.py` — graph-aware batching and confidence-weighted relational distillation.
- `scripts/tools/summarize_splice_scores.py` — selected-concept score summaries.
- `scripts/tools/render_report_figure.py` — report figure generation.
- `scripts/Run-HomeExperiments.ps1` — selected Windows experiment runs.
- `scripts/Start-ReportRuns.ps1` — priority queue for the current report.
- `scripts/train_crp.sbatch` — CRP training entry point for Slurm.
- `scripts/train_cqt.sbatch` — CQT training entry point for Slurm.

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

### Routing controls

Tasks are baseline, semantic, shuffled, matched-random, and augment-all:

```powershell
.\scripts\Run-HomeExperiments.ps1 `
  -Family routing -Seeds 4 -Tasks 1,2 `
  -DataFolder "D:\Datasets\waterbirds"
```

### Strong-augmentation components

Tasks `0..4` are All, Crop, ColorJitter, Grayscale, and Blur.

```powershell
.\scripts\Run-HomeExperiments.ps1 `
  -Family augmentation -Seeds 3,4 -Tasks 1,2,3,4 `
  -DataFolder "D:\Datasets\waterbirds"
```

### Hyperparameter sweep

Tasks `0..8` are baseline, augmentation quantiles
`.50/.75/.90/.95`, and correlation weights `.001/.01/.1/1.0`.

Cluster sweeps use the method-specific entry point. Edit one configuration at a
time in its top block; W&B names include the relevant values.

```bash
sbatch scripts/train_crp.sbatch
sbatch scripts/train_cqt.sbatch
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
`scripts/README.md` for Windows launch/recovery instructions.

## SpLiCE-CRP v2 frozen audit

The first canonical CRP v2 implementation is deliberately separated from SSL
training. It consumes one `.pt` cache with dataset-ordered `sample_ids`, normalized
`clip_embeddings`, `image_mean`, dense non-negative `splice_codes`, `dictionary`,
`vocabulary`, normalized `dino_embeddings`, and `cache_version: 1`. An optional
`provenance` mapping may record checkpoint and dataset identifiers; users do not
need to calculate hashes manually. Annotation keys such as `labels`,
`targets`, `metadata`, or `groups` are rejected at the cache boundary.

```bash
python -m splice.crp \
  --cache outputs/crp/waterbirds_train_features.pt \
  --output outputs/crp/waterbirds_teacher_graph.json
```

The command clusters active concepts, projects the full centered CLIP embedding,
keeps reciprocal relations supported by DINO, calibrates group selection against
matched random-subspace and shuffled-code nulls, caps donor indegree, and writes one
complete, readable JSON teacher graph. Selecting no group
is valid and produces an empty graph, in which case training automatically reduces
to SimCLR.

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

CRP training and graph hyperparameters are grouped near the top of
`scripts/train_crp.sbatch`. CQT has an independent top-level configuration in
`scripts/train_cqt.sbatch`; no terminal-side environment variables are required.

For CRP graph sweeps, edit the named numeric variables near the top of
`scripts/SpLiCE_CRP_v2_frozen_audit.sbatch`, for example
`MIN_GROUP_SIZE="2"`, and submit the pipeline again. The fixed feature file is
reused without submitting another cache job. ResNet and CRP-loss settings live
in `scripts/train_crp.sbatch`.

The audit still constructs the graph. If it contains no accepted edges, training
runs but the CRP loss is exactly zero, so the result is the SimCLR fallback. The
ungated run must not be reported as having passed the frozen audit.

Both training scripts enable Weights & Biases by default. CRP v2 runs use
project `Spur_SpLiCE` and group by dataset/seed/protocol. A persistent W&B login
must already exist on the cluster account; API keys are deliberately not stored
in the repository.

Two dedicated sanity-check entry points are also available. The first produces a
self-contained post-hoc HTML with representative Waterbirds image pairs and exact
cosine changes for every selected CRP group (or CQT factor). The second disables
the SimCLR term and trains with relational KL alone while retaining the normal
linear evaluation:

```bash
sbatch scripts/concept_ablation_examples.sbatch
sbatch scripts/train_kl_only.sbatch
```

Set `REPORT_METHOD=cqt` or `KL_METHOD=cqt`, respectively, through Slurm exports to
run the CQT variant. See `scripts/README.md` for the exact commands and output path.
