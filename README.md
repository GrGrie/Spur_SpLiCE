# Spur SpLiCE

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
- `scripts/tools/summarize_splice_scores.py` — selected-concept score summaries.
- `scripts/tools/render_report_figure.py` — report figure generation.
- `scripts/Run-HomeExperiments.ps1` — selected Windows experiment runs.
- `scripts/Start-ReportRuns.ps1` — priority queue for the current report.
- `scripts/*.sbatch` — Slurm arrays for cluster runs.

## Training length and learning-rate schedules

Both SSL and linear-probe milestone schedules accept either explicit epochs or
`auto`. Automatic schedules scale with the requested training length:

- SSL: 70%, 80%, and 90% of `--epochs`;
- linear probe: 60%, 75%, and 90% of `--linear_probe_epochs`.

Thus 1,000 SSL epochs resolve to `700,800,900`, while 500 resolve to
`350,400,450`.

```bash
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --epochs 1000 \
  --lr_decay_epochs auto \
  --linear_lr_decay_epochs auto \
  --splice_mode none
```

Shortened 500-epoch runs are operationally supported, but they are not directly
comparable with the existing epoch-1000 report tables.

## Experiment families

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

Cluster example:

```bash
sbatch --array=0-8 --export=ALL,DATASET=waterbirds,SEED=4 \
  scripts/waterbirds_SpLiCE_hyperparameter_array.sbatch
```

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
