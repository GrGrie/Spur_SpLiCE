# Home training on Windows

These scripts reproduce the Slurm experiment families sequentially on one
Windows GPU. They default to the report-compatible 1,000-epoch protocol, do
not throttle the GPU or change process priority, and create resumable
checkpoints every 25 epochs.

## Before connecting through AnyDesk

1. Update the NVIDIA driver.
2. Make the Waterbirds dataset available locally. The `datasets/` directory is
   gitignored and will **not** arrive through GitHub. The directory passed to
   `-DataFolder` must contain `metadata.csv` and the image paths referenced by
   it.
3. From the repository root, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
conda activate grgrie-train
.\scripts\Test-HomeTraining.ps1 -DataFolder "D:\Datasets\waterbirds"
wandb login
```

The scripts automatically find an active or installed Conda environment named
`grgrie-train`. Use `Setup-HomeTraining.ps1` only if that environment is not
available. RTX 5080 (`sm_120`) requires a Blackwell-capable CUDA wheel.

## Recommended report queue

```powershell
.\scripts\Start-ReportRuns.ps1 -DataFolder "D:\Datasets\waterbirds"
```

The first two jobs are the highest-priority missing semantic/shuffled seed-4
pair. They are followed by a same-RTX-5080 seed-4 baseline, the matched-random
controls, and only then component ablations. Runs are sequential, so only one
training process occupies the GPU.

To launch only selected experiments:

```powershell
# Routing: 0 baseline, 1 semantic, 2 shuffled, 3 random, 4 augment-all
.\scripts\Run-HomeExperiments.ps1 `
  -Family routing -Seeds 4 -Tasks 1,2 `
  -DataFolder "D:\Datasets\waterbirds"

# Components: 0 all, 1 crop, 2 color jitter, 3 grayscale, 4 blur
.\scripts\Run-HomeExperiments.ps1 `
  -Family augmentation -Seeds 3,4 -Tasks 1 `
  -DataFolder "D:\Datasets\waterbirds"

# Broad sweep: 0 baseline, 1-4 augmentation q=.50/.75/.90/.95,
# 5-8 correlation weights .001/.01/.1/1.0
.\scripts\Run-HomeExperiments.ps1 `
  -Family sweep -Seeds 4 -Tasks 7 `
  -DataFolder "D:\Datasets\waterbirds"
```

Use `-Epochs 500` only for triage. The runner scales the learning-rate
milestones to `350,400,450` and tags the W&B run `protocol_short_e500`, because
it is not directly comparable with the report's 1,000-epoch results.

## Layer-localized SpLiCE comparison (500 epochs)

The dedicated runner launches a matched baseline and localized run:

```powershell
.\scripts\Run-LayerLocalized500.ps1 `
  -Variant both -Seeds 0,1,2 `
  -DataFolder "D:\Datasets\waterbirds"
```

Use `-DryRun` to print both commands without starting training. On Slurm,
array task 0 is the baseline and task 1 is the localized method:

```bash
sbatch --array=0-1 --export=ALL,DATA_FOLDER=/path/to/datasets,SEED=0 \
  scripts/waterbirds_layer_localized_500.sbatch
```

## Logging and recovery

- GPU utilization and power are not limited. The scripts do not stop training
  based on temperature and do not lower Python process priority.
- Python stdout/stderr are written continuously to `outputs/home_logs/`.
- Checkpoints are in `outputs/home_checkpoints/`. Re-running the same command
  automatically resumes the newest `epoch_*.pth`; a completed `last.pth` is
  skipped unless `-Force` is supplied.
- `Ctrl+C` stops the current Python process tree. The latest periodic
  checkpoint remains available.
