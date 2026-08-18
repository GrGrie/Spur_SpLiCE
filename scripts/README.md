# Home training on Windows

## SpLiCE-CRP v2 frozen audit on Slurm

The first job caches dataset-ordered OpenCLIP, SpLiCE, and DINOv3 features. The
second exports the teacher graph plus its JSON audit. Only caching requests a GPU;
neither job reads labels into its artifact or starts SSL training.

```bash
# Cache first, then start one audit only after caching succeeds.
CACHE_JOB=$(sbatch --parsable --export=ALL,DATASET=waterbirds,DATA_FOLDER=./datasets \
  scripts/SpLiCE_CRP_v2_cache_features.sbatch)
sbatch --dependency=afterok:${CACHE_JOB} \
  --export=ALL,CACHE_PATH=outputs/crp/waterbirds_train_features.pt \
  scripts/SpLiCE_CRP_v2_frozen_audit.sbatch

# Or repeat null calibration with three seeds after an existing cache is ready.
sbatch --array=0-2 --export=ALL,CACHE_PATH=outputs/crp/waterbirds_train_features.pt \
  scripts/SpLiCE_CRP_v2_frozen_audit.sbatch
```

Override `PROJECT_DIR`, `CONDA_ENV`, `DATASET`, `OUT_DIR`, or `SEED_BASE` in the
same way as the existing jobs. Advanced audit settings can be passed as JSON, for
example `CRP_CONFIG_JSON='{"null_trials":8,"similarity_chunk_size":256}'`.
The cache job reuses an existing output unless `FORCE=true`; the audit job fails
before allocating audit work when `CACHE_PATH` does not exist.

## Universal cluster arrays

Two Slurm arrays implement the intervention-utility method. Each submission
targets one dataset through the `DATASET` variable (`waterbirds`, `celeba`, or
`spur_cifar10`). `SpLiCE_intervention_utility_discovery_array.sbatch` has four
tasks: intervention utility, error contrast, the published gradient probe, and
the metadata oracle under the same frozen SpLiCE-CBM intervention.

`SpLiCE_intervention_utility_training_array.sbatch` has five tasks for the
selected dataset: SimCLR, unedited CLIP distillation, proposed utility-selected
class neutralization, zero-out, and matched random-coordinate neutralization.
Repeat the array with different `SEED` values; do not pool unmatched epoch
budgets.

```bash
sbatch --export=ALL,DATASET=waterbirds scripts/SpLiCE_intervention_utility_discovery_array.sbatch
sbatch --export=ALL,DATASET=waterbirds,SEED=0 scripts/SpLiCE_intervention_utility_training_array.sbatch
```

These scripts reproduce the Slurm experiment families sequentially on one
Windows GPU. They default to the 500-epoch protocol used by the
synthesis--distillation stage, do not throttle the GPU or change process
priority, and create resumable checkpoints every 25 epochs. Pass
`-Epochs 1000` to reproduce the legacy tables in the report.

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

# Synthesis + distillation control battery:
# 0 baseline, 1 original CLIP, 2 class neutralize, 3 random coordinates,
# 4 shuffled donor, 5 same-class random donor, 6 zero-out,
# 7 oracle core-matched cross-background swap
.\scripts\Run-HomeExperiments.ps1 `
  -Family synthesis -Seeds 0 -Tasks 0,1,2,3,4,5,6,7 `
  -Epochs 1 -DistillationWeight 0.1 -InterventionStrength 1.0 `
  -DataFolder "D:\Datasets\waterbirds" -NoWandb
```

## Synthesis stage: run the pilot before the full battery

```powershell
# Pilot: baseline vs unedited CLIP teacher vs edited target, one seed.
.\scripts\Run-HomeExperiments.ps1 `
  -Family synthesis -Seeds 4 -Tasks 0,1,2 `
  -DataFolder "D:\Datasets\waterbirds"
```

Expand to `-Tasks 3,4,5,6,7` and more seeds only if task 2 separates from
**both** task 0 and task 1. Task 1 is the decisive control: if plain CLIP
distillation already produces the gain, the sparse edit is redundant, and if it
*degrades* worst-group accuracy, the edited target must be judged against that
degraded reference instead of the SimCLR baseline.

Sweep the two distillation hyperparameters with `-DistillationWeight`
(&lambda;, try `0.05, 0.1, 0.25, 1.0`) and `-InterventionStrength`
(&alpha;, try `0.5, 1.0, 1.5`). Both values appear in the W&B run name, so
sweep points never collide.

## Run naming

W&B runs are named `{Dataset}_S{seed}_{Task}_w{lambda}a{alpha}_e{epochs}`, for
example `Waterbirds_S4_SynNeutralize_w0.1a1_e500`. Baselines omit the
hyperparameter block (`Waterbirds_S4_Baseline_e500`) and the unedited-teacher
control omits &alpha;, which is inert when no edit is applied
(`Waterbirds_S4_OrigCLIP_w0.1_e500`). The full reproducibility name stays in the
checkpoint directory and `args.json`. Runs are grouped as
`{dataset}_{family}_e{epochs}_seed{seed}` and tagged with `protocol_e{epochs}`,
plus `lambda_*` / `alpha_*` for synthesis runs.

Epoch counts are part of every name and group because a 500-epoch run is **not**
comparable with a 1,000-epoch run. Compare only within one budget, against a
baseline trained under that same budget.

## Logging and recovery

- GPU utilization and power are not limited. The scripts do not stop training
  based on temperature and do not lower Python process priority.
- Python stdout/stderr are written continuously to `outputs/home_logs/`.
- Checkpoints are in `outputs/home_checkpoints/`. Re-running the same command
  automatically resumes the newest `epoch_*.pth`; a completed `last.pth` is
  skipped unless `-Force` is supplied.
- `Ctrl+C` stops the current Python process tree. The latest periodic
  checkpoint remains available.
