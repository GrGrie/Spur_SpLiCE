# Automatic layer-localized SpLiCE scrubbing

This branch tests whether a shortcut should be removed at the first stable
stage where it becomes linearly accessible, instead of regularizing only the
final representation.

## Algorithm

1. Frozen SpLiCE discovers and caches the selected semantic concept weights.
2. Every `--localized_probe_freq` epochs, deterministic, non-augmented training
   images are passed through `stem`, `layer1` through `layer4`, encoder output,
   and projection-head output.
3. Each concept score is residualized within the target class using
   training-fold class means. Ridge probes are fitted on one fold and evaluated
   out of sample on a held-out fold.
4. Leakage is `R2_observed - mean(R2_permuted)`. An onset is locked at the
   earliest stage whose leakage exceeds the threshold for
   `--localized_stability` consecutive probe checkpoints.
5. Concept-probe rows assigned to each onset are orthogonalized against a
   target-label probe. The resulting row space defines a low-rank soft
   projector at exactly that stage:

   ```text
   h <- h - alpha h W^T (W W^T + epsilon I)^-1 W
   ```

   `alpha` grows linearly from zero at `--localized_leakage_threshold` to one
   at `--localized_leakage_max`.

Probes run with existing erasers temporarily suspended. This prevents an
eraser from making its own measured leakage disappear and oscillating on/off.
Once an onset is stable it remains locked, while its direction and strength
are refitted. Probe history, locked onsets, and projector factors are part of
the model state and survive checkpoint resume.

Each update is appended to
`<save_folder>/localized_leakage.jsonl`, including per-concept/per-layer
leakage, permutation baselines, onsets, ranks, and alphas.

## Matched 500-epoch comparison

Windows:

```powershell
.\scripts\Run-LayerLocalized500.ps1 `
  -Variant both `
  -Seeds 0,1,2 `
  -DataFolder "D:\Datasets\waterbirds"
```

Slurm:

```bash
sbatch --array=0-1 --export=ALL,DATA_FOLDER=/path/to/datasets,SEED=0 \
  scripts/waterbirds_layer_localized_500.sbatch
```

Cross-platform Python (also useful with `--dry_run`):

```bash
python scripts/run_layer_localized_500.py \
  --variant both \
  --data_folder ./datasets \
  --seed 0
```

Task/variant `baseline` uses `--splice_mode none`; `localized` uses automatic
top-10 concept discovery. Both use 500 SSL epochs, automatically scaled
`350,400,450` LR milestones, the same seed and optimizer settings, and
validation-only periodic linear probes. Test evaluation remains guarded by the
existing `--final_test` protocol.

## Required ablations for a paper

- final-layer correlation regularization (`corr_reg`);
- every-layer scrubbing and a fixed chosen layer;
- onset without stability or permutation correction;
- hard removal (`alpha=1`) versus leakage-scaled soft removal;
- with and without target-subspace protection;
- oracle background concepts versus automatic SpLiCE concepts;
- concept/group labels versus the label-free discovery alternative;
- at least Waterbirds, CelebA, SpurCIFAR10, multiple seeds, and architectures.

The method identifies conditional association, not causality. Without group
metadata, environments, interventions, or another explicit assumption,
automatic concept discovery cannot guarantee that a correlated concept is
spurious.
