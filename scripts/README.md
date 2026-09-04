# Experiment launchers

The maintained CRPv4 workflow separates cheap concept-group screening from full
teacher-graph construction and SSL training.

## 1. Frozen feature cache

Edit `cache_openimages_crp.conf`, then run:

```bash
sbatch scripts/cache_openimages_crp.sbatch
```

The cache is reused by all group-screen configurations. It contains no target,
context, group, or metadata annotations.

## 2. CRPv4 spatial evidence

Edit `prepare_crpv4_spatial.conf`, then run:

```bash
sbatch CoBalT/scripts/prepare_crpv4_spatial.sbatch
```

This prepares `vanilla_patchwise`, `vanilla_slots`, `sclip_patchwise`, and
`sclip_slots` artifacts. Patchwise variants require no Slot Attention training;
slot variants train only the label-free spatial grouping module. All variants
produce evidence in the exact SpLiCE vocabulary.

## 3. Group-only screen and mini audit

The screen is the first CRPv4 decision point. It does not build a teacher graph
or train a student.

Each concept group is collapsed to one normalized text prototype. Image-specific
group weights are sums of the corresponding SpLiCE weights. The report measures
how closely that compact mixture reproduces the original full SpLiCE
representation. Groups are ordered by total activation mass, and the smallest
evaluated prefix reaching the requested image coverage is passed to a sampled
intervention audit.

The default experiment story is:

1. `current_t070_c020`: current text plus coactivation grouping;
2. `semantic_t070`: remove coactivation so mutually exclusive names can merge;
3. `semantic_t065`: lower text similarity to create broader semantic families.

Edit `crpv4_group_screen.conf`, then run:

```bash
sbatch scripts/crpv4_group_screen.sbatch
```

Every record produces:

```text
outputs/crpv4_group_screen/<dataset>/<variant>/group_screen.json
outputs/crpv4_group_screen/<dataset>/<variant>/group_screen.html
```

The HTML is self-contained and includes:

- a clear `FAIL_RECONSTRUCTION`, `FAIL_INTERVENTION`, `REVIEW_GROUPS`, or
  `PROVISIONAL_GO` decision;
- the reconstruction coverage curve and resolved hyperparameters;
- poor, median, and good reconstruction examples;
- post-hoc target/context slices, clearly isolated from selection;
- a searchable table of every discovered group;
- image contact sheets for reconstruction-selected groups;
- mini-audit null outcomes, the audited/selected group count, and representative
  anchor/raw-neighbour/projected-neighbour triplets across a rank-spaced group sample.

### Windows

The Windows launcher contains the same three default experiment records and
automatically looks for the existing local Waterbirds cache and dataset root:

```powershell
.\scripts\run_crpv4_group_screen.ps1
```

Inspect paths and records without running the experiment:

```powershell
.\scripts\run_crpv4_group_screen.ps1 -ValidateOnly
```

Run one explicit spatial screen after the spatial artifact exists:

```powershell
.\scripts\run_crpv4_group_screen.ps1 -VariantRecords @("semantic_t065_slots|0.01|0.95|0.65|0.00|vanilla_slots|0.25|0.0")
```

Use `-SkipMiniAudit` when iterating only on reconstruction and visual grouping.

## 4. Full frozen graph audit

Only configurations that receive a satisfactory group report should proceed to
the full null-calibrated teacher graph:

```bash
sbatch scripts/SpLiCE_CRP_v2_frozen_audit.sbatch
```

Despite its retained compatibility filename, this launcher uses the current CRP
graph implementation. Full graph settings live in `train_crp.conf` or in the
standalone audit launcher, depending on the chosen workflow.

Post-hoc graph diagnostics remain available through:

```bash
sbatch scripts/SpLiCE_CRP_v2_report.sbatch
sbatch scripts/SpLiCE_CRP_v2_posthoc_waterbirds.sbatch
sbatch scripts/concept_ablation_examples.sbatch
```

## 5. SSL training

Full CRP training uses:

```bash
sbatch scripts/train_crp.sbatch
```

The separate KL-only diagnostic uses:

```bash
sbatch scripts/train_kl_only.sbatch
```

Full SSL training must keep W&B enabled. Disabling it is allowed only for an
explicit smoke test or local debugging.

## Retained controls and utilities

- `prepare_concepts.conf` and `CoBalT/scripts/prepare_concepts.sbatch` retain the
  legacy CoBalT compatibility control.
- `SpLiCE_CRP_v2_baseline_*.sbatch` retain raw-CLIP graph baselines.
- `scripts/tools/` contains active cache, reporting, evaluation, and baseline
  modules; these are not retired launchers.
- `load_config.sh` is the shared trusted `.conf` loader for Slurm entry points.
