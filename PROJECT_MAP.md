# Spur SpLiCE — project map

This is the small navigation layer for the current repository.

Its purpose is to tell an agent what the project is, where the main code lives,
and which larger document to read only when needed.

## Current direction

The active method is SpLiCE-CRP v4. It uses a frozen OpenCLIP/SpLiCE model to
produce language-aligned sparse concept activations, then constructs a label-free
teacher graph by projecting out selected concept-group subspaces. The trainable
student is a SimCLR ResNet whose representation is regularized toward the teacher
relation geometry.

The current CRPv4 extension is a CoBalT-style image-specific spatial amplifier in
the exact SpLiCE vocabulary. It produces a separate balance signal that directly
changes the SpLiCE codes used by grouping and relation construction while leaving
the frozen cache untouched. The original VQ-based CoBalT path remains only as a
separate legacy compatibility control.

The first active control study isolates ordinary SpLiCE-CRP against SimCLR,
CRP-sampler-only, and a matched raw-CLIP teacher. Spatial/slots are an optional
follow-up, not a prerequisite; legacy CoBalT is excluded from this study.

## Core pipeline

```text
image
  ├─ frozen OpenCLIP → centered CLIP embedding
  └─ frozen SpLiCE  → sparse concept codes
                         │
                         ├─ concept grouping and subspace projection
                         ├─ projected kNN candidate search
                         ├─ residual SpLiCE agreement gate (optional)
                         └─ null-calibrated sparse teacher graph
                                      │
                                      └─ graph-aware batching + weighted KL

student image views → trainable SimCLR ResNet → SimCLR loss + graph KL
```

The current student objective is:

```text
L = L_SimCLR + lambda_graph * L_graph_KL
```

Graph-linked examples are not extra SimCLR positives. The relational term uses
the fixed row-stochastic teacher graph, scheduled warm-up/decay, and anchor
confidence. Empty graphs safely reduce training to ordinary SimCLR.

## Main code map

| Area | Files |
|---|---|
| Training entry | `spur_splice.py`, `scripts/train_crp.sbatch` |
| CRP audit | `splice/crp.py`, `splice/graph_io.py` |
| CRP training | `splice/crp_training.py`, `experiments/spurious_eval/training/ssl_loop.py` |
| Group screen | `splice/crp_group_screen.py`, `scripts/tools/render_crp_group_screen.py` |
| Historical optional diverse audit | `splice/crp_diverse.py` |
| Cache | `scripts/tools/cache_crp_features.py` |
| Spatial CoBalT path | `CoBalT/train_spatial.py`, `CoBalT/extract_spatial_balance.py`, `splice/spatial_balance.py` |
| Baseline/report tools | `scripts/tools/build_crp_baseline_graphs.py`, `scripts/tools/render_concept_ablation_examples.py` |
| Config paths | `scripts/tools/crp_config_path.py`, `scripts/train_crp.conf`, `scripts/shared_training.conf` |
| Tests | `tests/test_splice_pipeline.py`, `tests/test_crp_group_screen.py`, `tests/test_crp_diverse.py`, `tests/test_crp_config_path.py` |
| Converged probe | `experiments/spurious_eval/training/logistic_probe.py`, `experiments/spurious_eval/linear_probe.py` |
| Control study / interpretation | `scripts/run_crp_controls.ps1`, `scripts/run_crp_controls.conf`, `docs/RESEARCH_PROTOCOL.md` |

## Canonical commands

```bash
sbatch scripts/cache_openimages_crp.sbatch
sbatch scripts/crpv4_group_screen.sbatch
sbatch scripts/SpLiCE_CRP_v2_frozen_audit.sbatch
sbatch CoBalT/scripts/prepare_crpv4_spatial.sbatch
sbatch scripts/train_crp.sbatch
sbatch scripts/concept_ablation_examples.sbatch
sbatch scripts/train_kl_only.sbatch
```

Edit the referenced `.conf` file before launch. Full SSL training must keep W&B
enabled; disable it only for an explicit smoke test or local debugging.

## What to read next

- For current method/architecture details:
  `docs/CURRENT_METHOD.md`
- For experiments, probes, evaluation, reproducibility, and supported claims:
  `docs/EXPERIMENT_PROTOCOL.md`
- For an older method, result, launcher, reporting snapshot, or design decision:
  search `docs/Project History.md` for the relevant term/date.

Do not read all three large documents by default.
