# Spur SpLiCE — current project ground truth

## 1. Current goal

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

## 2. Information boundary

The discovery cache contains only dataset-ordered sample IDs, frozen CLIP
embeddings, the frozen image mean, non-negative SpLiCE codes, the normalized
SpLiCE dictionary, vocabulary, cache version, and optional provenance. Target,
spurious, group, and metadata fields are rejected at validation.

Hidden labels may be used only by post-hoc diagnostics and final evaluation.
They never select concept groups, teacher edges, or training weights.

## 3. Current architecture

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

## 4. CRP relation construction

For each selected concept group, CRP removes the full orthonormal subspace from
centered CLIP embeddings and searches projected neighbours. Relations are kept
when they show positive intervention gain, sufficient activation difference, and
pass the optional residual SpLiCE agreement threshold. Candidate edges are
scored with confidence, calibrated against random-subspace and shuffled-code
nulls, capped by absolute indegree, and written as a sparse row-stochastic graph.

The active artifacts are:

- `splice_crp_v3_teacher_graph` for ordinary CRP;
- `splice_crp_v4_teacher_graph` when spatial balancing is enabled;
- `splice_crp_v2_teacher_graph` as a legacy compatibility format.

## 5. CRPv4 spatial balancing

The spatial path consumes frozen CLIP patch-token evidence and produces aligned
image-specific concept support in four variants: vanilla patchwise, vanilla plus
slots, SCLIP patchwise, and SCLIP plus slots. It preserves the original SpLiCE
cache and passes the resulting artifact into the CRP audit.

The spatial artifact is prepared by
`CoBalT/scripts/prepare_crpv4_spatial.sbatch` and selected through
`CRP_SPATIAL_BALANCE`, `CRP_SPATIAL_BALANCE_VARIANT`, and
`CRP_SPATIAL_BALANCE_PATH` in `scripts/train_crp.conf`.

Before a full graph audit, the canonical preparation workflow runs a lightweight
group screen. Every flat group is collapsed to one normalized text prototype and
the activation-ranked prototype mixture is compared with the original full
SpLiCE reconstruction. A deterministic image subset then receives the unchanged
CRP projection/null checks on a rank-spaced sample spanning the selected group
list, without materializing a teacher graph. The resulting self-contained HTML
separates label-free selection from post-hoc class slices.

## 6. CoBalT boundary

The active CRPv4 CoBalT integration does not use a separate concept vocabulary or
vector-quantization codebook. Legacy CoBalT discovery/classifier files remain for
the historical compatibility control; the spatial CRPv4 path uses the frozen
SpLiCE vocabulary and its own image-specific evidence.

## 7. Status and claims

Implemented: frozen CLIP/SpLiCE cache construction, the reconstruction/mini-
intervention group screen, CRP v2/v3/v4 graph audits, CRPv4 spatial artifacts,
graph-aware relational training, legacy CoBalT control, W&B-enabled training
entry points, and label-isolated post-hoc reports.

Not established: improvement in worst-group accuracy, superiority of any graph
variant, or transfer of a graph relation to a particular nuisance attribute.
All such statements require matched real-data runs, controls, and final-test
evaluation under the repository protocol.

## 8. Canonical files

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

## 9. Canonical commands

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

## 10. Reproducibility rules

- Keep frozen cache, graph, spatial artifact, configuration, seed, and split aligned.
- Treat graph preparation and HTML/report jobs as diagnostics, not training.
- Keep validation splits separate from final-test evaluation.
- Record material method, loss, protocol, and interpretation changes in `Project History.md`.
- Preserve historical reports and chronology instead of silently rewriting them.
