# Spur SpLiCE

Spur SpLiCE investigates whether sparse semantic concepts can improve visual
representations under spurious object–context correlations. The main method is
**SpLiCE-CRP**: frozen OpenCLIP/SpLiCE features define concept subspaces, a
label-free audit constructs a sparse teacher graph and a SimCLR ResNet learns
with graph-aware batches and confidence-weighted relational KL.

The graph and SSL stages use no target or group annotations. Linear evaluation
uses labels and a group-balanced subset. The teacher is unnecessary at student
inference. Spatial evidence is an optional extension.

Read [PROJECT_MAP.md](PROJECT_MAP.md) for navigation and
[Spur_SpLiCE.tex](Spur_SpLiCE.tex) for the method and empirical evidence.
[paper_results.json](paper_results.json) contains unrounded results, convergence
records and source digests. [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md) records
the cleanup decisions and verification.

## Current evidence

All results are **Waterbirds validation**, SSL epoch 500. The ordinary CRP study
uses one frozen graph seed and four student seeds. At KL weight 2, mean
average/worst-group accuracy is 51.21/45.92% for CRP and 52.11/45.54% for SimCLR.
CRP sampler-only reaches 52.46/47.39%. This comparison does not establish a
consistent benefit from KL at weight 2.

A separate two-seed screen reaches 54.55/50.80% with CRP weight 0.5. This
exploratory validation result requires independent replication. The latest
configured experiment compares direct raw-CLIP transfer, SpLiCE reconstruction
transfer, shuffled reconstruction and matched SimCLR. Completed student results
for that protocol and the spatial variants are absent from the available evidence.

## Environment

```bash
conda activate grgrie-train
pip install -e '.[test]'
python -m pytest tests -q
```

Dataset files and experimental outputs are local artifacts. Set the dataset root
in the corresponding configuration. Authenticate W&B outside the repository
before full training.

## Main workflows

| Task | Entry point |
|---|---|
| Frozen cache | `python -m scripts.tools.cache_crp_features` |
| CRP graph audit | `python -m splice.crp` |
| Student training | `spur_splice.py` / `scripts/train_crp.sbatch` |
| Matched controls | `scripts/tools/run_crp_controls.py` |
| Linear evaluation | `linear_probe.py` |
| Group screening | `python -m splice.crp_group_screen` |
| Optional spatial evidence | `CoBalT/scripts/prepare_crpv4_spatial.sbatch` |
| Latest direct-transfer follow-up | `scripts/run_next_tests_2026-09-07.sbatch` |
| Probe, signal and graph diagnostics | `scripts/crp_signal_diagnostics.sbatch` |

Supported SSL modes are `none`, `crp_relational` and `frozen_concept_distill`
(the latter currently supports Waterbirds). Graph-linked images affect batches
and relational targets; they do not become extra SimCLR positive pairs.
Empty CRP graphs fall back to SimCLR except in the KL-only diagnostic.

```bash
# Edit the corresponding .conf files before using cluster paths.
sbatch scripts/cache_openimages_crp.sbatch
sbatch scripts/train_crp.sbatch
sbatch scripts/run_next_tests_2026-09-07.sbatch
```

The generic training configuration differs from the locked historical controls.
Use the matched-control configurations to reproduce the manuscript.
[scripts/README.md](scripts/README.md) explains launch order;
[experiments/spurious_eval/README.md](experiments/spurious_eval/README.md)
explains evaluation.

The complete cleaned Open Images V7 vocabulary is the default. Cache builders
download its class-label dictionary only. Cache rows, vocabulary and graph
sample IDs must match exactly.

## Paper and retained evidence

The manuscript is standalone LaTeX with embedded references and a vector diagram:

```bash
pdflatex Spur_SpLiCE.tex
pdflatex Spur_SpLiCE.tex
```

Alternatively run `tectonic Spur_SpLiCE.tex`. The locally verified PDF is
`output/pdf/Spur_SpLiCE.pdf`. The manuscript reports validation findings; it
makes no final-test or state-of-the-art claim.

Retain `tmp/crp_signal_checks_v1`: despite its historical location, it contains
the manuscript's source experiments. Preserve its feature tensors and logs for
audit. Write new runs to their configured `outputs/` directories.
