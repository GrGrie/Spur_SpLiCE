# Spur SpLiCE — project map

The research question is whether frozen sparse semantic concepts can guide
self-supervised learning toward spurious-correlation robustness. SpLiCE-CRP is
the main architecture. Spatial balancing and direct reconstruction transfer
are implemented follow-ups.

## Data flow

```text
training images -> frozen OpenCLIP + SpLiCE -> aligned cache
  -> concept grouping -> full-subspace projection -> candidate neighbours
  -> residual concept gate + null audit -> fixed sparse teacher graph

training views -> ResNet + SimCLR head -> SimCLR loss
teacher graph -> graph-aware batches + confidence-weighted backbone KL
frozen trained ResNet -> supervised linear probe -> validation Avg./WGA
```

Target and group annotations are confined to evaluation and post-hoc diagnosis.
The balanced probe subset uses group metadata. External teacher pretraining
uses image–text pairs. The label-free claim applies to graph discovery and SSL.

## Code navigation

| Responsibility | Source |
|---|---|
| Sparse decomposition | `splice/model.py`, `splice/splice.py`, `splice/admm.py` |
| Frozen cache | `scripts/tools/cache_crp_features.py` |
| Graph audit | `splice/crp.py`, `splice/graph_io.py` |
| Graph sampler and KL | `splice/crp_training.py` |
| Student training | `spur_splice.py`, `experiments/spurious_eval/training/ssl_loop.py` |
| Dataset adapters | `experiments/spurious_eval/datasets/` |
| Logistic evaluation | `experiments/spurious_eval/linear_probe.py`, `experiments/spurious_eval/training/logistic_probe.py` |
| Group screen | `splice/crp_group_screen.py` |
| Optional search and safe graph | `scripts/tools/search_crp_graphs.py`, `splice/crp_graph_selection.py`, `splice/crp_safe_graph.py` |
| Spatial evidence | `CoBalT/spatial.py`, `CoBalT/train_spatial.py`, `CoBalT/extract_spatial_balance.py`, `splice/spatial_balance.py` |
| CoBalT compatibility control | `CoBalT/train_discovery.py`, `CoBalT/train_classifier.py`, `splice/cobalt_check.py` |
| Matched raw teacher | `scripts/tools/build_crp_baseline_graphs.py` |
| Control orchestration | `scripts/tools/run_crp_controls.py` |
| Saved-probe and signal checks | `scripts/tools/check_saved_probes.py`, `scripts/tools/run_crp_signal_checks.py` |
| Direct reconstruction transfer | `splice/concept_distillation.py`, `scripts/tools/run_concept_transfer.py` |
| Gradient diagnostics | `scripts/tools/run_gradient_diagnostic.py` |
| Evidence and manuscript | `paper_results.json`, `Spur_SpLiCE.tex` |

## Experiment state

- The four-seed main study compares five arms at 500 epochs and KL weight 2.
- The weight screen compares CRP and raw CLIP at 0.2 and 0.5 on seeds 1 and 2.
- Frozen representations, earlier saved-probe integrity and graph localization
  are reviewed in `docs/RESULTS_REVIEW_2026-09-06_TRANSFER.md`.
- Direct transfer now has eight completed runs; full W&B histories and three
  gradient diagnostic runs are retained in `outputs/next_tests_2026-09-07/review`.
  The direct screen failed; diagnostics have AMP measurement and recipe limitations.
- Read `docs/TRANSFER_REVIEW_2026-09-07.md` and `docs/ACTIVE_PLAN.md` for the
  current result and follow-up. Keep direct results separate from CRP tables.
- Spatial variants remain outside this completed direct-transfer review.

Use [scripts/README.md](scripts/README.md) for commands and the manuscript for
exact settings and limitations. The generic `train_crp.conf` is not the locked
configuration of every historical experiment.
