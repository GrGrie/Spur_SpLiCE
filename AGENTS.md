# Project instructions

## Training experiment tracking

- Every script that launches full SSL training must enable and configure W&B tracking.
- This requirement applies to SSL training (including SimCLR and CRP training), but not to graph-build, frozen-audit, report, or other diagnostic-only jobs.
- Pass the W&B project, entity, run name, group, tags, resolved configuration, runtime versions, per-epoch SSL metrics, and evaluation metrics to the training entry point.
- Treat W&B as the durable record of a training experiment: `.pt`/`.pth` checkpoints may be processed and deleted after training, so do not make them the only source of experiment results.
- Disabling W&B is acceptable only for explicit smoke tests or local debugging, not for real training runs.
