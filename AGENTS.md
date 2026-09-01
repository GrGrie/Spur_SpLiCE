# Project instructions

## Communication constraint

- Never in this project bring up or mention SHA256 or any other SHA.

## Project context and chronology

- At the beginning of every new chat, read `PROJECT_GROUND_TRUTH.md` completely
  before proposing, implementing, or interpreting project work.
- Treat `PROJECT_GROUND_TRUTH.md` as both the canonical current specification and
  the durable, human-readable research chronology.
- After every material method, architecture, loss, experiment-protocol, reporting,
  or interpretation change, append a dated entry to the chronology in
  `PROJECT_GROUND_TRUTH.md` during the same task.
- A chronology entry must record the state before the change, the change itself,
  the reason, its consequences for experiments and claims, and whether it belongs
  to CRP, CQT, a shared trainer, a baseline, or reporting only.
- Never rewrite an older chronology entry to make it match the current method.
  Add a new correction or superseding entry so historical reporting snapshots
  remain reconstructable.
- Keep the current research state separate from the reporting snapshot. A report
  may intentionally describe an earlier CRP-only stage while current development
  has already advanced to CQT.

## Training experiment tracking

- Every script that launches full SSL training must enable and configure W&B tracking.
- This requirement applies to SSL training (including SimCLR and CRP training), but not to graph-build, frozen-audit, report, or other diagnostic-only jobs.
- Pass the W&B project, entity, run name, group, tags, resolved configuration, runtime versions, per-epoch SSL metrics, and evaluation metrics to the training entry point.
- Treat W&B as the durable record of a training experiment: `.pt`/`.pth` checkpoints may be processed and deleted after training, so do not make them the only source of experiment results.
- Disabling W&B is acceptable only for explicit smoke tests or local debugging, not for real training runs.
