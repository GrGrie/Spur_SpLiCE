# AGENTS.md

This file is the entry point for coding agents working in this repository.

Keep it concise. It defines how agents should work, what must not be violated,
and which project documentation to read next.

## 1. Startup

At the beginning of a task:

1. Read this file.
2. Read `PROJECT_MAP.md`.
3. Decide which project context is actually needed for the task.
4. Read only that context.
5. Inspect only the relevant source files.

Do not read all documentation by default.
Do not read the full project history by default.

Typical context:

- Ordinary implementation task:
  `AGENTS.md` + `PROJECT_MAP.md` + relevant code.
- Architecture/planning task:
  also read `docs/CURRENT_METHOD.md`.
- Experiment, probe, evaluation, or claim-related task:
  also read `docs/EXPERIMENT_PROTOCOL.md`.
- Historical question:
  search `docs/Project History.md` for the relevant topic/date; do not read it
  completely unless the task genuinely requires the full chronology.
- Implementation from an existing plan:
  prefer `AGENTS.md` + `PROJECT_MAP.md` + the plan + referenced code. Do not
  re-read broad architecture documentation unless the plan is ambiguous.

## 2. Agent workflow

For non-trivial work, separate architecture from implementation.

Prefer small, reviewable changes over broad autonomous modifications.

Do not:
- expand task scope without explicit justification;
- perform unrelated cleanup;
- introduce speculative abstractions;
- add dependencies without a concrete need;
- repeatedly inspect files already understood;
- create subagents unless explicitly requested.

Use repository documentation as persistent context instead of relying on
previous agent conversations.

### Architecture / planning mode

When explicitly asked to plan, design, investigate architecture, or produce
an implementation specification:

- inspect the relevant implementation;
- identify affected files, symbols, interfaces, and configuration;
- identify invariants, edge cases, risks, and compatibility concerns;
- make the necessary architectural decisions;
- produce ordered implementation steps;
- define verification.

Unless explicitly requested:
- do not modify production code;
- do not implement the proposed solution;
- do not perform unrelated refactors;
- do not spawn subagents;
- do not run broad or expensive test suites.

The final plan should be concrete enough that another coding agent can
implement it without making substantial new architectural decisions.

Stop after producing the plan.

### Implementation mode

When an implementation plan or specification is provided:

- treat it as the architectural source of truth for the task;
- inspect the referenced code before modifying it;
- implement the plan in order;
- prefer the smallest correct diff;
- follow existing repository conventions;
- run narrow relevant verification after changes.

Do not redesign the architecture merely because another approach is possible.

If the plan conflicts with the actual repository state or appears technically
incorrect, report the conflict instead of silently inventing a new design.

### Debugging / investigation mode

When asked to diagnose unexpected behavior:

- reproduce or localize the issue before changing architecture;
- distinguish evidence from hypotheses;
- inspect the narrowest relevant execution path first;
- avoid speculative fixes;
- report the identified cause and supporting evidence.

If a fix requires an architectural decision, surface that decision instead
of silently broadening the task.

## 3. Research-code principles

This is a research repository.

Reproducibility, experimental clarity, falsifiability, and inspectability are
more important than production-style infrastructure.

Prefer:
- explicit experiment configuration;
- simple implementations;
- deterministic behavior where practical;
- clear separation between method, experiment, evaluation, and reporting;
- minimal abstractions that directly help research.

Avoid engineering machinery that does not materially improve research
correctness, reproducibility, or interpretation.

Do not introduce infrastructure merely because it would be conventional in a
production software project.

## 4. Information boundary

Discovery and SSL must remain label-free.

Target, spurious-attribute, group, and metadata fields must not influence
concept selection, teacher-edge selection, graph weights, or SSL training.

Hidden labels may be used only by post-hoc diagnostics and supervised
downstream evaluation.

If a task touches this boundary, read `docs/CURRENT_METHOD.md` and
`docs/EXPERIMENT_PROTOCOL.md` before changing code.

## 5. Verification

Every implementation must have explicit verification.

Prefer:
1. targeted unit or functional test;
2. relevant smoke test;
3. broader suite only when justified.

Do not run expensive training or broad test suites merely for reassurance.

Never claim that a test, experiment, or command succeeded unless it was
actually executed.

## 6. Experiment tracking

Every script that launches full SSL training must enable and configure W&B
tracking.

This applies to real SSL training runs, including SimCLR and CRP training.
It does not apply to graph building, frozen audits, reports, or other
diagnostic-only jobs.

For real training runs, record enough information to reconstruct and
interpret the experiment, including:
- project/entity/run identity;
- group and tags;
- resolved configuration;
- relevant runtime/version information;
- per-epoch SSL metrics;
- evaluation metrics.

Treat W&B as the durable experiment record.

Local checkpoints may be processed or deleted after training and must not be
the only source of experiment results.

Disabling W&B is acceptable only for explicit smoke tests or local debugging.

## 7. Cluster jobs

Cluster job submission must use:

    sbatch path/to/standalone.sbatch

Do not prefix `sbatch` with temporary environment assignments or
experiment-specific command-line arguments.

Experiment-specific overrides must be placed in the configuration file
referenced by the `.sbatch` script before launch. The `.sbatch` file must load
that configuration itself.

Diagnostic commands such as `squeue`, `sacct`, and `scontrol` may use their
normal command-line arguments.

Current canonical launchers are listed in `PROJECT_MAP.md`.

## 8. Documentation maintenance

`PROJECT_MAP.md` is a short navigation layer, not a changelog.

`docs/CURRENT_METHOD.md` describes the current method and architecture.

`docs/EXPERIMENT_PROTOCOL.md` describes current controls, evaluation,
reproducibility rules, and supported-claim boundaries.

`docs/Project History.md` is the durable research chronology.

Update current-state documentation only when the corresponding current
method, architecture, protocol, or file map materially changes.

Record material method, loss, experiment-protocol, reporting, or
interpretation changes as new dated entries in `docs/Project History.md`.

Never rewrite old history to make it match the current method.

## 9. Context and token efficiency

Keep repository exploration targeted to the current task.

Do not load large logs, datasets, generated outputs, checkpoints, coverage
reports, or the full project history unless necessary.

When only part of a file is relevant, inspect that part rather than unrelated
content.

Prefer concise summaries of command output over retaining large raw outputs.

For implementation tasks with an existing plan, use the plan to decide what
repository context is actually required.
