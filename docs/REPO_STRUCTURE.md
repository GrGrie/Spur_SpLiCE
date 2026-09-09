# Repository structure and artifact policy

This document is the canonical rulebook for file placement, naming, retention,
and Git synchronization in Spur SpLiCE. Code, launchers, and cleanup tools must
follow it. Historical paths in old JSON files are provenance, not current
placement instructions.

## Repository map

```text
Spur_SpLiCE/
  splice/                         reusable method and artifact-lifecycle code
  experiments/
    manifests/                   versioned experiment definitions
    spurious_eval/               datasets, training, and evaluation code
    runner.py                    canonical seed/arm launcher
  scripts/
    tools/                       maintenance, collection, and reporting tools
    *.sh                         user-facing cluster submission helpers
    *.sbatch                     Slurm job bodies
  schemas/                       versioned machine-readable contracts
  tests/                         unit tests and small committed fixtures
  data/
    vocab/                       small immutable vocabularies
    means/                       small immutable model statistics
  outputs/                       Git-synchronized records; never bulk tensors
  docs/
    REPO_STRUCTURE.md            this canonical policy
    history/                     dated migration and recovery records
  paper_results.json             checked compact registry for paper claims
  CoSpRo.tex                     paper source
```

`output/`, `save/`, `tmp/`, `wandb/`, and `datasets/` are machine-local legacy
or runtime directories. New code must not write research results there.

## Canonical `outputs/` tree

```text
outputs/
  seeds/<study>/seed_<NN>/<arm>/<attempt_id>/
    run.json
    command.json
    execution.json
  reports/<study>/
    results.json
    *.csv
  shared/<dataset>/
    graphs/*.json
    configs/*.json
  reference/
    wandb_exports/*.json
    vocabularies/*.txt
  SLURM/
    .gitkeep
  shared/legacy/                  ignored quarantine only
```

### Meanings

- `seeds/` contains one immutable record per execution attempt. `run.json` is
  the local source of truth for configuration, metric events, final metrics,
  runtime provenance, W&B identity, failures, and artifact attestations.
- `reports/` contains study-level aggregates. The primary analysis input for
  Astra/Sol is `outputs/reports/<study>/results.json`, generated from all
  expected seed/arm run records.
- `shared/` contains small, seed-independent and reproducible inputs such as
  graph JSON and configuration JSON. Tensor caches do not belong in Git.
- `reference/` contains small external exports needed to interpret results.
- `SLURM/` contains ignored stdout/stderr. Only `.gitkeep` is versioned.
- `shared/legacy/` is temporary quarantine. Its files are never committed.

## Git synchronization matrix

| Artifact | Git | Windows | Cluster |
|---|---:|---:|---:|
| Source, manifests, schemas, tests, maintained docs | yes | yes | yes |
| `run.json`, `results.json`, compact JSON/CSV/YAML/TXT | yes | yes | yes |
| Teacher graph represented as compact JSON | yes | yes | yes |
| `.pt`, `.pth`, `.ckpt`, `.safetensors` | no | disposable | scratch only |
| Large feature tensors and caches | no | disposable | scratch only |
| Slurm logs, W&B runtime directories | no | unnecessary | local runtime |
| HTML, images, PDFs, compressed visual reports | no | optional local | optional local |
| Datasets | no | external path | external path |
| Legacy quarantine/archive payload | no | temporary | scratch/archive |
| Legacy archive manifest with hashes | yes | yes | yes |

Git accepts only `.json`, `.jsonl`, `.csv`, `.yaml`, `.yml`, `.md`, and `.txt`
inside the four canonical output trees. This is an allowlist: a new binary
extension remains ignored until the policy is deliberately changed.

## Cluster storage

`SPUR_SPLICE_OUTPUT_ROOT` selects the Git-facing output tree and normally stays
unset, making it `<repository>/outputs`.

`SPUR_SPLICE_SCRATCH_ROOT` selects non-Git binary storage and defaults on the
cluster to:

```text
/scratch/xar68reb/CoSpRo/
  checkpoints/Spur_SpLiCE/<study>/seed_<NN>/<arm>/<attempt_id>/
  features/Spur_SpLiCE/<study>/seed_<NN>/<arm>/<attempt_id>/
  legacy_archives/Spur_SpLiCE/
```

`SPUR_SPLICE_ARTIFACT_ROOT` is accepted only as a compatibility fallback for
the scratch root. It must never redirect the Git-facing `outputs/` tree.

Retained cluster checkpoints must be registered in `run.json` with kind,
stage, epoch, `artifact://` URI, byte size, SHA-256, verification time, and
retention state. A run may be declared complete only after retained artifacts
pass integrity verification. If a checkpoint is intentionally removed after
all required metrics have been materialized, its record must say so explicitly
rather than leaving a broken `retained` attestation.

## Naming rules

- Use lowercase `snake_case` for studies and arms: `waterbirds_crp`,
  `splice_crp_kl`.
- Use `seed_<NN>` with at least two digits: `seed_01`.
- Use an immutable attempt identifier: `primary`, a Slurm-derived ID, or an
  ISO-like UTC timestamp. Never overwrite a different attempt.
- Use stable semantic names, not dates alone: `test_summary.json`,
  `teacher_graph.json`, `results.json`.
- Dates are appropriate for snapshots and migrations, for example
  `pre-unification-2026-09-09.json`.
- Avoid spaces, parentheses, `copy`, and numeric duplicate suffixes such as
  `(1)` in canonical paths.

## Result lifecycle

1. Define the study in `experiments/manifests/<study>.json`.
2. Launch through `experiments.runner` or `scripts/submit_experiment.sh`.
3. Write small records to the attempt directory under `outputs/seeds/`.
4. Route checkpoints and large feature payloads to the scratch root.
5. Record metrics and artifact integrity in `run.json` atomically.
6. Run `scripts.tools.collect_results` after the full matrix. A missing,
   failed, or unattested required run makes the aggregate `partial`.
7. Commit and push the compact run records and aggregate result JSON.
8. Apply retention only after the aggregate and required downstream analyses
   exist. Record intentional checkpoint removal in provenance.

The result JSON must be sufficient for ordinary scientific comparison without
W&B or checkpoint access: it should include the manifest, expected and missing
matrix entries, statuses, final average and worst-group metrics, run identity,
and limitations. Checkpoints are retained only for analyses that genuinely
require model weights, not as substitutes for missing metrics.

## Windows policy

Windows may run experiments, but it follows the same output schema. Commit the
compact JSON/CSV records so they reach the cluster. Local tensor caches and
checkpoints are disposable only when a valid JSON result records the completed
metrics or when the tensor is a reproducible cache. Otherwise move the binary
and its evidence into `outputs/shared/legacy/` until reviewed.

Do not copy the cluster scratch tree into the repository. A temporary packaged
cluster copy belongs in quarantine and must be removed after its compact
results have been promoted and its archive has been verified.

For the pre-unification Windows quarantine, run
`python scripts/tools/cleanup_local_outputs.py` without flags first. Its
`--apply` mode records the complete decision and promoted legacy CSV rows in
`outputs/reports/local-cleanup/windows-pre-unification.json`. It deletes a
checkpoint only with result evidence, deletes reproducible tensor caches, and
keeps ambiguous files for manual review. The utility supports Windows paths
longer than 260 characters.

## Legacy cleanup

Unknown historical material first goes to `outputs/shared/legacy/`. On the
cluster use:

```bash
bash scripts/submit_legacy_archive.sh --scan
bash scripts/submit_legacy_archive.sh --archive pre-unification
```

The archive payload stays under scratch; its hash manifest goes under
`outputs/reports/legacy-archive/` and is committed. After review, the same
archive is reverified before deletion:

```bash
bash scripts/submit_legacy_archive.sh --delete-after-verify pre-unification
```

If scratch has an automatic purge policy, copy the archive to durable storage
before deleting quarantine. A manifest proves integrity but is not a backup.

If a historical archive manifest was already lost, promote its compact run and
analysis records first. Reconstructing a full per-file manifest is optional
when the retained archive is not being deleted; the recovery status must state
that limitation and provide a reproducible reconstruction command. New archive
operations still require a manifest before source deletion.

## Rules for changing the layout

- Resolve destinations through `splice.artifacts`; do not hard-code new output
  roots in training or analysis code.
- Update this document, `.gitignore`, `outputs/README.md`, and relevant tests in
  the same commit when adding an artifact class.
- Never commit a large file merely because Git permits its size.
- Never delete ambiguous experimental data. Archive or quarantine it first.
- Prefer one aggregate result per study over thousands of redundant report
  files. Keep raw per-attempt facts in `run.json` and derived comparisons in
  `results.json`.
