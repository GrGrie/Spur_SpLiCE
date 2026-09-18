# AGENTS.md — Working with Astra

Source: [OpenAI Astra prompting guidance](https://developers.openai.com/api/docs/guides/latest-model), reviewed 2026-09-06. Condensed and adapted to this repository.

- Infer intent and scope from context. Complete authorized work; make reasonable assumptions for routine gaps.
- Treat action requests as instructions to act. For analysis-only or planning requests, deliver the artifact before stopping; do not implement unrequested changes.
- Ask focused questions when missing information materially affects the result. Continue independent work meanwhile.
- Prepare reviewable work before requesting necessary approval. Do not repeat authorization requests already resolved.
- Check relevant instruction files for ambiguity. Explicit user instructions override skill guidance, subject to higher-priority rules.
- When a skill blocks progress, link its file, quote the rule, and explain its application. Separate requirements from interpretation.
- Lead with the result. Use concise, plain paragraphs; lists and tables should clarify steps or comparisons. Avoid jargon and stock phrases.
- Delegation policy: use subagents only when explicitly requested by the user.
- Match verification to the change. Broaden or repeat checks only for changes, failures, or unresolved concerns.
- Incorporate corrections and answer side questions while preserving the larger task.
- When reporting information to me, be extremely concise and sacrifice grammar for the sake of concision.

## Project rules

### Language and writing style

Applies to code, comments, docstrings, log messages, commit messages, Markdown files and the paper (`.tex`).

- Write everything in English. Collaborators read this repository in English.
- List items as "A, B and C" and join clauses as "X and Y", with no comma before "and".
- Refer to the confounding factor as the "spurious attribute" or "spurious concept". The word "nuisance" stays out of all text.
- State each claim affirmatively: say what a thing is and what it does. Keep negation and "X, not Y" contrast for the rare sentence where the contrast is itself the result.

### Cluster execution (SLURM)

- The university SLURM cluster runs all training, probing, SpLiCE caching, concept grouping, teacher-graph builds and anything that writes checkpoints. The local PC runs plotting, report rendering and short CPU checks.
- The cluster accepts work only through `sbatch`. Every cluster task ships with its own `.sbatch` or `.sh` launcher in `scripts/`, added in the same change as the Python entry point. Launchers source `scripts/load_splice_cluster_env.sh` and take hyperparameters from the experiment manifest or CLI arguments.
- Resources per job: one V100 (`--gres=gpu:1`), `--cpus-per-task` at most 5, `--mem` at most 8G per requested CPU (1 CPU → 8G, 2 → 16G, 5 → 40G). Set DataLoader `num_workers` within `--cpus-per-task`.
- V100 supports float16 mixed precision; use `torch.float16` autocast with a `GradScaler`.

### Storage

- `/home` has a 100 GB quota with about 70 GB in use. The repository lives there. Checkpoints, `.pt` feature caches and other bulky binaries go to `/scratch/xar68reb/CoSpRo/` through `splice.artifacts`.
- `/scratch` is a slow HDD: write each binary once at the end of a stage and delete intermediate epoch checkpoints during training.
- Every job writes its compact result JSON under `outputs/`. Git synchronizes these files between the cluster, the laptop and the PC, so every machine sees run status and metrics. Placement rules live in [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md).

### Experiment tracking

- Every SSL training job logs to Weights & Biases. Worst-group accuracy (WGA) is the headline metric: log it at every probe epoch under a stable key and store it in the W&B run summary and in `run.json`.
