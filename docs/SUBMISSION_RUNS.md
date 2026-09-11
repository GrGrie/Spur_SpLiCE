# Submission runs

Run from the repository root with the existing `grgrie-train` environment.
The scripts add experiments only; they do not submit jobs automatically or
modify completed results. The random-crop probe protocol is unchanged.

## Missing projection-control seeds

```bash
export DATA_FOLDER=/home/xar68reb/Datasets
python -m experiments.complete_projection_controls --seed 2
python -m experiments.complete_projection_controls --seed 4
```

Cluster alternative (each job runs the two controls sequentially):

```bash
mkdir -p outputs/SLURM
sbatch scripts/run_projection_controls.sh --seed 2
sbatch scripts/run_projection_controls.sh --seed 4
```

Add `--dry-run` to either Python command to inspect commands without loading
data or launching training. Existing attempts are refused. Choose
`--attempt-id repeat-01` for an intentional fresh attempt, or `--existing reuse`
to skip a completed matching attempt when continuing the two-control sequence.

The manifests reproduce the saved `command.json` settings for seeds 1 and 3:
ResNet-18 large stem, 500 epochs, batch 128, SGD 0.01/momentum 0.9/weight decay
0.0001, LR milestones 350/400/450 with factor 0.1, SimCLR temperature 0.05,
original two-view augmentations, and the original 224-example random-crop
`ds_train` logistic probe. Validation is measured every 25 epochs with L2
0.001, tolerance 1e-6, maximum 200 probe epochs. The endpoint is epoch 500;
these commands do **not** change the historical controls to test evaluation.

Semantic transfer uses the retained semantic graph, KL weight 0.5 and relation
temperature 0.25. Direct reconstruction transfer uses weight 0.1 and the
historical 512-to-512-to-512 ReLU prediction head. Both losses start after
epoch 10 and warm up for ten epochs. Direct-transfer loading, head and loss
were restored from `78cdce2^`, not replaced with a new distillation objective.
Historical gradient diagnostics are observational and are not rerun; the
loss/head/encoder updates and all endpoint settings are preserved.

The launcher validates both inputs before starting either control. The semantic
graph is `outputs/shared/waterbirds/graphs/semantic_splice_graph.json`.
Its historical fingerprint is reproduced after normalizing JSON formatting;
the packaged newline does not change its contents. The target bank defaults to:

```text
/scratch/xar68reb/CoSpRo/legacy_archives/Spur_SpLiCE/next_actions_after_transfer_2026-09-07/direct_transfer/targets_v1.pt
```

`SPUR_SPLICE_SCRATCH_ROOT` replaces the `/scratch/xar68reb/CoSpRo` prefix.
If relocated, pass `--targets /absolute/path/to/targets_v1.pt`. The launcher
requires SHA-256 `fde777c311169e14ac2381a418dd2d68434a0d5e923f7f2a77edc59501273123`,
the retained seed-1/3 target bank. It does not regenerate targets from a new cache.

## LA-SSL

Reference: Zhu, Liu, Fernandez-Granda and Razavian,
[Making Self-supervised Learning Robust to Spurious Correlation via Learning-speed Aware Sampling](https://arxiv.org/pdf/2311.16361),
arXiv:2311.16361v2 (the version cited by `zhu2023lassl` in `CoSpRo.tex`).
The implementation follows Algorithm 1 and Eq. (3), with the similarity
definition in Section 4.1: `exp(dot(normalized projections) / temperature)`.
It computes two-view similarity for **every** training image each epoch,
smooths it by EMA, and after warm-up periodically sets sampling weights to
`max(0, s_star - gamma * (s - s_star))`, normalized to sum to one. `s_star` is
the configured quantile. Sampling is with replacement, with N draws and
ceil(N/128) optimizer steps per epoch. A weighted permutation would not
implement the required upsampling. No labels or group annotations set weights.

```bash
export DATA_FOLDER=/home/xar68reb/Datasets
python -m experiments.runner experiments/manifests/waterbirds_la_ssl.json --seed 1 --arm la_ssl
mkdir -p outputs/SLURM
sbatch scripts/run_la_ssl.sh --seed 1
```

Choose seed 1, 2, 3 or 4. The Python and SLURM commands are alternatives.
Use `--dry-run` for inspection. Default evaluation is periodic validation.
The manifest also supports the existing `--locked-test` final-only test
protocol, which starts a separate training attempt named `locked-test`; it is
not a checkpoint-only re-evaluation command. Use it only after settings are fixed.

Declared adaptation details (not a reproduction of the original numerical table):

- The cited version does not evaluate Waterbirds. We match this repository's
  ResNet-18, batch size, temperature, augmentations, 500 epochs, optimizer and
  group-balanced supervised probe rather than its ResNet-50 experiment setup.
- Adopt the paper's CelebA sampling settings: gamma 10, quantile 0.1, update
  every two epochs and ten warm-up epochs. We update at epochs 12, 14, ... .
- The cited version does not give a numerical EMA coefficient. We explicitly
  set `la_ssl_eta=0.1` (new-score weight), initialize with the first observation,
  and expose it in the manifest/CLI and W&B config. This is an adaptation choice.
- A separate FP32 eval-mode scoring pass precedes training each epoch; scores
  and EMA use float64. Scoring changes no BatchNorm statistics or gradients and
  uses isolated randomness. This adds N two-view forward evaluations per epoch;
  optimizer-step budgets match, total compute does not. Scoring time and sample
  count are logged. Scores also refresh for images not sampled for optimization.
- Sampling probabilities, EMA, epoch and scoring-generator state are saved in
  checkpoints. Restoring a checkpoint without this state is rejected for LA-SSL.

## W&B and outputs

All three additions log to project `Spur_SpLiCE`, using the existing entity
default. Each config records `study`, `arm`, `seed`, `attempt_id`, hyperparameters
and artifact paths. Training losses and periodic/final probe accuracy, WGA,
per-group metrics and convergence use the existing logger. LA-SSL additionally
logs scoring time, EMA mean, effective sample size and probability statistics.

Default run names (`S` is the seed, `A` the attempt, normally `primary`):

| Arm | W&B name | W&B group |
|---|---|---|
| semantic_splice | next_actions_graph_ablation_v1_semantic_splice_seedS_A | next_actions_graph_ablation_v1 |
| splice_reconstruction | next_actions_corrected_direct_v1_splice_reconstruction_seedS_A | next_actions_corrected_direct_v1 |
| la_ssl | waterbirds_la_ssl_seedS_A | waterbirds_la_ssl |

The first two retain the historical W&B groups for comparison with seeds 1/3.
Git-facing records and metrics follow:

```text
outputs/seeds/next_actions_after_transfer_2026_09_07_graph_ablation/seed_0S/semantic_splice/A/
outputs/seeds/next_actions_after_transfer_2026_09_07_direct_transfer/seed_0S/splice_reconstruction/A/
outputs/seeds/waterbirds_la_ssl/seed_0S/la_ssl/A/
```

Each contains `execution.json`, `command.json`, `run.json` and `training/`.
`SPUR_SPLICE_OUTPUT_ROOT` or `--output-root` changes the Git-facing root.
Checkpoints and large feature tensors use the existing scratch policy:
`$SPUR_SPLICE_SCRATCH_ROOT/{checkpoints,features}/Spur_SpLiCE/<study>/seed_0S/<arm>/A/`.
Run records link the retained artifact attestations and endpoint metrics.
The launchers do not overwrite historical aggregate reports; collect results
later with the existing `scripts.tools.collect_results` tool, supplying a new
`--output` filename if preserving an existing aggregate.

Exact manuscript replacements are in [SUBMISSION_CLAIM_CHANGES.md](SUBMISSION_CLAIM_CHANGES.md).

## Files added or changed

| Files | Purpose |
|---|---|
| `experiments/complete_projection_controls.py` | Seed launcher and historical input checks |
| `experiments/manifests/waterbirds_semantic_completion.json`, `waterbirds_direct_completion.json`, `waterbirds_la_ssl.json` | Existing runner configurations |
| `scripts/run_projection_controls.sh`, `scripts/run_la_ssl.sh` | Cluster launchers |
| `splice/concept_distillation.py` | Restored historical transfer implementation |
| `experiments/spurious_eval/training/la_ssl.py` | Learning-speed scoring and sampling state |
| `spur_splice.py` | Optional control/baseline wiring, arguments and input recording |
| `experiments/runner.py` | Scratch-path and attempt-name substitutions |
| `experiments/spurious_eval/datasets/waterbirds.py` | Restored direct-transfer dataset branch |
| `experiments/spurious_eval/models/simclr.py` | Restored optional transfer head |
| `experiments/spurious_eval/training/ssl_loop.py` | Restored transfer loss path and added diagnostics |
| `experiments/spurious_eval/training/checkpointing.py` | Adaptive sampler checkpoint state |
| `tests/test_submission_controls.py` | Historical settings, alignment, gradients, sampling and resume checks |
| `CoSpRo.tex`, `README.md`, this document, `SUBMISSION_CLAIM_CHANGES.md` | Claim calibration and reproduction instructions |
