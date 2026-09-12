# Finish the checkpoint comparison and graph figure

These tools implement points 1 and 4 only. LA-SSL seeds 2–4 already have launchers; additional datasets are already running. The larger-probe-budget suggestion is withdrawn. No new SSL training is required by either workflow below.

Run commands from the repository root on the cluster in `grgrie-train`. The checkpoints and Waterbirds images are not present on this Windows checkout, so the real evaluation/rendering must run where those artifacts live.

## 1. Evaluate the eight completed control checkpoints

The evaluator selects the already reported semantic-graph and reconstruction runs for seeds 1–4 by W&B run ID. It reads canonical `run.json` attestations and the retained historical reports; it does not query W&B or choose a checkpoint by accuracy. Old seeds 1/3 and new seeds 2/4 may have different scratch layouts, which are resolved automatically. `SPUR_SPLICE_SCRATCH_ROOT` supports relocating the scratch prefix.

```bash
export DATA_FOLDER=/home/xar68reb/Datasets

# Inspect the eight identities/paths without loading data or checkpoint tensors.
python -m scripts.tools.evaluate_submission_checkpoints prepare --dry-run

# Validate and hash all checkpoints and create a timestamped evaluation plan.
# This reads checkpoint metadata but does not evaluate test images.
python -m scripts.tools.evaluate_submission_checkpoints prepare

# Run the two frozen-encoder probes for each seed (four jobs).
mkdir -p outputs/SLURM
sbatch scripts/run_submission_checkpoint_tests.sh
```

Alternatively, run sequentially without SLURM:

```bash
python -m scripts.tools.evaluate_submission_checkpoints run
```

This calls the existing standalone linear-probe implementation, never the SSL trainer or the manifest `--locked-test` training route. Only the probe is fitted. Settings remain ResNet-18 large, SSL epoch 500, seed-matched randomness, 224 group-balanced `ds_train` images, batch size 128, four workers, the existing random-crop/flip probe-training views and resized test views, float64 logistic fitting, L2 0.001, tolerance 1e-6 and maximum 200 external probe epochs. It does not switch to a different probe preprocessing protocol. The auxiliary background probe and W&B logging are omitted because neither is needed for this comparison; target-probe fitting is unchanged.

Existing successful evaluations are reused only if their receipt and result hashes match the plan. An interrupted probe can be rerun with the same command. Checkpoints are never modified. Changed checkpoint/data/code hashes fail explicitly. If evaluation code changes after preparation, use a new output directory and prepare again; do not edit the old lock.

If files were relocated beyond the scratch-prefix change, create a JSON map containing only overrides you need:

```json
{
  "semantic_splice:2": "/your/location/semantic_seed2/last.pth",
  "splice_reconstruction:4": "/your/location/reconstruction_seed4/last.pth"
}
```

Pass `--checkpoints-json paths.json` to `prepare`. Archived hashes still have to match. `--artifact-root` changes the metadata/report root. `--output-dir` selects a new evaluation directory; supply the same value to prepare, run, summarize and the SLURM script.

After all four jobs finish:

```bash
python -m scripts.tools.evaluate_submission_checkpoints summarize
```

Default output: `outputs/reports/submission_checkpoint_test/`:

- `lock.json`: timestamp, checkpoint identities/hashes, exact probe protocol, dataset metadata hash, evaluation-code hash and hashes of the 20 reused baseline test probes.
- `seed_0S/ARM/`: individual test probe JSON and completion receipt. Large feature tensors follow the existing scratch policy.
- `results.json`: all 28 endpoints, four-seed mean/sample SD and paired CoSpRo-minus-control differences.
- `results.md`: aggregate, paired and individual/per-group tables.
- `table.tex`: main table ready to incorporate into `CoSpRo.tex`.

Aggregation refuses validation results, unconverged probes, incorrect group counts, changed evidence, missing seeds or duplicate rows. It reuses the five original arms' existing held-out test results rather than rerunning their training. The final ten probe epochs are averaged within a run; uncertainty is measured across the four student seeds. No numerical result or manuscript table is updated until the real jobs finish.

## 4. Build the graph-evidence figure

The existing detailed renderer requires ReportLab; the compact figure also uses Matplotlib and Pillow. Install missing rendering dependencies in your chosen rendering environment if necessary:

```bash
python -m pip install matplotlib reportlab pillow
```

Use the directory that directly contains `metadata.csv` and the Waterbirds images:

```bash
python -m scripts.tools.build_submission_figure \
  --dataset-root /path/to/waterbirds \
  --output-dir outputs/reports/submission_graph_figure
```

No GPU or model inference is required. The tool reads the existing frozen 20-pair manifest and both teacher graphs. It checks sample IDs and retention status against the graph, annotates pairs from training metadata, and generates:

- `graph_evidence.pdf` and `.png`: four illustrative categories (retained/rejected crossed with same-target cross-background/wrong-target), plus relation-composition charts.
- `evidence.json`: every annotated pair, the deterministic selection rule, selected examples, all source hashes and raw/percentage graph-mass statistics.
- `details/concept_panels.pdf` and `.json`: the existing full five-group panel audit, written from a copy of the source selection.
- `caption.txt` and `figure.tex`: caption and LaTeX inclusion snippet. Add `\usepackage{graphicx}` if needed; adjust the PDF path if copying the figure to another location.

The compact examples are explicitly label-stratified illustrations, selected by greatest gain within each category of the existing pool; they are not random samples or prevalence estimates. If the fixed pool lacks a category, the figure says so rather than silently substituting another example. The composition charts use **all graph edges**, weighted by `anchor_confidence[i] * weights[i,j]`, normalized separately within each source group. Wrong-target mass includes both backgrounds. Labels enter only this post-hoc analysis.

The tool refuses to overwrite an existing output directory. The original graph and selection files remain unchanged. Inspect the real image panels before including them in the paper; synthetic test images validate rendering only and provide no scientific evidence.

## Verification performed locally

The automated tests check the exact eight run identities, all 20 existing test-probe schemas, checkpoint epoch/identity rejection, probe-only execution and reuse, changed-checkpoint rejection, incomplete/duplicate result rejection, confidence-weighted graph mass, stale annotations, absent illustration categories, preservation of the original manifest and end-to-end figure generation on synthetic images. Real checkpoint inference and real-image figures require the cluster artifacts.
