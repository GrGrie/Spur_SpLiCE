# Spurious Evaluation

This folder contains reusable spurious-correlation dataset, SSL-training, and linear-probing support copied in behavior from `../SpurSSL` without modifying that project.

Example:

```bash
python linear_probe.py \
  --data_folder ./datasets \
  --dataset waterbirds \
  --train_set_linear_layer ds_train \
  --eval_split val \
  --model resnet18_large \
  --ckpt /path/to/last.pth \
  --epochs 100 \
  --batch_size 256 \
  --learning_rate 1.0 \
  --lr_decay_epochs 60,75,90 \
  --use_wandb
```

By default, wandb logs to `gsgrechkin-rptu/Spur_SpLiCE`. Authenticate outside
the repo with `wandb login` or `WANDB_API_KEY` rather than storing the key in
source files.

The expected dataset layout is either `DATA_FOLDER/waterbirds/metadata.csv`,
`DATA_FOLDER/waterbird_complete95_forest2water2/metadata.csv`, or a
`DATA_FOLDER` that directly contains `metadata.csv`.

## Code Map

- `datasets/`: dataset adapters, WILDS-style compatibility, transforms, and dataset registration.
- `models/`: encoder and SSL model definitions.
- `losses/`: contrastive losses.
- `training/`: SSL/probe loops, checkpointing, and optimizer utilities.
- `metrics.py`: group accuracy and representation-rank metrics.
- `linear_probe.py`: reusable linear-probing command implementation used by the root `linear_probe.py` entrypoint.

The root `spur_splice.py` is the SSL training entrypoint. Use `--splice_mode`
to enable SpLiCE-guided augmentation, correlation regularization, or both.
Development runs evaluate linear probes on `val` by default. The test split is
guarded against accidental hyperparameter selection: use `--final_test` only
after the configuration has been fixed from validation results. Development
runs produce periodic validation curves every 25 SSL epochs; final-test runs
perform one probe only at the final SSL epoch.

## SpLiCE-Guided Waterbirds Experiments

Waterbirds is a binary classification benchmark where the target label is the
bird type (`landbird` vs. `waterbird`) and the spurious attribute is the
background (`land` vs. `water`). Standard training can achieve good average
accuracy while failing on minority groups such as landbirds on water or
waterbirds on land. For this reason, worst-group accuracy is the main metric.

SpLiCE is used here as a frozen interpretability and control module. It
decomposes CLIP image embeddings into sparse, human-readable concept weights
such as `water`, `lake`, `forest`, `tree`, or `grass`. The SimCLR model remains
the trainable ResNet encoder; SpLiCE does not replace the SSL backbone. Instead,
SpLiCE produces both a scalar routing score and a vector of the selected concept
weights from names or vocabulary indices passed with `--splice_concepts`. The
scalar is used only to decide which samples receive targeted augmentation. The
vector is retained for regularization so mutually exclusive concepts such as
`water` and `forest` are not collapsed into the same value.

During training, SpLiCE affects downstream predictions only indirectly through
the learned representation. In `augment`, high-score images can receive
explicitly enabled stronger SSL augmentations, which can weaken easy visual shortcuts. The first
SimCLR view remains standard and the second becomes targeted, explicitly forming
an invariance pair. In `corr_reg`, the SSL loop penalizes the full cross-correlation
matrix between encoder features and frozen SpLiCE concept vectors after centering
both within each target class. This targets within-class spurious variation without
automatically suppressing signal merely because it is correlated with the label.

The current implementation supports two interventions:

- `augment`: concept-aware augmentation. Before SSL training starts, SpLiCE
  scores every Waterbirds training image. Images whose selected concept score is
  at least `--splice_score_threshold` receive only the stronger SimCLR
  components explicitly enabled with `--splice_strong_*` flags. This is meant
  to support controlled augmentation ablations while preserving the normal
  SimCLR objective.
- `corr_reg`: conditional vector correlation regularization. Each SSL batch
  carries one frozen weight per selected concept and image. The training loop
  penalizes their normalized cross-correlation with ResNet features within
target classes, scaled by `--splice_weight`.
  This default objective uses target labels during SSL regularization; use
  `--splice_conditional_on_target false` for a fully label-free ablation.

The methods can be compared independently or together:

```bash
# Baseline SimCLR
python spur_splice.py --dataset waterbirds --data_folder ./datasets --splice_mode none

# Option B: concept-aware augmentation only
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --splice_mode augment \
  --splice_concepts "water,lake,forest,tree,grass" \
  --splice_score_threshold 0.01 \
  --splice_strong_crop \
  --splice_strong_color_jitter \
  --splice_strong_grayscale_p \
  --splice_strong_blur_p

# Option C: SpLiCE/feature correlation regularization only
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --splice_mode corr_reg \
  --splice_concepts "water,lake,forest,tree,grass" \
  --splice_weight 0.1

# Options B + C together
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --splice_mode augment_corr_reg \
  --splice_concepts "water,lake,forest,tree,grass" \
  --splice_score_threshold 0.01 \
  --splice_weight 0.1
```

Use concept names for readability when possible. Integer vocabulary indices are
also accepted, which is useful after running a concept-discovery script and
recording exact indices. The selected SpLiCE concepts are printed at startup.

To discover a candidate concept list automatically, run the standalone helper:

```bash
python tools/discover_splice_spurious_concepts.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --split train \
  --out_path outputs/waterbirds_splice_concepts.json \
  --top_k 20
```

This helper does not change training behavior. It decomposes the selected
split with frozen SpLiCE and ranks concepts by conditional association with the
spurious attribute rather than the target label. For every concept weight `c`,
binary spurious attribute `s`, and binary target label `y`, the score is:

```text
mean_y abs(E[c | s=1,y] - E[c | s=0,y])
  - label_penalty * mean_s abs(E[c | y=1,s] - E[c | y=0,s])
  - instability_penalty * std_y(E[c | s=1,y] - E[c | s=0,y])
```

The first term rewards concepts that separate the spurious attribute inside
each target class. The second term penalizes concepts that separate the target
label inside each spurious group. The instability term penalizes concepts whose
spurious effect changes sharply across target classes. This keeps discovery
dataset-aware through metadata, without hard-coding Waterbirds-specific concept
names.

Dataset adapters expose which metadata columns are target and spurious. After a
new binary spurious-correlation dataset is registered in `DATASET_REGISTRY`, you
can override those columns explicitly:

```bash
python tools/discover_splice_spurious_concepts.py \
  --dataset celeba \
  --data_folder ./datasets \
  --split train \
  --target_metadata_index 1 \
  --spurious_metadata_index 0 \
  --out_path outputs/celeba_splice_concepts.json
```

The helper writes:

- `outputs/waterbirds_splice_concepts.json`: detailed scores and group means.
- `outputs/waterbirds_splice_concepts.concepts.txt`: comma-separated concept
  names for `--splice_concepts`.
- `outputs/waterbirds_splice_concepts.indices.txt`: comma-separated vocabulary
  indices for `--splice_concepts`.
- Optional `outputs/waterbirds_splice_concepts.per_image_top10.jsonl`: target,
  spurious metadata, and top concepts per image. Enable it explicitly with
  `--per_image_top_k 10`; the default is disabled.

Example follow-up command:

```bash
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --splice_mode corr_reg \
  --splice_concepts "$(cat outputs/waterbirds_splice_concepts.concepts.txt)" \
  --splice_weight 0.1
```

By default, `augment` automatically uses the 75th percentile of the training
score distribution, so approximately 25% of samples receive the targeted second
view. Change it with `--splice_score_quantile`, or provide a fixed
`--splice_score_threshold`. To inspect the distribution manually:

```bash
python tools/summarize_splice_scores.py \
  --data_folder ./datasets \
  --split train \
  --splice_concepts "water,lake,forest,tree,grass" \
  --out_path outputs/waterbirds_splice_score_summary.json
```

Use the printed percentiles to choose how many samples should receive the
explicitly enabled targeted augmentation components. For example, using the
`p75` value means roughly the top 25% most background-concept-heavy images are
routed to the configured strong transform; using `p90` limits that route to
roughly the top 10%.

Expected outcomes:

- Average accuracy may stay similar or drop slightly.
- Worst-group accuracy should improve if the selected concepts are genuinely
  spurious background cues.
- If `--splice_score_threshold` is too low, too many samples receive the
  configured strong augmentation components and training may become noisy.
- If `--splice_weight` is too high, useful representation quality can degrade.
- If the chosen concepts describe the bird itself rather than the background,
  both average and worst-group accuracy may suffer.

Strong augmentation is now controlled as explicit deltas over the standard
SimCLR transform. If `--splice_mode augment` is used without any
`--splice_strong_*` arguments, high-score images still go through the standard
SimCLR transform. SpurCIFAR10 line recoloring is an explicit oracle ablation,
not part of the automatic default. Add only the components you want to ablate:

```bash
# Strong crop only, using the old strong default min scale 0.08.
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --splice_mode augment \
  --splice_concepts "water,lake,forest,tree,grass" \
  --splice_score_threshold 0.01 \
  --splice_strong_crop

# Strong color jitter only, with custom jitter strengths and probability.
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --splice_mode augment \
  --splice_concepts "water,lake,forest,tree,grass" \
  --splice_score_threshold 0.01 \
  --splice_strong_color_jitter 0.7,0.7,0.7,0.15 \
  --splice_strong_color_jitter_p 0.85

# Recreate the previous bundled strong transform.
python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --splice_mode augment \
  --splice_concepts "water,lake,forest,tree,grass" \
  --splice_score_threshold 0.01 \
  --splice_strong_crop \
  --splice_strong_color_jitter \
  --splice_strong_grayscale_p \
  --splice_strong_blur_p
```

For a single Slurm script, pass boolean variables as argument values:

```bash
SPLICE_STRONG_CROP="true"
SPLICE_STRONG_COLOR_JITTER="false"
SPLICE_STRONG_GRAYSCALE_P="true"
SPLICE_STRONG_BLUR_P="0.25"

python spur_splice.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --splice_mode augment \
  --splice_concepts "water,lake,forest,tree,grass" \
  --splice_score_threshold 0.01 \
  --splice_strong_crop "$SPLICE_STRONG_CROP" \
  --splice_strong_color_jitter "$SPLICE_STRONG_COLOR_JITTER" \
  --splice_strong_grayscale_p "$SPLICE_STRONG_GRAYSCALE_P" \
  --splice_strong_blur_p "$SPLICE_STRONG_BLUR_P"
```

Each strong argument accepts `true`, `false`, or a custom value. For example,
`--splice_strong_crop 0.12`, `--splice_strong_color_jitter 0.7,0.7,0.7,0.15`,
and `--splice_strong_blur_sigma 0.1,1.0` are valid.

`spur_splice.py` can also run the full SpLiCE concept pipeline before training.
Set `--splice_concepts auto` or leave `--splice_concepts` empty when a SpLiCE
mode is enabled. The script will discover top concepts, write a score summary,
then train with the selected concepts:

```bash
python spur_splice.py \
  --dataset celeba \
  --data_folder ./datasets \
  --splice_mode augment \
  --splice_concepts auto \
  --splice_auto_top_k 10 \
  --splice_score_reduction max \
  --splice_score_threshold 0.03 \
  --splice_strong_crop true
```

Automatic discovery writes `outputs/<dataset>_splice_concepts.json`,
`outputs/<dataset>_splice_concepts.concepts.txt`,
`outputs/<dataset>_splice_concepts.indices.txt`,
`outputs/<dataset>_splice_score_summary.json`.
The per-image JSONL is opt-in through `--splice_per_image_top_k`.

For `spur_cifar10`, each CIFAR-10 class is assigned one of ten fixed horizontal-line
colors. Training images receive their class-associated color with probability `0.95`;
otherwise they receive one of the other nine colors uniformly. Validation and test use
correlation `0.1`, making line color independent of class. Worst-group accuracy is
computed over the 100 `(class, line_color)` combinations. Use the CIFAR stem with
`--model resnet18` rather than `resnet18_large`.
Explicit line recoloring is deliberately disabled in the automatic pipeline,
because it injects human knowledge of the shortcut. Enable the oracle upper-bound
ablation explicitly with `--splice_strong_line_recolor true`.

CelebA follows the SpurSSL/LateTVG protocol: `Blond_Hair` is the prediction
target and `Male` is the spurious attribute used to form worst-case groups.

## SpLiCE-CBM intervention baseline

This baseline mirrors the intervention in the SpLiCE paper: train an L1 logistic
probe on sparse SpLiCE weights, then evaluate both zeroed representation columns
and zeroed probe coefficients.

```bash
python splice_cbm.py \
  --dataset waterbirds \
  --data_folder ./datasets \
  --intervention_concepts auto \
  --use_wandb
```

SpLiCE-CBM also defaults to validation. Add `--final_test` only for the locked
final evaluation.

The sparse matrices are cached without allocating a dense
`num_images x vocabulary_size` array. `auto` is the default and applies the
same metadata-conditioned concept ranking as SSL training; a manual concept
list is retained only for diagnostic ablations.

Recommended experiment order:

1. Train the `none` baseline and record average and worst-group linear-probe
   accuracy.
2. Run `augment` with a small, auditable concept list.
3. Run `corr_reg` with the same concept list and a conservative
   `--splice_weight`.
4. Run `augment_corr_reg` only after the individual methods are understood.
5. Compare linear-probe worst-group accuracy, average accuracy, and logged
   `SSL splice loss`. The linear evaluation also logs a spurious-attribute
   leakage probe by default; disable it with `--linear_spurious_probe false`.
   For comparable target accuracy, lower spurious-probe accuracy indicates that
   the shortcut is less linearly accessible in the frozen representation.
