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
SpLiCE produces a scalar spurious-background score from a manual list of concept
names or vocabulary indices passed with `--splice_concepts`.

The current implementation supports two interventions:

- `augment`: concept-aware augmentation. Before SSL training starts, SpLiCE
  scores every Waterbirds training image. Images whose selected concept score is
  at least `--splice_score_threshold` receive stronger SimCLR augmentations
  (tighter random crops, stronger color jitter, grayscale, and blur). This is
  meant to weaken easy background shortcuts while preserving the normal SimCLR
  objective.
- `corr_reg`: correlation regularization. Each SSL batch carries the frozen
  SpLiCE score for every image. The training loop penalizes squared correlation
  between ResNet encoder features and those scores, scaled by `--splice_weight`.
  This encourages the learned representation to encode less information about
  the selected background concepts.

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
  --splice_score_threshold 0.01

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

Expected outcomes:

- Average accuracy may stay similar or drop slightly.
- Worst-group accuracy should improve if the selected concepts are genuinely
  spurious background cues.
- If `--splice_score_threshold` is too low, too many samples receive strong
  augmentation and training may become noisy.
- If `--splice_weight` is too high, useful representation quality can degrade.
- If the chosen concepts describe the bird itself rather than the background,
  both average and worst-group accuracy may suffer.

Recommended experiment order:

1. Train the `none` baseline and record average and worst-group linear-probe
   accuracy.
2. Run `augment` with a small, auditable concept list.
3. Run `corr_reg` with the same concept list and a conservative
   `--splice_weight`.
4. Run `augment_corr_reg` only after the individual methods are understood.
5. Compare linear-probe worst-group accuracy, average accuracy, and logged
   `SSL splice loss`.
