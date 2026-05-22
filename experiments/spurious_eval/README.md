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

The root `spur_splice.py` is the SSL training entrypoint. Use `--use_splice` and
`--splice_weight` there to enable SpLiCE as an additional training loss once the
regularizer is implemented in `splice/ssl_regularization.py`.
