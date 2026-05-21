# Spurious Evaluation

This folder contains a small, reusable Waterbirds linear-probing path copied in behavior from `../SpurSSL` without modifying that project.

Example:

```bash
python waterbirds_linear_probe.py \
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
