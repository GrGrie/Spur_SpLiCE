# Spurious evaluation

This package supplies Waterbirds, CelebA and SpurCIFAR10 adapters, ResNet/SimCLR
models, checkpointing and supervised evaluation of frozen representations.
The root `spur_splice.py` connects them to CRP and direct reconstruction transfer.

## Supported training modes

- `none`: ordinary SimCLR with two standard augmented views.
- `crp_relational`: graph-aware batches and confidence-weighted backbone KL.
- `frozen_concept_distill`: Waterbirds fixed raw/reconstructed/shuffled targets
  predicted through a separate head.

Graph and transfer datasets expose sample indices to their losses without target
or group annotations. Dataset objects also support supervised evaluation.
Legacy routed augmentation, conditional correlation and edited-code synthesis
have been retired.

## Linear evaluation

```bash
python linear_probe.py \
  --dataset waterbirds --data_folder ./datasets \
  --train_set_linear_layer ds_train --eval_split val \
  --model resnet18_large --ckpt /path/to/epoch_500.pth \
  --probe_solver logistic --probe_l2 0.001 \
  --probe_tolerance 0.000001 --probe_max_epochs 200
```

The logistic solver standardizes features using training statistics and optimizes
float64 cross-entropy plus weight L2 with full-batch L-BFGS. It checks gradient
convergence for ten consecutive external optimization epochs. The final ten
converged epochs form the averaging window; they are not independent seeds.

`ds_train` is group-balanced and uses metadata: on Waterbirds it contains
56 images per group. Validation has 1,199 images. The 500-epoch controls retain
the historical seeded probe-view protocol. Frozen-signal diagnostics use
deterministic teacher preprocessing and are separate comparisons.

Development evaluates on validation. Final-test evaluation requires
`--final_test` after fixing the protocol. Periodic probing is RNG-isolated.
Check retention settings before expecting to re-extract historical features.

## Source map

- `datasets/`: splits, transformations, metadata and registration.
- `models/`: encoder and projection heads.
- `losses/`: contrastive loss.
- `training/`: SSL/probe loops, checkpointing and optimization.
- `metrics.py`: average, group and representation-rank metrics.
- `linear_probe.py`: standalone probe implementation.

See [the project map](../../PROJECT_MAP.md) and
[the manuscript](../../Spur_SpLiCE.tex) for research interpretation.
