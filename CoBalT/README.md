# CoBalT reproduction

This directory is an independent PyTorch implementation of **CoBalT** from
Md Rifat Arefin et al., *Unsupervised Concept Discovery Mitigates Spurious
Correlations* (ICML 2024): <https://arxiv.org/abs/2402.13368>.

The authors' repository still says “Code is coming soon”, so this code is based
on the paper, its appendix, and the official implementation of the SlotCon
method on which stage 1 is based: <https://github.com/CVMI-Lab/SlotCon>.

## Implemented protocol

The current scope is the two datasets already supported by this project and
needed for a matched comparison: **Waterbirds** and **CelebA**.

1. `train_discovery.py` trains the ImageNet-pretrained ResNet-50 student and EMA
   teacher, spatial projector, four semantic slots, and an eight-entry VQ
   concept dictionary. The optimized objective is
   `L_distillation + L_contrastive + L_vq` (paper equation 5).
2. `extract_concepts.py` freezes stage 1 and assigns every train/validation/test
   image to the dictionary entries represented by its active slots.
3. `train_classifier.py` trains a separate ImageNet-pretrained ResNet-50 with
   Algorithm 1's concept-balanced sampling. It records checkpoints selected by
   average validation accuracy (`avg`, CoBalT_avg), inferred worst-group
   accuracy (`ig`, CoBalT_ig), and human worst-group accuracy (`hg`, CoBalT_hg).

No target or spurious-attribute label enters concept discovery. The later
classifier is supervised, and human group metadata is used only for `hg` model
selection and final reporting.

## Paper defaults

| Setting | Waterbirds | CelebA |
|---|---:|---:|
| Discovery epochs | 50 | 20 |
| Classifier epochs | 300 | 60 |
| Sampling lambda | 2 | 1 |
| Discovery optimizer | Adam, lr 2e-4, wd 5e-4 | same |
| Classifier optimizer | SGD, lr 1e-4, momentum .9, wd .1 | same |
| Batch size | 128 | 128 |
| Slots / slot dimension | 4 / 32 | 4 / 32 |
| Student / teacher temperature | .1 / .07 | .1 / .07 |
| Teacher / dictionary EMA | .99 / .9 | .99 / .9 |

The default codebook size is 8. The paper explicitly uses 8 for IN-9L and
reports 8 as the best CelebA ablation, but does not unambiguously state the size
used for every main-table run. It is therefore a visible CLI setting rather
than a hidden assumption.

The paper's codebook update equation writes a batch sum while describing an EMA
of representations. A literal sum makes code magnitude depend on batch
occupancy; this implementation uses the conventional per-code batch mean and
L2-normalizes the updated code. The exact crop schedule and every preprocessing
detail are also absent from the paper. We use the published SlotCon-style two
random resized crops, color jitter, grayscale, horizontal flip, and compare
attention only on their geometric overlap.

## Run

The three-seed Slurm array launcher enables W&B for both full training stages and records the
resolved configuration, runtime versions, per-epoch losses/metrics, selection
metrics, and final test metrics:

```bash
sbatch CoBalT/scripts/train_cobalt.sbatch
```

For CelebA, change `DATASET="celeba"` in the launcher. The paper reports the
following targets:

| Dataset | CoBalT_ig worst / average | CoBalT_avg worst / average |
|---|---:|---:|
| Waterbirds | 89.0±1.6 / 92.5±1.7 | 90.6±0.7 / 93.8±0.8 |
| CelebA | 89.2±1.2 / 92.3±0.6 | 81.1±2.7 / 92.8±0.9 |

W&B can be disabled only for an explicit smoke test by passing both `--smoke`
and `--no-wandb`. Full runs reject `--no-wandb`.

## Label-free check inside CRP/CQT grouping

To use only Stage 1 concepts as an optional balance check while building SpLiCE
concept groups, run:

```bash
sbatch CoBalT/scripts/prepare_concepts.sbatch
sbatch --export=ALL,COBALT=true scripts/train_crp.sbatch
```

The preparation job runs discovery and fixed concept extraction, but does not
train the CoBalT classifier. CRP/CQT converts memberships into mean-one sample
weights proportional to the sum of inverse concept frequencies. Those weights
change concept frequency filtering and coactivation during grouping only. This
preserves the project's label-free graph boundary and should be reported as a
CoBalT-inspired concept-balance check, not as the full CoBalT Stage 2 method.
