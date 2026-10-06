# Results book

One page per dataset and method, built from every run record by `python -m cospro.cli.build_results_book`. Best arm: highest mean val WGA among arms with at least two finished seeds.

| dataset | method | studies | runs | best arm (≥ 2 seeds) | val WGA | val acc |
|---|---|---|---|---|---|---|
| celeba | [Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks)](celeba/concept_factors.md) | 1 | 4 | factors_celeba / xfit_norm | 85.9 ± 0.5 | 88.6 ± 0.3 |
| celeba | [CoSpRo (teacher graph: sampler and relational KL)](celeba/cospro.md) | 1 | 1 |  | | |
| celeba | [SimCLR](celeba/simclr.md) | 1 | 2 | factors_celeba / simclr | 84.9 ± 0.0 | 87.8 ± 0.5 |
| metashift | [Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks)](metashift/concept_factors.md) | 1 | 132 | factors_metashift / xfit_meaning_w5_clippca_holdout | 54.2 ± 5.4 | 65.9 ± 4.5 |
| metashift | [CoSpRo (teacher graph: sampler and relational KL)](metashift/cospro.md) | 2 | 44 | metashift_cospro / cospro_laion | 47.7 ± 4.3 | 55.5 ± 2.5 |
| metashift | [SimCLR](metashift/simclr.md) | 2 | 24 | factors_metashift / simclr_lt | 51.4 ± 1.5 | 55.3 ± 0.1 |
| spur_cifar10 | [Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks)](spur_cifar10/concept_factors.md) | 1 | 38 | factors_spur_cifar10 / spatial_w5_holdout | 38.1 ± 0.5 | 69.3 ± 1.0 |
| spur_cifar10 | [CoSpRo (teacher graph: sampler and relational KL)](spur_cifar10/cospro.md) | 1 | 1 |  | | |
| spur_cifar10 | [SimCLR](spur_cifar10/simclr.md) | 1 | 2 | factors_spur_cifar10 / simclr | 15.6 ± 1.0 | 59.5 ± 0.2 |
| waterbirds | [Concept factors (F1 conditioned batches, F2 factor distillation, F3 concept blocks)](waterbirds/concept_factors.md) | 1 | 48 | factors_waterbirds / xfit_meaning_w5_response_holdout | 54.4 ± 3.9 | 65.4 ± 0.3 |
| waterbirds | [Frozen concept transfer (direct distillation of CLIP/SpLiCE targets)](waterbirds/concept_transfer.md) | 2 | 18 | next_actions_after_transfer_2026_09_07_direct_transfer / raw_distillation | 47.7 ± 2.7* | 52.9 ± 0.1 |
| waterbirds | [CoSpRo (teacher graph: sampler and relational KL)](waterbirds/cospro.md) | 11 | 66 | crp_signal_checks_v1_transfer_lambda_0_5 / splice_crp_kl | 50.8 ± 1.1* | 54.5 ± 1.8 |
| waterbirds | [CoSpRo + LateTVG](waterbirds/cospro_latetvg.md) | 1 | 2 |  | | |
| waterbirds | [LA-SSL](waterbirds/la_ssl.md) | 1 | 4 | waterbirds_la_ssl / la_ssl | 46.4 ± 0.8 | 51.7 ± 1.2 |
| waterbirds | [LateTVG](waterbirds/latetvg.md) | 1 | 6 |  | | |
| waterbirds | [SimCLR](waterbirds/simclr.md) | 4 | 11 | crp_followup_controls_v1_seeds34 / simclr | 47.0 ± 0.5* | 53.1 ± 0.6 |
