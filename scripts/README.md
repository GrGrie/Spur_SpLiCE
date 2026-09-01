# Запуск экспериментов

## Обучение на Slurm

Для CRP и CQT используются отдельные точки входа:

```bash
sbatch scripts/train_crp.sbatch
sbatch scripts/train_cqt.sbatch
```

В [`train_crp.sbatch`](train_crp.sbatch) наверху сгруппированы dataset, основные
SSL/CRP параметры и CRP frozen-audit параметры. В
[`train_cqt.sbatch`](train_cqt.sbatch) в таком же порядке находятся dataset,
основные SSL/CQT параметры и все CQT factor/transport параметры. Переключать
`MODE` вручную между CRP и CQT больше не требуется.

Запуск CRP:

```bash
sbatch scripts/train_crp.sbatch
```

Запуск CQT:

```bash
sbatch scripts/train_cqt.sbatch
```

Для CRP/CQT с ImageNet-предобученной ResNet-50 измените `MODEL` в нужном
скрипте:

```bash
MODEL="resnet50_pretrained"
```

При первом запуске torchvision автоматически скачает веса
`ResNet50_Weights.IMAGENET1K_V2` в PyTorch cache. Encoder использует тот же
ImageNet stem (`7x7`, stride 2, max-pool) и те же 224x224 transforms, что и
`resnet18_large`; во время SSL все его веса продолжают обучаться. Для
контроля той же архитектуры без предобучения используйте
`MODEL="resnet50_large"`.

В обоих скриптах `EPOCHS` по умолчанию равен 500, а лимит Slurm job — 8 часов.
Общие evaluation- и инфраструктурные настройки находятся ниже основных knobs.

Имена W&B runs и checkpoint-папок строятся автоматически из режима, seed и
ключевых гиперпараметров.

Основные значения видны прямо в W&B run name. Например, CRPv2-конфигурация
выше получит имя:

```text
waterbirds_S0_resnet18_large_CRP_precision_simclrWei_1.0_splWei_0.01_crpTemp_0.25_graphPos_true_decay_200-350_lr_0.01_e500
```

## Sanity-check конкретных изображений

Один самодостаточный HTML с двумя требуемыми Waterbirds-парами, найденными
concept groups/factors и всеми `cosine before / after / delta` создаётся так:

```bash
sbatch scripts/concept_ablation_examples.sbatch
```

По умолчанию отчёт использует выбранные CRP groups и сохраняется в
`outputs/diagnostics/concept_ablation_crp_seed0.html`. Для CQT:

```bash
sbatch --export=ALL,REPORT_METHOD=cqt scripts/concept_ablation_examples.sbatch
```

Скрытые Waterbirds labels/background используются только в этом post-hoc
diagnostic для выбора и подписи двух пар. Frozen cache и teacher graph строятся
тем же preparation path, что и у соответствующего training entry point.

## KL-only ablation

Обучение backbone только confidence-weighted relational KL, с нулевым весом
SimCLR/NT-Xent и штатным downstream linear classifier:

```bash
sbatch scripts/train_kl_only.sbatch
```

По умолчанию это CRP; для CQT:

```bash
sbatch --export=ALL,KL_METHOD=cqt scripts/train_kl_only.sbatch
```

Запуск включает W&B, начинает KL с первой эпохи, не применяет поздний decay и
не включает graph positives. Пустой teacher graph считается ошибкой, поскольку
при нулевом SimCLR он оставил бы модель без обучающего сигнала. Avg Accuracy и
Worst-Group Accuracy вычисляются существующим periodic linear-probe pipeline.

## CRP-подготовка и диагностика

Оставшиеся `.sbatch`-файлы не обучают ResNet и не содержат альтернативных
training-гиперпараметров. Они нужны только для построения и проверки CRP teacher
graph:

| Скрипт | Назначение |
|---|---|
| `SpLiCE_CRP_v2_cache_features.sbatch` | один раз кеширует frozen признаки |
| `SpLiCE_CRP_v2_frozen_audit.sbatch` | строит и аудирует teacher graph |
| `SpLiCE_CRP_v2_report.sbatch` | формирует label-free go/no-go отчёт |
| `SpLiCE_CRP_v2_posthoc_waterbirds.sbatch` | post-hoc проверка по скрытым Waterbirds labels |
| `SpLiCE_CRP_v2_baseline_graphs.sbatch` | строит raw-CLIP/DINO baseline graphs |
| `SpLiCE_CRP_v2_baselines_posthoc.sbatch` | сравнивает CRP и baseline graphs |
| `SpLiCE_CRP_v2_baseline_compare.sbatch` | объединённый baseline diagnostic job |
| `concept_ablation_examples.sbatch` | создаёт один HTML с изображениями и cosine ablations |

CRP-обучение запускается через `train_crp.sbatch`, CQT — через
`train_cqt.sbatch`; оба entry point умеют автоматически подготовить graph.

## Локальное обучение на Windows

Windows runner оставлен отдельно, потому что PowerShell не использует Slurm:

```powershell
.\scripts\Test-HomeTraining.ps1 -DataFolder "D:\Datasets\waterbirds"
.\scripts\Run-HomeExperiments.ps1 -Family routing -Seeds 0 -Tasks 0,1,2
```

Он не влияет на кластерную конфигурацию в `train_crp.sbatch` или
`train_cqt.sbatch`.
