# Запуск экспериментов

## Обучение на Slurm

Для обучения существует одна точка входа:

```bash
sbatch scripts/train.sbatch
```

Все гиперпараметры находятся в верхнем блоке
[`train.sbatch`](train.sbatch) между комментариями `EDIT ONLY THIS CONFIGURATION
BLOCK` и `IMPLEMENTATION BELOW`. Для обычного эксперимента код ниже этого блока
менять не нужно.

Главный переключатель — `MODE`:

| `MODE` | Что запускается |
|---|---|
| `none` | обычный SimCLR baseline |
| `crp_relational` | label-free CRP relational distillation |
| `augment` | targeted augmentation |
| `corr_reg` | корреляционная регуляризация SpLiCE |
| `augment_corr_reg` | augmentation и корреляционная регуляризация вместе |

Пример следующего CRP-эксперимента уже выставлен по умолчанию:

```bash
MODE="crp_relational"
EPOCHS="500"
SPLICE_WEIGHT="0.05"
CRP_TEMPERATURE="0.1"
CRP_START_EPOCH="10"
CRP_WARMUP_EPOCHS="10"
```

Чтобы запустить baseline, достаточно изменить одну строку:

```bash
MODE="none"
```

Для CRP/CQT с ImageNet-предобученной ResNet-50 оставьте нужный `MODE` и
измените общий encoder:

```bash
MODEL="resnet50_pretrained"
```

При первом запуске torchvision автоматически скачает веса
`ResNet50_Weights.IMAGENET1K_V2` в PyTorch cache. Encoder использует тот же
ImageNet stem (`7x7`, stride 2, max-pool) и те же 224x224 transforms, что и
`resnet18_large`; во время SSL все его веса продолжают обучаться. Для
контроля той же архитектуры без предобучения используйте
`MODEL="resnet50_large"`.

В конфигурации сначала расположены наиболее важные параметры текущего CRPv2,
затем отдельный legacy-блок для `augment/corr_reg/augment_corr_reg`. Общие,
evaluation- и инфраструктурные настройки находятся ниже. `EPOCHS` по умолчанию
равен 500, но может быть свободно изменён.

Имена W&B runs и checkpoint-папок строятся автоматически из режима, seed и
ключевых гиперпараметров.

Основные значения видны прямо в W&B run name. Например, CRPv2-конфигурация
выше получит имя:

```text
waterbirds_S0_resnet18_large_CRP_splWei_0.05_crpTemp_0.1_lr_0.01_e500
```

Для legacy-режимов имя аналогично содержит `q`, routing, `splWei` и `lr`, если
они применимы к выбранному режиму.

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

После появления `CRP_TEACHER_GRAPH` всё дальнейшее обучение снова запускается
только через `train.sbatch`.

## Локальное обучение на Windows

Windows runner оставлен отдельно, потому что PowerShell не использует Slurm:

```powershell
.\scripts\Test-HomeTraining.ps1 -DataFolder "D:\Datasets\waterbirds"
.\scripts\Run-HomeExperiments.ps1 -Family routing -Seeds 0 -Tasks 0,1,2
```

Он не влияет на кластерную конфигурацию в `train.sbatch`.
