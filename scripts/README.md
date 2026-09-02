# Запуск экспериментов

## Обучение на Slurm

Для CRP и CQT используются отдельные точки входа:

```bash
sbatch scripts/train_crp.sbatch
sbatch scripts/train_cqt.sbatch
```

Настройки вынесены в редактируемые `.conf`-файлы рядом с launcher-ами:

- [`train_crp.conf`](train_crp.conf) — CRP training и frozen-audit параметры;
- [`train_cqt.conf`](train_cqt.conf) — CQT training и factor/transport параметры;
- [`prepare_crp_group_sweep.conf`](prepare_crp_group_sweep.conf) — CRP sweep matrix;
- [`prepare_cqt_graph_sweep.conf`](prepare_cqt_graph_sweep.conf) — CQT sweep matrix;
- [`train_crp_seed0_sweep.conf`](train_crp_seed0_sweep.conf) и
  [`train_cqt_seed0_sweep.conf`](train_cqt_seed0_sweep.conf) — training sweeps;
- [`prepare_concepts.conf`](prepare_concepts.conf) — CoBalT discovery;
- [`cache_openimages_crp.conf`](cache_openimages_crp.conf) — frozen feature cache.

После изменения конфигурации обычный запуск выглядит как `sbatch` без длинного
списка параметров. Для одноразовой альтернативы можно передать `CONFIG_FILE`,
но основной workflow этого не требует:

Каталог `scripts/tools/` намеренно содержит Python-модули. Это не старые
launcher-ы: cache, graph audit, summaries и HTML reports вызывают их напрямую
через `python -m scripts.tools...`. Удалять эти файлы с кластера нельзя, если
нужны соответствующие preparation/diagnostic jobs.

Запуск CRP:

```bash
sbatch scripts/train_crp.sbatch
```

Запуск CQT:

```bash
sbatch scripts/train_cqt.sbatch
```

### DINO-абляция и CoBalT concept check

Оба training entry point поддерживают два независимых переключателя:

```bash
# edit USE_DINO=false in train_crp.conf or train_cqt.conf
sbatch scripts/train_crp.sbatch
sbatch scripts/train_cqt.sbatch
```

При `USE_DINO=false` DINO-модель не загружается, её признаки не кешируются, а
DINO gate в CRP/CQT отключается. CRPv3 в этом режиме сохраняет reciprocal
projected-neighbour check, добавляет residual SpLiCE semantic gate и cross-fold
edge validation. По умолчанию граф ограничен `top_k=3` и
`max_indegree=10`; эти значения задаются в `train_crp.conf`. No-DINO cache и
graphs пишутся в отдельные пути, поэтому обычные результаты не перезаписываются.

Для CoBalT-проверки сначала один раз обучите label-free discovery stage и
выгрузите фиксированные concept memberships:

```bash
sbatch CoBalT/scripts/prepare_concepts.sbatch
# edit COBALT=true in train_crp.conf or train_cqt.conf
sbatch scripts/train_crp.sbatch
sbatch scripts/train_cqt.sbatch
```

`COBALT=true` балансирует частоты и coactivation при составлении SpLiCE concept
groups весами, полученными только из CoBalT memberships. Target label и
spurious attribute для этого не читаются. Это label-free concept-balance check,
а не supervised classifier-balancing stage из статьи. Путь к другому артефакту
задаётся в `COBALT_CONCEPTS_PATH` соответствующего `.conf`-файла.
При `COBALT=true` teacher graph пересобирается из текущего concept artifact,
даже если файл графа уже существует; frozen SpLiCE cache при этом переиспользуется.

При прямом Python-запуске доступны запрошенные CLI-формы
`--use_dino true|false`, `--cobalt true|false` и
`--cobalt-concepts /path/to/concepts.pt`.

В CRPv3 CoBalT concept artifacts, созданные заново через
`CoBalT/scripts/prepare_concepts.sbatch`, также содержат label-free confidence
из spatial slot separation. Он включён параметром
`CRP_USE_COBALT_CONFIDENCE=true`; для старых artifacts без этого поля будет
использован нейтральный confidence и это отмечается в graph metadata.

Для CRP/CQT с ImageNet-предобученной ResNet-50 измените `MODEL` в нужном
`.conf`-файле:

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

### Seed-0 CRP/CQT sweeps

Сначала строятся label-free graph variants; этот шаг не обучает ResNet и не
выбирает конфигурацию по downstream WGA:

```bash
sbatch scripts/prepare_crp_group_sweep.sbatch
sbatch scripts/prepare_cqt_graph_sweep.sbatch
```

После проверки слов, размера групп, coverage, null margin и hubness запускаются
полные W&B-tracked обучения:

```bash
sbatch scripts/train_crp_seed0_sweep.sbatch
sbatch scripts/train_cqt_seed0_sweep.sbatch
```

CRP sweep по умолчанию ожидает graph variant `g2_t070_c020_k12`; его можно
изменить в `train_crp_seed0_sweep.conf`. CQT sweep сначала отделяет
эффект graph-aware sampler, graph positives и KL на одном и том же graph, затем
сравнивает два более широких frozen graph variants.

Основные значения видны прямо в W&B run name. Например, CRPv3-конфигурация
выше получит имя:

```text
waterbirds_S0_resnet18_large_CRP_precision_simclrWei_1.0_splWei_0.01_crpTemp_0.25_graphPos_true_decay_200-350_lr_0.01_e500
```

## Sanity-check конкретных изображений

Самодостаточный HTML-аудит с четырьмя типичными парами, выбранными concept
groups/factors, `cosine before / after / delta`, агрегатной сводкой графа и
реальными retained teacher edges создаётся так:

```bash
sbatch scripts/concept_ablation_examples.sbatch
```

Четыре пары покрывают: одинаковый target при разных backgrounds, разные targets
при одинаковом background, одинаковые target/background и разные
target/background. Для каждого типа выбирается пара с исходным CLIP cosine,
ближайшим к медиане всех допустимых пар; результат interventions на выбор не
влияет. По умолчанию показываются не более 12 выбранных interventions и одно
типичное retained edge на группу. `REPORT_MAX_INTERVENTIONS=0` показывает все
eligible interventions, а `REPORT_EDGES_PER_GROUP=0` скрывает edge examples.

Graph path автоматически включает все параметры frozen audit. Необязательный
`RELATIONAL_GRAPH_VARIANT` служит только читаемой меткой: полная конфигурация всё
равно добавляется ниже неё, поэтому другое содержимое не перезапишет граф.
`graph_audit.html` сохраняется рядом с соответствующим `teacher_graph.json`.
Для CQT:

```bash
sbatch --export=ALL,REPORT_METHOD=cqt scripts/concept_ablation_examples.sbatch
```

Скрытые Waterbirds labels/background используются только в этом post-hoc
diagnostic для выбора, подписи четырёх пар и aggregate edge metrics. Frozen cache и teacher graph строятся
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
| `cache_openimages_crp.sbatch` | один раз кеширует frozen признаки с Open Images V7 vocabulary |
| `SpLiCE_CRP_v2_frozen_audit.sbatch` | строит и аудирует teacher graph |
| `SpLiCE_CRP_v2_report.sbatch` | формирует label-free go/no-go отчёт |
| `SpLiCE_CRP_v2_posthoc_waterbirds.sbatch` | post-hoc проверка по скрытым Waterbirds labels |
| `SpLiCE_CRP_v2_baseline_graphs.sbatch` | строит raw-CLIP/DINO baseline graphs |
| `SpLiCE_CRP_v2_baselines_posthoc.sbatch` | сравнивает CRP и baseline graphs |
| `SpLiCE_CRP_v2_baseline_compare.sbatch` | объединённый baseline diagnostic job |
| `concept_ablation_examples.sbatch` | создаёт один HTML с изображениями и cosine ablations |
| `../CoBalT/scripts/prepare_concepts.sbatch` | обучает label-free CoBalT discovery и сохраняет memberships для `COBALT=true` |

CRP-обучение запускается через `train_crp.sbatch`, CQT — через
`train_cqt.sbatch`; оба entry point умеют автоматически подготовить graph.

Все обучения запускаются на кластере. Локальные PowerShell-launcher-ы удалены;
для cluster-only workflow используются только Slurm entry points и конфиги выше.
