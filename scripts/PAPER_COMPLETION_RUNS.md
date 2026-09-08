# Поэтапные запуски для статьи CoSpRo — 2026-09-08

Запускать из корня репозитория **на кластере**, после переноса новых файлов.
Скрипты подготовлены; из Codex задания не отправлялись.

## Один запуск для всей очереди

```bash
cd /home/xar68reb/Spur_SpLiCE
mkdir -p logs
sbatch scripts/paper_all.sbatch
```

`paper_all.sbatch` ставит весь pipeline. `00_inventory` → `01_prepare` →
`02_direct` и `03_graph` (параллельно) → `07_summary`. Затем dispatcher читает
получившийся список отсутствующих core-моделей: если он непустой, отправляет только
эти элементы `04_missing_core`, после чего автоматически ставит `05_lock_test` →
`06_test` → `08_test_summary`. Если список пуст, 04 пропускается.

Все зависимости — `afterok`: ошибка на любом этапе не запускает зависимый этап.
Начальные job IDs находятся в
`outputs/paper_completion_2026-09-08/submitted_all_JOBID.json`; IDs условной
core/test-цепочки — в `submitted_test_chain_JOBID.json`, созданном dispatcher'ом.
Открытие held-out test теперь входит в этот единый запуск по вашему запросу.

Основа: существующие `run_crp_controls_cluster_array.sbatch`,
`run_next_actions_after_transfer_2026-09-07.sbatch` и их Python runners.
Ресурсы на **одну задачу**: максимум 40G RAM, 5 CPU, 4 DataLoader workers,
1 V100 для SSL/test. Array ограничен четырьмя одновременно выполняемыми задачами.
Две одновременно запущенные серии могут вместе занять восемь GPU; это не общий
лимит 40G на все задания. Можно уменьшить параллелизм через `--array=0-7%2`.

Все SLURM `.out/.err` лежат в **корневой `logs/`**. Создайте её **до sbatch**:
SLURM открывает логи раньше, чем выполняет тело скрипта.

```bash
cd /home/xar68reb/Spur_SpLiCE
mkdir -p logs
```

По умолчанию используются partition `informatik-mind`, GPU `v100`, conda env
`grgrie-train`, данные `/home/xar68reb/Datasets`. При необходимости partition/GPU/time
переопределяются флагами `sbatch`; окружение — `SPLICE_CONDA_ENV`.
Core/probe поддерживают `DATA_FOLDER`; direct/graph используют путь в существующих
`next_actions_*_2026-09-07.conf`. Пути к старым результатам ожидаются под текущим
корнем проекта, с исходными именами каталогов.

W&B включён у всех SSL по умолчанию: проект `Spur_SpLiCE`, entity
`gsgrechkin-rptu`, `WANDB_MODE=online`. Используется существующая авторизация W&B
в conda-окружении. Ключи в скрипты не вставлять. Короткие CPU проверки и test-probe
сохраняют локальные JSON; отдельные W&B runs для них не создаются.

## Что действительно нужно

Чеклист считает основной Waterbirds validation core (5 arms × seeds 1–4) завершённым.
Это не повод повторять 20 обучений. Старые direct temp=.5 также не повторяются.
На скриншоте есть corrected direct temp=.05 и graph ablation; их имена/зелёные точки
не доказывают наличие финальных файлов. Этапы 02/03 проверяют локальные completed,
команду и финальный converged probe и пропускают завершённые задачи.

Недописанные таблицы/рисунки не создают дополнительных SSL задач. Randomized graph,
второй датасет и spatial — дополнительные исследования, не входят в этот минимальный
набор. В частности, незаполненная строка randomized graph в статье не трактуется
как безусловное требование запустить её. Текущие direct/semantic серии сохранены
отдельно от основной пятиветочной test-матрицы.

## Ручной запуск отдельных этапов

Команды ниже оставлены для повторного запуска или диагностики отдельного этапа.

### 00. Инвентаризация моделей — без SSL

```bash
sbatch scripts/paper_00_inventory.sbatch
```

Результаты: `outputs/paper_completion_2026-09-08/inventory.json` и
`missing_core_tasks.txt`. Проверяется содержимое `last.pth` / `epoch_500.pth`,
а не только имя. Для каждого исторического run перечисляются доступные сведения
W&B из `run_status.json`. Поиск ограничен исходными каталогами нужных arms/seeds
и новым каталогом восстановления; произвольные похожие модели не подмешиваются.

**Отсутствует локально ≠ отсутствует в W&B.** Проверьте model artifacts по этим
run IDs и другим сохранённым W&B записям, восстановите checkpoint вместе с его
`args.json` в исходный каталог, затем повторите 00. Скрипт не обращается к W&B API
и не скачивает artifacts. Если в исходном каталоге несколько независимых моделей,
он остановится, чтобы выбор не зависел от порядка файлов.

### 01–03. Завершение текущих direct/graph серий

Можно поставить подготовку и обе зависимые серии одной группой команд:

```bash
prep=$(sbatch --parsable scripts/paper_01_prepare.sbatch); prep=${prep%%;*}
direct=$(sbatch --parsable --dependency=afterok:$prep scripts/paper_02_direct.sbatch); direct=${direct%%;*}
graph=$(sbatch --parsable --dependency=afterok:$prep scripts/paper_03_graph.sbatch); graph=${graph%%;*}
sbatch --dependency=afterok:$direct:$graph scripts/paper_07_summary.sbatch
```

01 проверяет исходные fingerprints cache/CRP/raw, готовит отсутствующие direct targets
и semantic graph и выполняет существующий smoke. Исходные cache/CRP/raw **не строятся
заново**: при их отсутствии нужно восстановить оригинальные artifacts.

02: максимум 8 × 500 эпох: matched SimCLR, raw distillation, SpLiCE reconstruction,
shuffled reconstruction, seeds 1/3, temp=.05, alpha=.1 у transfer arms.

03: максимум 4 × 500 эпох: CRP и semantic SpLiCE graph, seeds 1/3, lambda=.5.
Оба этапа используют прежние конфиги, выходные каталоги и W&B groups.

| Array ID | 02 direct | 03 graph |
|---|---|---|
| 0 | seed1 matched_simclr | seed1 crp |
| 1 | seed1 raw_distillation | seed1 semantic_splice |
| 2 | seed1 splice_reconstruction | seed3 crp |
| 3 | seed1 shuffled_reconstruction | seed3 semantic_splice |
| 4–7 | те же четыре arms, seed3 | — |

Не запускайте их одновременно с уже работающим старым launcher этих же серий:
сначала проверьте `squeue -u "$USER"`. При существующем незавершённом training-каталоге
новый runner останавливается, чтобы не перезаписать/дублировать его. Автоматического
resume частичного обучения в этом наборе нет: сначала восстановите результат или
переместите конкретный неуспешный run в архив, затем повторите только его array ID.

### 04. Только утраченные core-модели

После проверки кластера/W&B и повторного этапа 00:

```bash
missing=$(cat outputs/paper_completion_2026-09-08/missing_core_tasks.txt)
if [[ -n "$missing" ]]; then
  sbatch --array="${missing}%4" scripts/paper_04_missing_core.sbatch
fi
```

Это новые воспроизведения для получения финальных моделей, а не новые независимые
seeds и не замена исторических validation результатов. Предел — 20 × 500 эпох,
фактически только отсутствующие модели. Используется `training_command` исходного
control runner, lambda=.5 у двух KL arms, исходные графы и seeds 1–4.
Сохраняются `last.pth`, RNG/optimizer state, `args.json`, `command.json` и probe JSON;
старые epoch checkpoints очищаются. W&B group:
`paper_completion_core_reproduction_2026-09-08`.

ID = `(seed - 1) * 5 + arm_index`, где arm_index:
0 SimCLR, 1 CRP sampler-only, 2 raw-CLIP sampler-only,
3 raw-CLIP KL=.5, 4 CoSpRo KL=.5.
Например, `--array=0,4,9%3` — seed1 SimCLR, seed1 CoSpRo, seed2 CoSpRo.

### 05–06–08. Фиксация и held-out test

После завершения восстановления (или сразу, если найдены все 20 моделей):

```bash
lock=$(sbatch --parsable scripts/paper_05_lock_test.sbatch); lock=${lock%%;*}
testjob=$(sbatch --parsable --dependency=afterok:$lock scripts/paper_06_test.sbatch); testjob=${testjob%%;*}
sbatch --dependency=afterok:$testjob scripts/paper_08_test_summary.sbatch
```

05 проверяет настройки каждой модели и сохраняет `final_test_lock.json`: модели,
SHA-256, args, графы/cache, revision и правило полного отчёта. Изменившийся lock
не перезаписывается. Убедитесь, что текущие validation серии уже закрыты или исключены
из выводов до запуска этой группы команд. В lock они явно исключены из core test.

06 запускает 20 независимых **linear probes**, без SSL: epoch500, supervised ds_train,
seed соответствующей модели, logistic L2=.001, tolerance=1e-6, max_epochs=200,
test split. Успешные существующие test JSON повторно не вычисляются. Для имеющегося
probe runner ключи метрик называются `val` даже при `eval_split=test`; итоговый
сборщик проверяет фактический `eval_split` и подписывает таблицу как test.
После открытия test не менять lambda/seed/эпоху по его результатам.

## Где искать результаты

| Результат | Каталог/файл от корня репозитория |
|---|---|
| Все SLURM `.out/.err` | `logs/paper-*.out`, `logs/paper-*.err` |
| Direct серии | `outputs/next_actions_after_transfer_2026-09-07/direct_transfer/` |
| Graph серии | `outputs/next_actions_after_transfer_2026-09-07/graph_ablation/` |
| Их сводки после 07 | `results.csv`, `summary.json` в соответствующих каталогах |
| Инвентаризация, preflight, lock | `outputs/paper_completion_2026-09-08/` |
| Новые core модели | `outputs/paper_completion_2026-09-08/core/seedN/ARM/training/*/last.pth` |
| Индивидуальные test probes | `outputs/paper_completion_2026-09-08/test/seedN/ARM/` |
| Все 20 test результатов | `outputs/paper_completion_2026-09-08/test_results.csv` |
| Групповые test метрики | `outputs/paper_completion_2026-09-08/test_results.json` |
| Mean/SD и paired deltas | `outputs/paper_completion_2026-09-08/test_summary.json` |

Статус: `squeue -u "$USER"`; завершение/ресурсы:
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS`.
`afterok` не продолжает цепочку после ошибки: сначала прочитайте соответствующий
`.err/.out`, затем повторите неуспешный элемент и нужные последующие этапы.

## Local artifact validation

Исторические fingerprints теперь служат только provenance и не блокируют запуск.
Для read-only диагностики используйте:

```bash
sbatch scripts/paper_00_diagnose_artifacts.sbatch
```

Отчёт: `outputs/paper_completion_2026-09-08/artifact_diagnosis.json`. Он содержит
ожидаемый и фактический BLAKE2b fingerprint, размер/время файла, значения из
`cache_identity.json` / `graph_identity.json` и совпадение `sample_ids` cache/graph.
Запуск блокируется только если отсутствует файл, cache не загружается, graph не
проходит валидацию или его `sample_ids` не совпадают с cache. Фактические identities
сохраняются в diagnosis и final manifest; результаты с отличающимся cache не следует
называть byte-identical исторической репликацией.
