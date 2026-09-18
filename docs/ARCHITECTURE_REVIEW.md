# Архитектурное ревью Spur_SpLiCE

Дата: 2026-09-18. Состояние: `main` @ `2da9387`.
Метод: скилл `improve-codebase-architecture` (mattpocock/skills) + оценка по SOLID.

Словарь (из скилла `codebase-design`, используется как есть):

- **module** — всё, у чего есть interface и implementation (функция, класс, пакет).
- **interface** — всё, что вызывающий обязан знать: сигнатура, инварианты, порядок вызовов, обязательная конфигурация.
- **deep / shallow** — много поведения за маленьким interface / interface почти такой же сложный, как implementation.
- **seam** — место, где поведение меняется без правки этого места.
- **adapter** — конкретная реализация, подключённая к seam. Один adapter — seam гипотетический, два — реальный.
- **leverage** — выигрыш вызывающего от depth. **locality** — изменения и баги сосредоточены в одном месте.

---

## 1. Коротко

Код **научно аккуратный, но архитектурно плоский**. Воспроизводимость, валидация артефактов,
манифесты экспериментов и run-records сделаны хорошо, лучше, чем в большинстве исследовательских репозиториев.
Модульность при этом низкая: почти всё поведение держится на одной строке `splice_mode` и одном
`argparse.Namespace`, который проходит через весь код. В итоге любое расширение (новый датасет,
новый режим обучения, новый словарь, новая ablation-ручка) требует правок в 4–10 местах в разных папках.

Ощущение «код сумбурный и его сложно править» верное, и у него три конкретные причины:

1. **Нет seam для «метода обучения».** SimCLR, CoSpRo-relational, frozen concept distillation и LA-SSL
   различаются через `if args.splice_mode == ...` и `getattr(regularizer, "requires_crp_indices")`,
   разбросанные по 6 файлам.
2. **Нет единой конфигурации.** Один и тот же гиперпараметр имеет до четырёх источников значения по умолчанию:
   argparse, манифест, `.sbatch`, `run_cospro_pipeline.sh`.
3. **Раскладка по папкам не совпадает с ролями кода.** Библиотечный код лежит в `experiments/`,
   этапы основного пайплайна — в `scripts/tools/` рядом с одноразовыми утилитами миграции,
   сторонний SpLiCE смешан с собственным методом CoSpRo в одном пакете `splice/`.

### Оценки (1 — плохо, 5 — отлично)

| Критерий | Оценка | Главная причина |
|---|:-:|---|
| **S** — Single Responsibility | 2 | `spur_splice.py` (1243 строки) — CLI, валидация, сборка данных, цикл обучения, пробы, очистка, W&B и run-record в одном файле; `splice/cospro.py` — 1528 строк, 5 ответственностей |
| **O** — Open/Closed | 1.5 | Новый режим или датасет = правка существующих `if`/`choices`/`case` в нескольких файлах |
| **L** — Liskov Substitution | 2.5 | Регуляризаторы не взаимозаменяемы: вызывающий код проверяет флаги-атрибуты и выбирает разные ветки |
| **I** — Interface Segregation | 2 | Каждая функция принимает весь `args` (~110 полей), хотя читает 3–5; проба получает поддельный `Namespace` из 35 полей |
| **D** — Dependency Inversion | 2 | Высокоуровневый `main()` создаёт конкретные `CrpRelationalRegularizer`, `wandb`, пути; адаптеры датасетов импортируют `splice.concept_distillation` |
| Тестируемость | 3 | 118 тестов, есть фикстуры; но тест пайплайна — 986 строк с `patch`, потому что interface-ы широкие |
| Воспроизводимость / провенанс | 4.5 | RNG-изоляция, fingerprint-ы, атомарная запись, `run.json`, аттестации артефактов — сильная сторона |
| Структура папок | 2 | См. раздел 4 |

---

## 2. Как устроено сейчас

```text
                     ┌─────────── scripts/*.sh, *.sbatch  (дефолты гиперпараметров №3, №4)
                     │
experiments/manifests/*.json (дефолты №2) ──> experiments/runner.py ──(строки CLI)──┐
                                                                                    ▼
                                                     spur_splice.py  (argparse, дефолты №1)
                                                       │  args: Namespace, мутируется по ходу
         ┌──────────────┬──────────────┬───────────────┼────────────────┬──────────────────┐
         ▼              ▼              ▼               ▼                ▼                  ▼
  datasets/*.py   splice/cospro_   splice/concept_  training/        linear_probe.main  splice/run_recording
  (знают про       training.py      distillation     ssl_loop.py      (получает         splice/artifacts
   splice_mode)    (граф, сэмплер,  (targets)        (ветвится по     поддельный
                   регуляризатор)                    getattr-флагам)  Namespace)

Офлайн-пайплайн учителя:
scripts/tools/cache_splice_dataset.py ─> scripts/tools/generate_cospro_concept_groups.py
  ─> scripts/tools/build_cospro_teacher_graphs.py ─> splice/cospro.py (1528 строк) ─> graph.json
```

---

## 3. SOLID: разбор с примерами

### S — Single Responsibility

**`spur_splice.py`** совмещает минимум 8 ответственностей:

| Ответственность | Где |
|---|---|
| CLI (110 аргументов) | `parse_args`, `spur_splice.py:65-286` |
| Бизнес-валидация всех режимов | `spur_splice.py:287-395` |
| Именование запусков и путей | `format_run_name`, `format_storage_name`, `spur_splice.py:432-500` |
| Сборка загрузчиков, в т.ч. логика графа учителя | `build_ssl_loader`, `spur_splice.py:581-670` |
| Сборка модели, оптимизатора, регуляризатора | `build_training_state`, `spur_splice.py:751-797` |
| Цикл обучения, периодические пробы, чекпоинты | `main`, `spur_splice.py:1090-1197` |
| Политика хранения и удаления файлов | `cleanup_*`, `spur_splice.py:881-980` |
| W&B и lifecycle run-record | `main`, `record_resolved_training_config` |

Следствие: чтобы поменять политику чекпоинтов, приходится читать код CoSpRo-графа, и наоборот.
Тест на любую из частей тянет за собой весь модуль.

**`splice/cospro.py`** (1528 строк): схема и валидация кэша, группировка концептов (disjoint-set,
лексические ключи), поиск соседей (exact и LSH), аудит интервенций с null-контролем, сборка
графа, чекпоинты по группам, CLI. `build_teacher_graph` — около 300 строк в одной функции.

**Адаптеры датасетов** отвечают за данные *и* за режимы обучения:
`make_waterbirds_ssl_loader` содержит ветку `frozen_concept_distill` (`waterbirds.py:182-196`).

### O — Open/Closed

Сценарий «добавить новый режим обучения» (например, новую ablation-версию регуляризатора) сегодня затрагивает:

1. `spur_splice.py:230` — `choices` у `--splice_mode`
2. `spur_splice.py:51` — `RELATIONAL_GRAPH_MODES`
3. `spur_splice.py:287-395` — валидация
4. `spur_splice.py:581-670` — `build_ssl_loader`
5. `spur_splice.py:751-797` — `build_training_state`
6. `spur_splice.py:799-843` — `record_resolved_training_config`
7. `ssl_loop.py:32-90` — `simclr_forward_loss`, ветки по `getattr`
8. `ssl_loop.py:93-120` — захардкоженный словарь диагностик
9. `datasets/{waterbirds,celeba,spur_cifar10}.py` — белый список `splice_mode` в каждом
10. `run_cospro_pipeline.py`, `export_wandb_runs.py` — разбор `splice_mode`

`grep splice_mode` находит 64 упоминания в 10 не-тестовых файлах.

Сценарий «добавить датасет»:

1. Скопировать ~250 строк из `celeba.py` / `waterbirds.py` — они совпадают на 80%:
   одинаковые `*_transforms`, `eval`, `make_*_loaders`, `make_*_ssl_loader`, `make_*_rank_loader`
2. `registry.py` — три словаря: canonical, aliases, legacy
3. `scripts/run_training.sbatch:63-67` — `case` для нормализации имени датасета в bash
4. `spur_splice.py:288, 356` — дефолтная модель и проверка разрешения завязаны на `spur_cifar10`
5. `run_cospro_pipeline.py:205` — та же логика ещё раз

Сценарий «добавить словарь концептов»:
`splice/splice.py` — `SUPPORTED_VOCAB`, `_vocabulary_path` (частный случай `openimages_v7`),
`_select_vocabulary_lines` (частный случай `laion`). Произвольный `.txt` подключить нельзя.

### L — Liskov Substitution

`CrpRelationalRegularizer` и `ConceptDistillationRegularizer` формально играют одну роль, но
вызываются по-разному (`ssl_loop.py:59-80`):

```python
if getattr(splice_regularizer, "requires_crp_indices", False):
    splice_loss = splice_regularizer(embeddings, sample_indices)            # сигнатура A
    ...; return loss, parts, bsz                                             # ранний выход
if getattr(splice_regularizer, "requires_concept_transfer", False):
    target_rows, valid_rows = splice_regularizer.targets_for_indices(...)   # доп. метод
    predictions = model.clip_distillation_head(embeddings)                  # знание о модели
    splice_loss = splice_regularizer(predictions, torch.cat([...]), valid_rows)  # сигнатура B
```

Один adapter не подставляется вместо другого: вызывающий код обязан знать, какой перед ним класс.
Это нарушение LSP, при котором seam формально есть, а по факту не работает.

### I — Interface Segregation

- Все функции принимают `args: argparse.Namespace`. Interface любой из них — это «весь CLI».
  По сигнатуре `train_one_epoch(..., args, ...)` не видно, что она читает `device`, `channels_last`,
  `amp`, `print_freq`, `simclr_weight` и поля warmup, которые добавляются в `args` уже после парсинга.
- `args` **мутируется** во время работы: `build_ssl_loader` записывает `args.teacher_graph_*`,
  `args.relational_graph_empty`; `main` записывает `args.run_recorder_instance`. Порядок вызовов
  становится скрытой частью interface.
- `build_linear_probe_args` (`spur_splice.py:712-749`) копирует 35 полей в новый `Namespace`,
  чтобы вызвать `linear_probe.main()`, который написан как CLI-точка входа.

### D — Dependency Inversion

- `build_training_state` сама создаёт `CrpRelationalRegularizer` / `ConceptDistillationRegularizer`
  и сама решает, подключать ли LA-SSL. Модули верхнего уровня зависят от конкретных классов.
- Датасеты импортируют `splice.concept_distillation`, то есть нижний слой зависит от метода.
- `splice/artifacts.py:19`: `DEFAULT_SCRATCH_ROOT = Path("/scratch/xar68reb/CoSpRo")`;
  `spur_splice.py:223`: `--entity gsgrechkin-rptu`; `run_training.sbatch`: `/home/xar68reb/Datasets`.
  Библиотечный код зависит от окружения конкретного пользователя.

---

## 4. Структура папок

Впечатление «разбросано странно» подтверждается. Названия папок не соответствуют тому, что в них лежит.

| Папка | Что в ней на самом деле | Проблема |
|---|---|---|
| `splice/` | (a) сторонний SpLiCE: `model.py`, `admm.py`, `splice.py`, авторы Oesterling и Bhalla; (b) ваш метод CoSpRo; (c) инфраструктура: `artifacts.py`, `run_recording.py`, `reporting.py`; (d) шимы `crp*.py` | 4 разные роли; неочевидно, какой код ваш, а какой сторонний |
| `experiments/spurious_eval/` | Ядро библиотеки: датасеты, модели, лоссы, цикл обучения, пробы, метрики | Это не «эксперименты»; импорт `experiments.spurious_eval.datasets...` читается как тестовый код |
| `experiments/` | Ещё `runner.py`, манифесты и одноразовый `complete_projection_controls.py` с захардкоженными SHA | Правильное место для манифестов и runner, но одноразовые скрипты туда не относятся |
| `scripts/tools/` | Этапы основного пайплайна: cache → groups → graph; одноразовые миграции: `migrate_outputs`, `archive_legacy`, `promote_legacy_results`, `cleanup_local_outputs`; инструменты под статью: `build_submission_figure`, `evaluate_submission_checkpoints`; 4 шима `*_crp_*` | Ключевые этапы метода лежат рядом с мусорными утилитами; нельзя понять, что из этого часть метода |
| `scripts/*.sh`, `*.sbatch` | 22 файла: launcher-ы и wrapper-ы; `build_crp_*.sh` — совместимые шимы | Хранят свои дефолты гиперпараметров |
| корень | `spur_splice.py` — главная точка входа; `export_wandb_runs.py`; `CoSpRo.tex`, `iclr2027_conference.sty`, `paper_results.json` | Статья и код перемешаны |
| `docs/` | Канонические правила (`REPO_STRUCTURE.md`) рядом с датированными заметками `SUBMISSION_*`, `PAPER_UPDATE_*` | Непонятно, какой документ актуален |
| `setup.py` | `name="splice"`, авторы — авторы оригинального SpLiCE, `py_modules=["spur_splice"]` | Пакет называется так же, как вендоренная библиотека; нет `pytest` в зависимостях, версии не закреплены |

Что сделано хорошо: `outputs/`-политика, allowlist в `.gitignore`, `schemas/`, `tests/fixtures/`.
Эти части трогать не нужно.

---

## 5. Кандидаты на углубление (deepening candidates)

Кандидаты упорядочены по отношению пользы к стоимости с учётом ваших планов: новые датасеты,
новые словари, ablation studies.

### C1. Seam «метод обучения» (`TrainingMethod`) — **Strong**

**Файлы:** `spur_splice.py`, `training/ssl_loop.py`, `splice/cospro_training.py`,
`splice/concept_distillation.py`, `training/la_ssl.py`, `datasets/*.py`.

**Проблема.** Режим обучения — строка, которая протекает через seam в 10 файлов. Регуляризаторы
не взаимозаменяемы (LSP). Каждая новая ablation-вариация метода означает хирургию в `spur_splice.py`.

**Решение.** Один deep module на каждый метод. Он сам собирает всё, что ему нужно, и отдаёт
тренеру узкий interface:

```python
class TrainingMethod(Protocol):
    name: ClassVar[str]
    Config: ClassVar[type]                          # dataclass со своими гиперпараметрами

    def wrap_loader(self, loader, dataset) -> DataLoader: ...  # граф-сэмплер, LA-SSL, targets
    def extra_loss(self, batch: SSLBatch, embeddings, model, epoch) -> LossTerms: ...
    def provenance(self) -> dict: ...                # что записать в run.json / W&B
    def input_artifacts(self) -> list[Path]: ...     # что зарегистрировать как вход

METHODS = {"simclr": SimCLROnly, "cospro_relational": CoSpRoRelational,
           "frozen_concept_distill": FrozenConceptDistill, "la_ssl": LaSSL}
```

Adapter-ов уже четыре, поэтому seam реальный, а не гипотетический.
Тренер вызывает `method.extra_loss(...)` без `getattr`-флагов. Датасеты больше не знают про `splice_mode`.

**Выгоды.**
- locality: весь CoSpRo-режим живёт в одном файле;
- новый метод или ablation-вариант — новый файл плюс строка в реестре, без правок существующего кода (OCP);
- interface тренера сужается до `extra_loss`, поэтому тесты тренера используют фиктивный метод;
- LSP восстанавливается: все методы вызываются одинаково.

### C2. Единая типизированная конфигурация — **Strong**

**Файлы:** `spur_splice.py:parse_args`, `linear_probe.py:parse_args/normalize_args`,
`experiments/runner.py:command_for`, `scripts/run_training.sbatch`, `scripts/run_cospro_pipeline.sh`,
`scripts/tools/run_cospro_pipeline.py`, `splice/cospro.py:CrpAuditConfig`.

**Проблема.** Пример: температура SimCLR задаётся как `0.5` в argparse, `0.05` в манифесте, `0.05` в `.sbatch`
и `0.05` в `run_cospro_pipeline.sh`. Похожая картина у `batch_size` (256/128) и `cospro_temperature` (0.1/0.25).
Валидация расписаний CoSpRo продублирована в `parse_args` и в `CrpRelationalRegularizer.__init__`.
Манифест превращается в строку CLI, а она обратно парсится argparse — лишний круг с потерей типов.

**Решение.** Вложенные frozen-dataclass-ы: `DataConfig`, `ModelConfig`, `SSLConfig`, `ProbeConfig`,
`RunIdentity`, `StorageConfig` и `method: <Method>.Config` из C1. Каждый класс валидирует себя
в `__post_init__`. Один загрузчик `load_config(manifest, arm, seed, overrides)` и один источник дефолтов.
CLI становится тонким adapter-ом: `--set ssl.temp=0.05` или путь к YAML/JSON. Shell-скрипты
передают только путь к манифесту и overrides и не хранят чисел.

Можно взять Hydra/OmegaConf, но для этого проекта хватит обычных dataclass-ов и небольшой функции
dotted-override. Так сохранятся текущие JSON-манифесты и `run.json`.

**Выгоды.**
- один источник правды для каждого гиперпараметра;
- ablation записывается в манифесте как `{"method.weight": 0}`, без нового флага argparse;
- ISP: функции принимают `SSLConfig`, а не весь `args`;
- мутация `args` заканчивается, порядок вызовов перестаёт быть частью interface.

**Риск.** `runner._require_same_command` сравнивает строку команды. После перехода нужна новая версия
схемы `command.json` (`experiment-command-v2`), чтобы старые незавершённые попытки не пытались резюмироваться.

### C3. Углубление адаптеров датасетов — **Strong**

**Файлы:** `datasets/waterbirds.py`, `celeba.py`, `spur_cifar10.py`, `registry.py`, `wilds_compat.py`,
`paths.py`, `scripts/run_training.sbatch`.

**Проблема.** Каждый датасет — shallow module: 250 строк, из которых уникальны примерно 40
(чтение метаданных и путь к картинке). Три загрузчика (`ssl`, `rank`, `probe`) скопированы трижды.
Спецификация задаётся словарём, а не типом, поэтому опечатка в ключе обнаружится только во время выполнения.

**Решение.** Базовый класс `SpuriousDataset` плюс декларативная спецификация:

```python
@register_dataset("waterbirds", aliases=["wb"])
class Waterbirds(SpuriousDataset):
    num_classes = 2
    group_fields = ("background", "y")
    normalization = IMAGENET
    image_size = 224
    def read_metadata(self, root) -> pd.DataFrame: ...   # колонки: path, y, spurious, split
    def load_image(self, row) -> Image: ...
```

Один generic `build_loader(dataset, role: Literal["ssl","rank","probe_train","probe_eval"], cfg)`.
Совместимость с моделью (32×32 против 224) — атрибут датасета, а не `if dataset == "spur_cifar10"` в трёх местах.
Alias-ы берутся из декоратора, `case` в bash удаляется.

**Выгоды.**
- новый датасет — около 50 строк в одном файле;
- тест базового класса один раз покрывает все датасеты;
- минус ~400 строк дублирования.

### C4. Разрезать `spur_splice.py`: тренер с callback-ами — **Strong**

**Файлы:** `spur_splice.py`, `training/ssl_loop.py`, `training/checkpointing.py`.

**Проблема.** `main()` на 220 строк переплетает цикл обучения, пробы, чекпоинты, W&B, run-record
и очистку. Логика try/except для W&B и статуса дублируется. `save_checkpoint(...)` вызывается 4 раза
с одинаковыми 8 аргументами.

**Решение.**
- `training/trainer.py`: цикл по эпохам и событиям (`on_epoch_end`, `on_train_end`, `on_failure`).
- Callback-и: `RankMetrics`, `PeriodicProbe`, `CheckpointPolicy` (вместе с retention/cleanup),
  `WandbLogger`, `RunRecordLogger`. У логирования два adapter-а (W&B и `run.json`), значит seam реальный.
- `TrainingState` (model, optimizer, scaler, loader generator, состояние метода) — один объект
  для `save_checkpoint`/`load_checkpoint` вместо 8 аргументов.
- `spur_splice.py` остаётся точкой входа примерно на 30 строк: `config → build → Trainer(...).fit()`.

**Выгоды.**
- политика хранения тестируется без обучения;
- можно запустить обучение без W&B и файловой системы, подставив in-memory callback;
- locality: удаление файлов собрано в одном модуле.

### C5. Линейная проба как библиотечная функция — **Worth exploring**

**Файлы:** `linear_probe.py` (764 строки), `training/probe_loop.py`, `training/logistic_probe.py`,
`spur_splice.py:704-749`.

**Проблема.** Тренер вызывает CLI-функцию `main()`, подсовывая ей поддельный `Namespace`.
Сохранение фич, W&B, run-record и вычисление метрик перемешаны.

**Решение.** `evaluate_probe(encoder, dataset, ProbeConfig) -> ProbeResult` без побочных эффектов
плюс отдельный `persist_probe_result(result, storage)`. CLI `linear_probe.py` становится тонким
adapter-ом над этой функцией.

**Выгоды.** Пробу можно вызывать из тренера, из `evaluate_submission_checkpoints.py` и из
будущих ablation-скриптов через один interface. Тесты проверяют `ProbeResult` на фиктивных фичах.

### C6. Пакет `cospro/` и seam «словарь концептов» — **Worth exploring**

**Файлы:** `splice/cospro.py`, `splice/splice.py`, `scripts/tools/cache_splice_dataset.py`,
`generate_cospro_concept_groups.py`, `build_cospro_teacher_graphs.py`.

**Проблема.** 1528 строк в одном module. Этапы пайплайна живут в `scripts/tools/`, хотя это ядро метода.
Словари подключаются через частные случаи по имени (`if name == "openimages_v7"`, `if name == "laion"`).

**Решение.**

```text
cospro/
  cache.py        # схема SpliceDatasetCache + validate/save/load
  dictionary.py   # ConceptDictionary: words, embeddings, provenance; реестр adapter-ов
  grouping.py     # build_concept_groups
  neighbors.py    # NeighborIndex: ExactNeighbors, LshNeighbors (2 adapter-а => реальный seam)
  audit.py        # скоринг интервенций, null-контроль
  graph.py        # build_teacher_graph (оркестрация, ~80 строк)
  graph_io.py, reporting.py
  pipeline.py     # cache -> groups -> graph как функции; CLI — adapter-ы в cli/
```

`ConceptDictionary` принимает `laion`, `openimages_v7` и **любой `.txt`/`.csv`** через
`dictionary: {kind: file, path: ..., order: frequency|file}`. Выбор подмножества (tail/head) задаётся
параметром, а не зашивается в имя.

**Выгоды.**
- ablation по словарям и порогам группировки — только конфиг;
- поиск соседей тестируется отдельно (exact против LSH на одной фикстуре);
- `build_teacher_graph` становится читаемым.

### C7. Отдельный слой совместимости `crp → cospro` — **Worth exploring (quick win)**

**Проблема.** Наследие CRP присутствует везде: `args.crp_*` (29 обращений), `CrpAuditConfig`,
`CrpRelationalRegularizer`, `build_crp_training_loader`, 4 шима в `scripts/tools/`,
3 шима в `splice/`, `build_crp_*.sh`, dest-ы argparse `--cospro_x → crp_x`. Внутренние имена
расходятся с публичными, и это усложняет чтение.

**Решение.** Все внутренние имена перевести на `cospro_*`. Чтение старых артефактов собрать в одной функции
`compat.upgrade_graph_artifact(payload) -> payload` и одной функции для старых ключей `run.json`.
Шимы удалить после одного релиза или перенести в `compat/`. Существующие артефакты не меняются.

### C8. Перекладка по папкам — **Strong, но делать последним**

Целевая структура:

```text
spur_splice/                  # единый установочный пакет (pip install -e .)
  config/                     # C2
  data/                       # C3: base.py, registry.py, transforms.py, waterbirds.py, celeba.py, spur_cifar10.py
  models/                     # resnet.py, simclr.py
  methods/                    # C1: base.py, simclr.py, cospro_relational.py, concept_distill.py, la_ssl.py
  training/                   # C4: trainer.py, callbacks.py, checkpointing.py, optim.py, losses.py
  evaluation/                 # C5: probe.py, logistic.py, metrics.py, protocol.py
  cospro/                     # C6
  infra/                      # artifacts.py, run_recording.py, html_report.py
  compat/                     # C7
  cli/                        # train.py, probe.py, cache.py, groups.py, graph.py, pipeline.py
third_party/splice/           # вендоренный SpLiCE (model.py, admm.py, loader) + NOTICE о модификациях
experiments/
  manifests/                  # как сейчас
  runner.py                   # как сейчас
slurm/                        # *.sbatch, *.sh — только ресурсы и окружение, без гиперпараметров
tools/
  maintenance/                # migrate_outputs, archive_legacy, cleanup_local_outputs, promote_*
  paper/                      # build_paper_results, submission figure/eval, render_concept_panels
paper/                        # CoSpRo.tex, iclr2027_conference.sty, paper_results.json
docs/
  REPO_STRUCTURE.md, ARCHITECTURE_REVIEW.md
  notes/                      # SUBMISSION_*, PAPER_UPDATE_* (датированные)
tests/                        # зеркалит spur_splice/
```

Правило размещения: `spur_splice/` импортируется и не имеет побочных эффектов при импорте;
`cli/`, `slurm/` и `tools/` — adapter-ы, которые ничего не считают сами.

---

## 6. Сценарии расширения: сейчас и после

| Задача | Сейчас | После C1–C3, C6 |
|---|---|---|
| Новый датасет | ~250 строк копипасты, 5 файлов, включая bash | 1 файл ~50 строк + `@register_dataset` |
| Новый словарь концептов | Правка `splice/splice.py` в 3 местах | `.txt` в `data/vocab/` + строка в конфиге |
| Новый режим обучения или регуляризатор | 10 файлов | 1 файл в `methods/` + регистрация |
| Ablation по существующей ручке, например `weight=0`, `temperature`, `start_epoch` | Новый arm в манифесте (уже работает) | То же, плюс любая вложенная ручка через dotted-key |
| Новая ablation-ручка внутри метода | argparse + валидация + проброс через `args` + регуляризатор | Поле в `Method.Config` |
| Сетка ablation (grid) | Руками выписать все arms | `"sweep": {"method.weight": [0, 0.25, 0.5]}` → runner раскрывает матрицу |

---

## 7. План миграции (без потери воспроизводимости)

Главный риск — сломать воспроизводимость уже опубликованных чисел. Поэтому сначала страховочная сеть,
потом рефакторинг небольшими шагами, и каждый шаг проверяется одними и теми же golden-тестами.

**Фаза 0 — страховочная сеть (1–2 дня).**
- Добавить `pytest` в dev-зависимости, в `pyproject.toml` перенести метаданные проекта из `setup.py`,
  закрепить версии (`requirements.lock` или `uv.lock`). Сейчас в локальном `Python 3.14` нет `pytest`,
  и тесты не запускаются.
- Golden-тест «команды не изменились»: `runner.command_for` для всех манифестов, всех seed и arm → snapshot.
- Golden-тест «численное поведение не изменилось»: 2 эпохи на синтетическом датасете 16×32×32 на CPU
  для каждого `splice_mode`; сравнение loss и метрик с сохранёнными значениями при фиксированном seed.
- Golden-тест графа: `build_teacher_graph` на фикстуре → `graph_fingerprint`.
- GitHub Actions: pytest на CPU.

**Фаза 1 — быстрые победы (≤1 день каждая).**
- Вынести пути и идентификаторы пользователя (`/scratch/xar68reb`, `gsgrechkin-rptu`,
  `/home/xar68reb/Datasets`) в переменные окружения или профиль `slurm/env.sh`.
- Убрать из `.sbatch` и `.sh` дублирующиеся дефолты гиперпараметров: оставить манифест единственным источником.
- Перенести `CoSpRo.tex`, `.sty` и `paper_results.json` в `paper/`, заметки — в `docs/notes/`.
- Разделить `scripts/tools/` на `tools/maintenance/` и `tools/paper/`, а этапы пайплайна отложить до C6.
- Переименовать внутренние `crp_*` в `cospro_*` (C7).

**Фаза 2 — C2 (конфиг)** вместе с `command.json v2`.
**Фаза 3 — C1 (методы).** После неё датасеты перестают знать про `splice_mode`.
**Фаза 4 — C3 (датасеты).**
**Фаза 5 — C4 и C5 (тренер, проба).**
**Фаза 6 — C6 (`cospro/`, словари).**
**Фаза 7 — C8 (финальная перекладка пакетов, `git mv` + временные re-export шимы на один релиз).**

Фазы 2–7 лучше проводить **между исследованиями**, а не во время незавершённой матрицы запусков:
`runner` резюмирует попытки по совпадению команды.

---

## 8. Мелкие замечания

- `spur_splice.py:1108` — `log_metrics = True`, следующее `if log_metrics or log_rank` всегда истинно.
- `spur_splice.py:390-393` — проверки `is None` для полей, у которых в argparse не-`None` дефолты; это мёртвый код.
- `spur_splice.py:362` — `--cudnn_benchmark` существует только ради того, чтобы запретить значение `true`.
- `spur_splice.py:621-626` — список допустимых артефактов графа продублирован с `TEACHER_GRAPH_ARTIFACTS`
  в `cospro_training.py:23`.
- `linear_probe.py` и `spur_splice.py` имеют каждый свои `set_seed`, `seed_worker`,
  `make_dataloader_kwargs` и `resolve_*_schedule`. Их стоит вынести в `training/reproducibility.py`.
- `CrpAuditConfig.indegree_factor` помечен как legacy fallback, но всё ещё является полем конфига.
- `LEGACY_DISABLED_CONFIG` (cobalt, spatial_balance) в `cospro_training.py` — мёртвые ветки прошлых экспериментов.
- `experiments/complete_projection_controls.py` — одноразовый скрипт с захардкоженными SHA: место в `tools/paper/`.
- `wilds_compat.py` (381 строка) — вендоренная часть WILDS; её тоже стоит пометить как third-party.
- `setup.py` указывает авторов оригинального SpLiCE и имя пакета `splice`, что конфликтует с upstream-пакетом.
- Смешанный стиль CLI: `spur_splice.py` использует `--snake_case`, `cache_splice_dataset.py` и `cospro.py` — `--kebab-case`.

---

## 9. Что сохранить как есть

- Манифесты и `experiments/runner.py`. Это уже правильный seam для ablation studies; его нужно развивать, а не переписывать.
- `splice/artifacts.py`, `splice/run_recording.py`, `schemas/run-record-v1.schema.json`, политика `outputs/`.
- Защита от утечки меток (`FORBIDDEN_CACHE_KEYS`, `FORBIDDEN_ANNOTATION_KEYS`) и fingerprint-ы графов.
- Изоляция RNG (`preserve_rng_state`) и детерминизм загрузчиков.
- Отдельные этапы cache → groups → graph с сериализованными промежуточными артефактами.

---

## 10. Главная рекомендация

**Начать с C1 (seam `TrainingMethod`) вместе с минимальной частью C2**: конфиг метода как dataclass.
Именно `splice_mode` сильнее всего мешает делать ablation studies, и именно отсюда режимы протекают
в датасеты, тренер и CLI. После C1 датасеты освобождаются от знания о методе, и C3 становится механической работой.
Перед этим обязательно выполнить фазу 0 (golden-тесты), иначе рефакторинг нельзя проверить на сохранение чисел из статьи.
