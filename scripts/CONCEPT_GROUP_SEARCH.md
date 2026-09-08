# Составные concept groups: гипотеза, визуальная проверка и конечный поиск

## Вывод до эксперимента

Идея обоснована, но пока это гипотеза, а не обнаруженный источник прироста Avg/WGA.
В текущем `splice/crp.py::group_concepts` обычное объединение требует одновременно
text cosine >=.8 и coactivation cosine >=.35 (плюс отдельное lexical объединение).
Транзитивное замыкание может образовывать цепочки, хотя крайние слова уже не близки.
После аудита выбираются до 12 групп по превышению над construction-null.
Из статьи/чеклиста следует, что текущие выбранные группы — singleton, включая
Bamboo, Great grey owl, Tawny owl. Сам по себе singleton не означает ошибку.

У разреженного словаря два близких слова могут конкурировать за реконструкцию:
`lake` активно на одних изображениях, `pond` — на других. Тогда coactivation низка,
хотя текстовая близость высока. Это **возможное объяснение**, которое новый поиск
проверяет. Оно не доказывает, что конкретно эти слова встречаются в исходном cache.

Группа `{water, sea, lake, pond}` не задаётся вручную в teacher: это внесло бы
наше знание о Waterbirds. `candidates.json` отдельно сообщает наличие точных слов
в vocabulary и частоту на discovery-половине. Этот diagnostic не участвует в рейтинге.
Отсутствие сочетания может объясняться vocabulary, фильтром частоты, coactivation,
null-отбором или последующим cap — эти причины нужно различать.

SpLiCE использует разреженную неотрицательную реконструкцию, поэтому гипотеза о
конкурирующих близких словах совместима с устройством представления, но не следует
из него как гарантированный эффект. [Статья SpLiCE](https://arxiv.org/abs/2402.10376),
[официальная реализация](https://github.com/AI4LIFE-GROUP/SpLiCE).

## Какие визуальные результаты будут содержательными

1. **Вес → общность изображений.** По всем retained edges строятся пять равных по
   числу рёбер bins по `anchor_confidence * row_weight`. Для каждого: same target,
   same background, same target / different background и случайная reference.
   Reference — 100 перестановок пар `(target, background)` между training images,
   seed 913, с неизменными graph/weights. При дисбалансе классов chance не равен .5.
   Интервал перестановок — reference-распределение, не CI независимых рёбер.

2. **Слепая оценка пар.** До просмотра изображений выбираются до 12 пар из low,
   middle и high deciles массы, плюс до 12 random-donor пар с теми же anchors, что
   high. В `blind_pairs.html` нет весов/названия графа. Сначала заполнить
   `ratings.csv`: общий объект/категория и общий фон отдельно, 0/1/2, заметка при
   неоднозначности. Желательно два оценщика независимо, с отдельными копиями CSV.
   Только после разметки открыть `pair_key.json`.
   Заранее интересующий контраст: **high − matched random по общности объекта**;
   secondary: high − low и изменение общности фона. Положительная разница только
   по фону может свидетельствовать о сохранении spurious связи. Сборщик возвращает
   разницу и bootstrap по anchors, не по зависимым рёбрам.

3. **Concept group → активирующие изображения.** До пяти выбранных групп,
   сначала составные, затем по null-excess, плюс одна лучшая отклонённая группа.
   Показываются четыре top-activating изображения с вкладом **каждого** слова.
   Это позволяет увидеть группу, фактически состоящую из одного доминирующего слова,
   или смесь нескольких объектов. Эффект интерпретируемости не равен росту WGA.

4. **Anchor → raw NN → projected NN.** Для каждой показанной группы два
   детерминированных anchor: top activation и низкая activation (квартиль).
   Рядом — raw/projected cosine к новому соседу и его наличие в итоговом graph.
   Сосед после проекции не обязан стать retained teacher edge: остаются residual,
   contrast, null и degree ограничения.

5. **Обязательный контрпример.** Отдельно отображается retained wrong-target edge
   с максимальной массой. Это явно post-hoc выбранный failure, а не часть слепой
   случайной выборки. Отрицательные случаи не удаляются ради красивого рисунка.

`row_weight` — нормированная вероятность донора **внутри строки**. При одном доноре
она равна 1 даже у слабого anchor; её нельзя сравнивать как абсолютную уверенность
между всеми рёбрами. Произведение `q_i*p_T(j|i)` лучше описывает бюджет supervision,
но тоже не является вероятностью семантической правильности или измеренным
градиентом: реальный KL дополнительно зависит от batch и student distribution.
Степень anchor записана в ключе слепых пар; bins не устраняют degree-конфаундинг.

## Почему sparsity важна — и почему «чем больше, тем лучше» неверно

Отдельно записываются:

| Величина | Что она различает |
|---|---|
| Размер группы | Число слов, а не число удаляемых направлений |
| Numerical rank | Сколько направлений действительно удаляет текущая full-subspace projection |
| Support | На какой доле изображений активируется хотя бы одно слово |
| Hoyer sparsity | Насколько активация всей группы концентрируется на небольшом числе изображений |
| Effective members = exp(entropy(shares)) | Несут ли несколько слов заметную суммарную массу или одно доминирует |
| Active members given active | Слова совместно работают или заменяют друг друга |
| Coactivation cosine | Эмпирическая совместность, диагностическая, не обязательный semantic gate |
| Min pairwise text cosine | Вся группа близка или является транзитивной цепочкой |
| Mean / p95 removed energy | Насколько сильно проекция изменяет CLIP representation |

Очень редкая группа может быть случайной, повсеместная — мало специфичной.
Высокую Hoyer sparsity нельзя автоматически награждать: это будет поощрять редкие
или почти мёртвые группы. В этой версии Hoyer — diagnostic; меняется **политика
отбора групп**, не коэффициент L1 SpLiCE и не сохранённые коды.

Синонимы могут иметь numerical rank >1 даже при сильной близости. Объединение
удаляет всё это подпространство, в том числе полезные признаки объекта. Rank penalty
и energy gate ограничивают этот риск, но не доказывают, что удаляется именно фон.
Truncated-SVD/rank-1 projection — отдельная будущая гипотеза, не скрытый параметр
текущего перебора. Ортогональная проекция по словам также не эквивалентна гарантии
линейного удаления размеченного концепта, как в [LEACE](https://arxiv.org/abs/2306.03819).

## Зафиксированный небольшой поиск

Данные: тот же frozen train cache и исходный CRP graph с проверкой fingerprints.
Target/background annotations не загружаются при построении, аудите и отборе.
Training IDs детерминированно разбиваются SHA-256 на две половины. Dictionary,
image_mean и исходная SpLiCE декомпозиция общие: это проверка устойчивости на
разных изображениях, **не новый независимый датасет**.

**Кандидаты.** Только частоты [.01,.95] на discovery-половине. Два фиксированных
порога text cosine .65/.75; complete-link наборы размером 2–4, без обязательной
coactivation. Поочерёдный выбор из четырёх strata «порог × пары/тройки-четвёрки»
по coherence × sqrt(union support), с дедупликацией. Максимум 24 составные группы.
Добавляются их singleton-составляющие (до 96) и текущие выбранные группы (до 12):
максимум 132 кандидата. Это ограниченный heuristic shortlist, не поиск глобального
оптимума по всем комбинациям слов.

**Аудит.** Две CPU array-задачи: discovery и confirmation. Каждая использует
неизменные canonical relation/residual/null правила, 32 random-subspace и
32 shuffled-activation trials на группу. Размерность random-subspace совпадает
с numerical rank группы. Group IDs и состав одинаковы в обеих половинах.

**Две политики discovery-отбора, до 12 групп каждая:**

- `semantic`: исходный null-excess score; support [.01,.95], p95 removed energy ≤.5.
- `compact`: те же условия; у составной группы effective members ≥1.5 и доля
  доминирующего слова ≤.85. Рейтинг:
  `null_excess * sqrt(1 - mean_removed_energy) / sqrt(rank)`.

В обеих политиках не берутся одновременно группы с Jaccard по составу >.5.
Singleton controls остаются в том же кандидатном пуле; составные группы не обязаны
победить. Политики различаются сразу несколькими связанными критериями, поэтому
это сравнение политик, **не чистый causal effect одной sparsity метрики**.

**Confirmation gate без переотбора:** не меньше половины discovery shortlist
снова проходит construction-null и energy gate; минимум одна составная группа
есть в shortlist и среди подтвердившихся. Не прошедшая политика не получает SSL.
Нельзя выбирать другой shortlist по результатам confirmation.
64 null scores и .95 threshold не дают FDR/семейной гарантии после перебора 132 групп.
Confirmation ограничивает произвол, но не превращает screen в формальную проверку
семантической правильности всех выбранных групп.

**Full graph.** До двух CPU-задач, только для прошедших политик, на полном train.
Состав discovery shortlist фиксирован; canonical full audit может отклонить часть.
Graph top-k=3, indegree cap=10, residual/null правила прежние. Для SSL нужны
coverage ≥.5 и хотя бы одна retained составная группа. Иначе это отрицательный
результат поиска, а не повод автоматически ослаблять критерии.

## SSL: максимум 6 × 500 эпох

Три arms: исходный graph (`baseline`, преимущественно singleton), `semantic`,
`compact`; seeds 1 и 3. Используется существующий `training_command` control runner:
ResNet18_large, lambda=.5, SimCLR temperature=.05, graph temperature=.25,
warm-up 10/10, group-balanced ds_train logistic probe L2=.001, validation каждые
25 эпох, фиксированный endpoint 500. Checkpoints и W&B сохраняются.

Если ни один новый graph не проходит gate, не запускается даже новый baseline.
Если проходит только один, запускаются 4 runs; если оба — 6. Baseline новый и
согласованный по текущему коду; исторические runs не подменяются и не суммируются
как дополнительные seeds. Если оба новых графа совпадут, это следует указать;
политики могут не давать различимых treatment.

Основной screen-критерий улучшения относительно нового baseline:

- ΔAvg >0 и ΔWGA >0 на **обоих** seeds;
- среднее ΔAvg ≥1 процентного пункта и среднее ΔWGA ≥2 п.п.

Обе политики и все результаты публикуются независимо от знака. При mixed результате
нельзя заявить устойчивое улучшение. Успех двух seeds — основание для отдельно
спланированной репликации, не для автоматического расширения или открытия test.
Support, row mass и фактические batches между графами различаются; это проверка
полного pipeline. Для изоляции KL от sampler потребуются отдельные matched controls.

## Запуск на кластере

Новые файлы нужно перенести вместе с `splice/crp.py`, `splice/concept_group_search.py`,
двумя новыми Python runners и тестом. Исходные cache/graph должны быть восстановлены.
Все `.err/.out` в корневой **logs**, создать её до первого sbatch.
Максимум 40G RAM/задачу, 5 CPU, 4 DataLoader workers. SSL: 1 V100/задачу,
до четырёх одновременно; frozen audits/builds: до двух CPU-задач одновременно.

```bash
cd /home/xar68reb/Spur_SpLiCE
mkdir -p logs
p=$(sbatch --parsable scripts/group_search_00_prepare.sbatch); p=${p%%;*}
a=$(sbatch --parsable --dependency=afterok:$p scripts/group_search_01_audit.sbatch); a=${a%%;*}
s=$(sbatch --parsable --dependency=afterok:$a scripts/group_search_02_select.sbatch); s=${s%%;*}
g=$(sbatch --parsable --dependency=afterok:$s scripts/group_search_03_build.sbatch); g=${g%%;*}
sbatch --dependency=afterok:$g scripts/group_search_04_visual.sbatch
```

Здесь можно остановиться и прочитать frozen/visual outputs. Не менять правила
отбора на основании post-hoc labels или красивых картинок. Для ограниченного SSL
этапа после успешной сборки:

```bash
t=$(sbatch --parsable scripts/group_search_05_ssl.sbatch); t=${t%%;*}
sbatch --dependency=afterok:$t scripts/group_search_06_summary.sbatch
```

Если ставите SSL заранее, добавьте `--dependency=afterok:$g`. ID массива SSL:
0 baseline seed1, 1 semantic seed1, 2 compact seed1,
3 baseline seed3, 4 semantic seed3, 5 compact seed3.
Повторно не запускать одну и ту же серию параллельно самой себе.
Завершённые результаты переиспользуются; неполные training-каталоги требуют
восстановления или архивирования перед повтором, автоматического resume нет.

W&B: `gsgrechkin-rptu/Spur_SpLiCE`, group `concept_group_search_v1`,
run names `group_search_POLICY_seedN`, online по умолчанию.
Test нигде не используется. При наличии `final_test_lock.json` прежней основной
серии подготовка останавливается: новый поиск нужно вести в отдельном исследовательском
checkout, сохранив основной зафиксированный эксперимент.

## Результаты и чтение разметки

Все результаты: `outputs/concept_group_search_v1/`.

| Файл | Содержание |
|---|---|
| `candidates.json` | Состав кандидатов, split IDs, water-word diagnostic, fingerprints |
| `discovery.json`, `confirmation.json` | Все group scores, nulls и sparsity/energy metrics, включая отвергнутые |
| `selection.json` | Замороженные shortlists и результат confirmation gate |
| `graphs/{semantic,compact}.json` | Full graphs, если политика прошла gate |
| `visual/POLICY/index.html` | Галерея групп/соседей, failure case и таблица relation bins |
| `visual/POLICY/blind_pairs.html` | Слепая оценка пар |
| `visual/POLICY/{ratings.csv,pair_key.json}` | Пустая форма разметки и отдельный ключ |
| `visual/POLICY/diagnostics.json` | Relation bins, full per-group relation audit и примеры |
| `ssl/seedN/POLICY/training/*/last.pth` | Финальная модель |
| `ssl_summary.json` | Все endpoints, per-group metrics, paired deltas и исход screen |

После заполнения CSV:

```bash
# В grgrie-train; task 0=baseline, 1=semantic, 2=compact.
python -m scripts.tools.visualize_concept_group_search --task 0 --summarize-ratings
```

Получится `visual/baseline/human_ratings_summary.json`. Для двух других — IDs 1/2.
CSV не перезаписывается повторным рендером. HTML использует соседние JPEG, поэтому
переносить/открывать нужно весь каталог `visual/POLICY`, не один HTML файл.
Повышение оценок интерпретируемости, качества рёбер и Avg/WGA — три разных вывода;
один не подставляется вместо другого.

## Полный concept-group cosine sweep на картинках

`scripts/group_search_07_compare.sbatch` запускает
`scripts/tools/render_group_similarity_sweep.py` и создаёт один self-contained
HTML. Обученные checkpoints и test split не используются. В начале отчёта
перечислены все группы `G*`; затем для каждой диагностической пары измеряются
исходный cosine, cosine после удаления каждой группы и
`gain = after - initial`.

Пары выбираются детерминированно только по frozen centered CLIP, до вычисления
intervention gains:

- `A+B` и `A+C`: одинаковый label, разные spurious data, общий anchor A;
  соответственно высокая и низкая исходная похожесть.
- `D+E` и `D+F`: одинаковые spurious data, разные labels, общий anchor D;
  соответственно высокая и низкая исходная похожесть.
- `K+N` и `K+L`: у всех metadata `(y, place)=(1,1)`, общий anchor K;
  соответственно высокая и низкая исходная похожесть.
- `X+Y` и `X+Z`: X имеет `(1,0)`, Y/Z имеют `(0,1)`, общий anchor X;
  соответственно высокая и низкая исходная похожесть.

Один запуск:

```bash
sbatch scripts/group_search_07_compare.sbatch
```

Меняемые параметры: `METHOD_POLICY`, `CACHE_PATH`, `GRAPH_PATH`, `DATA_FOLDER`,
`OUTPUT_PATH` и `GROUP_SCOPE`. По умолчанию `GROUP_SCOPE=selected`, поэтому в
индекс и каждый sweep входят только прошедшие graph selection группы. Значение
`GROUP_SCOPE=all` возвращает также отклонённые audited groups. Результат записывается в
`outputs/concept_group_search_v1/visual/POLICY/group_similarity_sweep.html`.
Все изображения встроены в HTML как data URI, поэтому файл можно переносить и
открывать отдельно.
