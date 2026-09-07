# Последняя серия перед встречей с руководителем

Цель: проверить преимущество SpLiCE над raw CLIP distillation, пользу intervention-механизма CRP и показать его на реальных изображениях. **Нынешние результаты превосходства ещё не доказывают.** Это план реализации и одного запуска серии; этой правкой training code не изменён. Прежний подробный план сохранён в `ARCHIVE_DETAILED_NEXT_ACTIONS_2026-09-07.md`.

## 1. Исправить temperature и AMP-диагностику

- Во всех новых runs явно задать **SimCLR temperature=.05**, для graph KL — **temperature=.25**. Старую direct серию при .5 сохранить отдельным результатом.
- В диагностике делать `autograd.grad` от scaled losses и делить возвращённые float32 gradients на scale. `.float()` только у итогового scalar не устраняет FP16 underflow. Сравнить измерение с FP32 на одном фиксированном batch; проверить, что диагностика не меняет optimizer update, RNG и BN buffers.
- Перед запуском: короткий smoke на совпадение image IDs с targets/graph, fixed shuffle, ненулевой gradient до encoder после включения loss и одинаковый update matched arms до его включения. Проверить сохранённый integrity report; при ошибке остановить jobs, а не повторять весь старый диагностический sweep.

## 2. Исправленный запуск: SpLiCE против CLIP distillation

**4 метода × seeds1/3 = 8 runs по500 epochs.**

| Метод | Дополнительная цель |
|---|---|
| Matched SimCLR | alpha=0; тот же head и data pipeline |
| Raw CLIP distillation | Нормированный centered CLIP embedding |
| SpLiCE distillation | Нормированный full reconstruction `codes @ dictionary` |
| Shuffled SpLiCE | Та же reconstruction, fixed permutation seed101 |

Повторить предыдущий direct screen **с единственным изменением training recipe: temperature .5→.05**. Сохранить alpha=.1, start10/warmup10, SGD lr=.01, milestones350/400/450, batch128, augmentations, head, targets и общий valid mask. Исправленный измеритель не должен менять обучение.

Основной endpoint: **epoch500**, frozen encoder h, logistic ds_train→val, L2=.001. Показать оба seeds и парные дельты Avg/WGA, не выбирать лучший checkpoint. Сравнение с SimCLR проверяет пользу переноса; с raw — дополнительную пользу SpLiCE; с shuffled — важность соответствия изображения и teacher.

## 3. Реализовать ablation из скриншота

**2 метода × seeds1/3 = 4 runs по500 epochs.**

- **CRP:** существующий frozen graph с concept projection/intervention, λ=.5.
- **SpLiCE semantic graph:** соседи по cosine similarity нормированного `codes @ dictionary`, без projection, intervention gain и residual-group gate при выборе соседей. Один вариант representation, без sweep по codes/reconstruction.

Реализовать по образцу matched raw-graph builder: тот же cache, поддержанные anchors, outdegree каждой строки, профиль весов, anchor confidence и indegree cap. Менять identities соседей. Одинаковые graph sampler algorithm и KL; SimCLR=.05, KL=.25, λ=.5, start10/warmup10 и остальные настройки прежнего CRP recipe. Проверить совпадение бюджетов, сохранить edge-overlap. Сделать свежие matched CRP runs, чтобы не предполагать эквивалентность изменённого кода старым запускам.

**Что проверяем:** нужны ли CRP relations сверх обычной SpLiCE similarity при фиксированной силе регуляризации. Budgets/confidence наследуются от CRP; это контроль выбора связей, а не полностью независимый от CRP pipeline. Изменение соседей меняет и batches. Поэтому результат относится к intervention-based graph pipeline целиком, не к одному projection operator. Raw CLIP graph и direct CLIP distillation — разные controls.

## 4. Реализовать визуальный эксперимент из скриншотов

**До5 панелей без student training.** Взять фактические concept groups с наибольшим числом приписанных retained edges; tie-break по group ID. Не использовать labels и ручной отбор красивых результатов. Если сохранённые группы — singleton concepts, показать их; не придумывать группы water/lake/pond из иллюстрации.

На группу:2 high-signal и2 low-signal пары anchor/donor из **того же group-specific candidate pool до фильтрации**. Главный показатель:

`gain_G(i,j) = cosine(u_i_without_G, u_j_without_G) − cosine(u_i, u_j)`.

Использовать centered CLIP vectors и реальные subspaces CRP. High/low — верхний/нижний децили gain; по возможности те же anchors и близкая исходная similarity. Выбор внутри корзин фиксировать seed0 до просмотра изображений. При нехватке кандидатов отметить это. Если candidate evidence не сохранён, пересчитать его с прежним cache/config, без обучения.

На панели: реальные изображения и IDs, реальные названия concepts, gain, activation contrast, residual similarity, null-calibrated confidence, final edge weight и retained/rejected. У rejected пары final weight=0; недоступная confidence=N/A. Confidence — score, не вероятность. Сохранить PNG/PDF и JSON с правилами выбора и показанными парами.

Подпись: соответствует ли видимое различие группе и сохраняется ли общая семантика; отметить контрпримеры. Визуализация объясняет механизм, **но сама не доказывает рост WGA или причинное устранение фона**.

## 5. Запуск, бюджет и материал к встрече

После smoke отправить **одну серию:8 direct +4 graph ablation =12 full runs**, всего6000 SSL epochs. Визуализацию делать параллельно. Исправленную gradient диагностику встроить в runs на первых4 batches epochs1/11/20/25/500, отдельно по encoder и direct head. Отдельная серия3×25 diagnostics сейчас не нужна. W&B online, новые protocol names, сохранить final checkpoints и probe artifacts.

Перед отправкой проверить длительности прежних500-epoch runs в W&B и доступные GPU: завершение6000 epochs за ночь не гарантировано. При ограниченных ресурсах завершать matched блоки целиком; не сокращать epochs отдельным arms и не выдавать промежуточную оценку за epoch500. Визуальные панели подготовить независимо от очереди.

Для руководителя: **две таблицы и до5 панелей**. Первая таблица — corrected direct screen с дельтами к raw/SimCLR/shuffled. Вторая — CRP vs semantic SpLiCE graph. Все seeds и отрицательные результаты включить.

Критерий direct успеха прежний: положительные Avg/WGA к SimCLR на обоих seeds, средние ≥+1/+2 п.п.; положительные обе средние дельты к raw и shuffled. Для ablation — положительные обе дельты CRP к semantic graph на обоих seeds как согласованный предварительный сигнал. Два development seeds не дают независимого статистического подтверждения.

Если выигрывает только WGA, один seed или только CRP против semantic graph — сообщить именно этот ограниченный результат. Если SpLiCE не превосходит raw distillation, утверждать обратное нельзя. Полноценная исследовательская работа может показать механизм, честные controls и границы эффективности; положительный исход определяется данными.

SSL, graph/target preparation и выбор визуальных примеров остаются label- и spurious-metadata-free. Labels/groups — только downstream evaluation и последующее объяснение, test закрыт. Новые crops/head sweeps, spatial variants, datasets и широкие диагностики в эту серию не входят.
