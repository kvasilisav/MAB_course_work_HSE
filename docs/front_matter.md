# Алгоритмы многорукого бандита как адаптивная альтернатива A/B-тестированию в задачах рекомендательных систем и оптимизации CTR

**Курсовая работа**

---

## Аннотация

Работа посвящена **онлайн-экспериментам** в рекомендательных системах с метрикой CTR: сравнению фиксированного A/B-разбиения трафика и адаптивных политик многорукого бандита (ε-greedy, UCB1, Thompson Sampling, LinUCB). Теоретически известно, что бандиты снижают regret во время эксперимента, тогда как A/B обеспечивает статистически защищённый вывод о rollout; на практике эти цели часто смешивают при интерпретации логов.

**Метод:** воспроизводимый пайплайн на Python с разделением четырёх типов оценки — онлайн-оптимизация, статистический вывод, офлайн-оценка (OPE) и продуктовая применимость. Эмпирика на синтетических средах, contextual synthetic и Open Bandit Dataset (OBD): **14 основных экспериментов** E1–E14 и бенчмарк latency (задержки) P1 — в основном тексте; разведочные batch-прогоны E3 и E8 — приложение А.

**Главный вывод:** адаптивное назначение трафика даёт измеримую пользу по кликам и regret (блок I), но **не** заменяет корректный статистический вывод (блок II): naïve A/B на логах бандита завышает ошибку I рода (14,5% против 3% у fixed split); IPS-weighted inference восстанавливает контроль (5%); при peeking на fixed split нужны sequential-процедуры (OBF, mSPRT). OPE на OBD работает как пайплайн, но не даёт уверенного ranking алгоритмов при низком CTR.

**Продуктовая рекомендация:** MAB — для максимизации кликов во время эксперимента; отдельный fixed A/B — для rollout; IPS — post-hoc на bandit-логах; OBF/mSPRT — только для мониторинга fixed A/B. План экспериментов на диплом (примерные направления) — §4.3.

**Ключевые слова:** многорукий бандит, A/B-тестирование, CTR, онлайн-эксперимент, off-policy evaluation, статистический вывод, Open Bandit Dataset.

---

## Сокращения

| Сокращение | Расшифровка |
|---|---|
| A/B | сплит-тест, сравнение контрольного и тестового вариантов |
| CTR | click-through rate, доля кликов |
| MAB | multi-armed bandit, многорукий бандит |
| OPE | off-policy evaluation, офлайн-оценка политики по логам |
| IPS / SNIPS | inverse / self-normalized propensity scoring |
| ESS | effective sample size, эффективный размер выборки |
| OBD | Open Bandit Dataset |
| OBF | границы O'Brien–Fleming в group sequential design |
| mSPRT | mixture sequential probability ratio test (always-valid) |
| BTS | Bernoulli Thompson Sampling (logging policy в OBD) |
| DR | doubly robust estimator |
| CUPED | controlled-experiment using pre-experiment data |
| MDE | minimum detectable effect, минимальный детектируемый эффект |
