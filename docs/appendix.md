# Приложение А. Протоколы вспомогательных экспериментов

Материал **не входит в основной нарратив** главы 3 (§3.5–3.8): в тексте курсовой приведены E1–E14 и P1; разведочные batch-прогоны **E3** и **E8** и иллюстративная pairwise-проекция **E13** вынесены сюда, чтобы не смешивать IPS/SNIPS replay-OPE (E2/E9) с приближённым batch-симулятором и постфактум-проекцией логов на две «карточки».

**Воспроизведение:** модули в `src/experiments/`; данные OBD — через `scripts/prepare_open_bandit.py`. Сводные CSV — в `outputs/` (часть закоммичена для сверки с текстом; полные прогоны — `run_all.ps1` / `run_all.sh`).

---

## А.0. Почему batch и replay-OPE разделены

В репозитории три режима работы с логами OBD:


| Режим                | Модуль / флаг                      | Counterfactual-награда                                                                             | Propensity  | Использование в выводах        |
| -------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------- | ----------- | ------------------------------ |
| **IPS/SNIPS replay-OPE** | `compare_ab_vs_bandits --mode ope` | Да (replay + SNIPS; candidate **заморожена**, `freeze_policy=True`)                            | Обязательна | E2, E2b/c, E9, E13 (OPE-часть) |
| **batch**            | `--mode batch`                     | Нет: при несовпадении руки награда **симулируется** по эмпирическому CTR (`LoggedClicksBanditEnv`) | Не для OPE  | Только E3, E8 (приложение)     |
| **онлайн-синтетика** | `--mode synthetic`                 | Оракул Bernoulli / contextual                                                                      | —           | E1, E6, E8b, E12               |


Batch-режим удобен для отладки пайплайна и сравнения «скорости обучения» политик на реальном распределении контекстов, но **не** даёт корректной counterfactual-оценки для LinUCB и не заменяет IPS/SNIPS. Поэтому выводы о контекстных бандитах в курсовой опираются на **E8b** (контролируемая синтетика), а не на E8.

**Исправление методологии replay-OPE (по рецензии).** В `src/ope/replay.py` и `obd_streaming_ope.py`: candidate-политика не обновляется при оценке (`freeze_policy`); propensity клипируется снизу (`propensity_floor=0.01`); события перетасовываются per seed; знаменатель IPS — число валидных строк лога, не общий объём файла.

---

## А.1. E3 — batch smoke test на OBD

### Цель

Разведочная проверка, что batch-обновление политик (`batch_size > 1`) на потоке событий OBD **технически работает**: загрузка логов, цикл `select_arm` → `update`, агрегация reward/regret.

### Постановка


| Параметр     | Значение                                                                |
| ------------ | ----------------------------------------------------------------------- |
| Данные       | `random/all`, 10 000 событий (малый релиз OBD)                          |
| Режим        | `batch` (`LoggedClicksBanditEnv`)                                       |
| `batch_size` | 200                                                                     |
| Политики     | `fixed_ab`, `thompson_sampling`, `epsilon_greedy` (типовой набор smoke) |
| Seeds        | 2                                                                       |
| Горизонт     | Все события файла (10k)                                                 |


### Механика награды 

Если выбранная политикой рука **совпадает** с залогированной в событии — используется наблюдаемый клик. Иначе награда **сэмплируется** из Bernoulli с CTR, оценённым по эмпирике лога для этой руки (сглаживание Beta(1,1)). Это **не** counterfactual из production-модели и **не** IPS/SNIPS replay-OPE.

### Команда

```powershell
python -m scripts.prepare_open_bandit --download --behavior-policy random --campaign all --include-context --output-path data/processed/obd_events.csv

python -m src.experiments.compare_ab_vs_bandits --mode batch `
  --events-path data/processed/obd_events.csv `
  --seeds 2 --batch-size 200 `
  --output-dir outputs/obd_batch_smoke
```

### Результат и статус

- Артефакты: `outputs/obd_batch_smoke/` (summary, results по seed).
- **Ranking политик не интерпретируется** в основном тексте.
- E3 подтверждает работоспособность batch-пайплайна перед E8; для продуктовых выводов используются E2/E9.

---

## А.2. E8 — batch с контекстом на OBD (LinUCB / TS)

### Цель

Попытка перенести **контекстные** политики (`linucb`, `thompson_sampling`, `fixed_ab`) на реальные логи OBD в том же batch-режиме, что и E3, с признаками из `item_context.csv`.

### Постановка


| Параметр      | Значение                                                      |
| ------------- | ------------------------------------------------------------- |
| Данные        | `data/processed/obd_events.csv`, `random/all`, 10 000 событий |
| Режим         | `batch`, `batch_size = 200`                                   |
| Политики      | `fixed_ab`, `thompson_sampling`, `linucb`                     |
| Seeds         | 20 (полный прогон в `run_full_experiments`)                   |
| `context_dim` | Из лога (7 признаков при `--include-context`)                 |


### Команда

```powershell
python -m src.experiments.run_full_experiments
# блок E8 → outputs/extended_full/e8_obd_batch_contextual/
```

или:

```powershell
python -m src.experiments.compare_ab_vs_bandits --mode batch `
  --events-path data/processed/obd_events.csv `
  --policies fixed_ab,thompson_sampling,linucb `
  --seeds 20 --batch-size 200 --include-context `
  --output-dir outputs/extended_full/e8_obd_batch_contextual
```

### Эмпирические итоги (20 seed, агрегат в `full_report.json`)


| policy_name       | mean cumulative reward | mean regret | suboptimal_share |
| ----------------- | ---------------------- | ----------- | ---------------- |
| thompson_sampling | 131,3                  | 213,5       | 0,976            |
| linucb            | 119,3                  | 225,6       | 0,980            |
| fixed_ab          | 112,7                  | 232,2       | 0,987            |


TS показывает наименьший regret в batch-режиме, LinUCB — между TS и `fixed_ab`.

### Почему E8 не в основном тексте

1. **Нет counterfactual для контекстной модели** — при выборе руки, отличной от залогированной, награда берётся из упрощённого CTR-prior, а не из модели с контекстом.
2. **Нельзя сопоставить с E8b** — на синтетике (E8b) LinUCB использует истинную связь контекст → CTR; на OBD batch это приближение.
3. **Строгий вывод о LinUCB** в курсовой сделан по **E8b** (§3.6.1) с осторожным bootstrap; OPE для LinUCB на full OBD не запускался (P1: ~23 ms/decision при 80 руках).

E8 сохранён как документация для кейса применения batch к реальным логам, без включения в product decision matrix.

---

## А.3. E13 — pairwise-проекция OBD 

### Цель (RQ7, иллюстрация)

Связать **80-ручный** каталог OBD с продуктовой постановкой «вариант A vs вариант B» одной карточки: показать, как меняются **acceptance rate**, **ESS** и exploratory inference при сужении до двух `item_id`. Это **методологический мост**, а не замена отдельного A/B-эксперимента с фиксированным split (rollout — E12 + E4 в основном тексте).

### Этап 1. EDA и выбор пары (`obd_pair_selection.py`)


| Правило           | Значение                                          |
| ----------------- | ------------------------------------------------- |
| Источник          | `data/raw/obd_random_all.csv` (10k, `random/all`) |
| Пул кандидатов    | Топ-20 `item_id` по числу показов                 |
| `min_impressions` | 100 на товар в пуле                               |
| Критерий выбора   | Минимальный |CTR_A − CTR_B| (квази-null)          |


**Выбранная пара:** `item_id` **41** и **50**.


|        | item 41 | item 50 |
| ------ | ------- | ------- |
| Показы | 136     | 136     |
| Клики  | 1       | 1       |
| CTR    | 0,735%  | 0,735%  |
| |gap|  | 0       |         |


Артефакты: `outputs/obd_pair/item_ctr_stats.csv`, `pair_selection.csv`, `pair_selection.json`.

```powershell
python -m src.experiments.obd_pair_selection --output-dir outputs/obd_pair
```

### Этап 2. Проекция лога

Скрипт `obd_pairwise_ope.py` / `convert_obd_to_pairwise_events`:

1. Оставляются строки с `item_id ∈ {41, 50}`.
2. Ремап в руки: 0 (control) и 1 (treatment).
3. Сохраняется `propensity` из исходного лога (random logging policy).

Итого **272 события** (~~2,7% исходного лога 10k), по **~~1 клику на руку** — крайне мало для rollout-выводов.

### Этап 3. IPS/SNIPS replay-OPE на 2 руках


| Параметр           | Значение                                                  |
| ------------------ | --------------------------------------------------------- |
| Оценщик            | SNIPS (тот же `ope/replay.py`, что E2)                    |
| Candidate-политики | `fixed_ab`, `thompson_sampling`, `epsilon_greedy`, `ucb1` |
| Seeds              | 10                                                        |


Типичные значения на проекции:

- **acceptance rate** ≈ 48–53% (против ~1,3% при 80 руках в E2) — метрика становится читаемой;
- **ESS** (`fixed_ab`) ≈ 130–145 при 272 событиях — сопоставимо с ESS ≈ 130 на 10k в 80-ручной постановке;
- **SNIPS** между политиками и seed **нестабилен** (единицы кликов, дискретность).

Артефакт: `outputs/obd_pair/ope_summary.csv`.

### Этап 4. Exploratory inference

На том же pairwise-логе однократно:

- **naïve A/B** (z-критерий, CTR_B > CTR_A);
- **IPS-weighted** pairwise inference.

При квази-null (gap = 0): `p_value = 1`, `reject_null = False` — ожидаемо. Артефакт: `outputs/obd_pair/inference_summary.csv`.

```powershell
python -m src.experiments.obd_pairwise_ope --output-dir outputs/obd_pair
```

### Сравнение с multi-arm OBD (смысл E13)


|                    | Multi-arm (E2, E9)            | Pairwise (E13)                                    |
| ------------------ | ----------------------------- | ------------------------------------------------- |
| Вопрос             | Ranking политик на 80 товарах | Поведение метрик на 2 «вариантах»                 |
| Руки               | 80                            | 2                                                 |
| События            | 10k – 1,37M                   | 272                                               |
| ESS (`fixed_ab`)   | 130 – 17 243                  | ~137                                              |
| Acceptance         | ~1,3% (80 рук)                | ~50%                                              |
| Вывод для продукта | Ranking хрупок при low CTR    | Иллюстрация интерпретируемости, **не** дизайн A/B |


Подробное сравнение — **табл. 3.19** главы 3; онлайн- и inference-часть pairwise — **E12** (синтетика, основной текст).

---

## А.4. Сводка команд воспроизведения 

```powershell
# Данные OBD (10k)
python -m scripts.prepare_open_bandit --download --behavior-policy random --campaign all --include-context --output-path data/processed/obd_events.csv

# E3 — smoke batch
python -m src.experiments.compare_ab_vs_bandits --mode batch --events-path data/processed/obd_events.csv --seeds 2 --batch-size 200 --output-dir outputs/obd_batch_smoke

# E8 — batch + context (или через run_full_experiments)
python -m src.experiments.run_full_experiments

# E13 — pairwise
python -m src.experiments.obd_pair_selection --output-dir outputs/obd_pair
python -m src.experiments.obd_pairwise_ope --output-dir outputs/obd_pair
```

Полный прогон основных экспериментов (без обязательного повтора E3): `.\run_all.ps1` (E12, E13 включены; E3/E8 — по командам выше).

---

## А.5. Ограничения приложения

- E3/E8 **не** реализуют DR [27] и **не** проходят проверки replay-OPE из §3.6.3.
- E13 **не** фиксирует propensity под гипотезой «две версии одного SKU»; это постфактум-фильтр по `item_id`.
- При другом random seed выбора пары или другом `min_impressions` пара 41/50 может смениться — протокол воспроизводим при фиксированном `pair_selection.csv`.

