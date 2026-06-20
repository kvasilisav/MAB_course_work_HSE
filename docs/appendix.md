# Приложение А. Протоколы вспомогательных экспериментов

Материал не входит в основной нарратив главы 3; воспроизведение — через `src/experiments/` и журнал разработки.

## А.1. E3 — batch smoke test на OBD

Разведочная проверка batch-режима: 10 000 событий `random/all`, `batch_size = 200`, 2 seed. Политика обновляется пакетами; при несовпадении с логом награда симулируется по эмпирическому CTR. **Не strict OPE** — ranking не интерпретируется. Артефакт: `outputs/obd_batch_smoke/`.

## А.2. E8 — batch с контекстом на OBD

Перенос LinUCB/TS на OBD в batch-режиме (20 seed). Batch-simulator не даёт counterfactual награды для контекстной модели — выводы только из E8b (синтетика). Артефакт: `outputs/extended_full/e8_obd_batch_contextual/`.

## А.3. E13 — выбор pairwise-пары на OBD

Скрипт `obd_pair_selection.py`: `random/all`, 10k событий; топ-20 `item_id` по показам; `min_impressions = 100`, `min_clicks = 1`; пара с минимальным |CTR_A − CTR_B|. Результат: items 41 и 50 (по 136 показов, CTR ≈ 0,735%, gap 0). Проекция: 272 события (2,7% лога). **Не A/B-дизайн** — иллюстрация метрик acceptance/ESS. Артефакты: `outputs/obd_pair/item_ctr_stats.csv`, `pair_selection.csv`.
