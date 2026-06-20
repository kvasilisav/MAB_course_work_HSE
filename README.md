# Курсовая работа на тему "Алгоритм Многорукого Бандита (MAB) как альтернатива A/B тестированию"

Воспроизводимый Python-проект: сравнение **фиксированного A/B-разбиения** и **адаптивных политик многорукого бандита** (ε-greedy, UCB1, Thompson Sampling, LinUCB) в постановке онлайн-эксперимента по CTR.

Два блока оценки (см. гл. 3):

1. **Блок I** — польза адаптивного назначения (regret, клики): E1, E6, E8b, E12.
2. **Блок II** — корректный статистический вывод (Type I, IPS, sequential): E4, E5, E11, E14.

Дополнительно: strict OPE на Open Bandit Dataset (E2, E9), latency (P1). E3/E8 — приложение А.

## Текст работы (исходники для LaTeX)


| Файл                                                                       | Раздел                                |
| -------------------------------------------------------------------------- | ------------------------------------- |
| `[docs/front_matter.md](docs/front_matter.md)`                             | Титул, аннотация, сокращения          |
| `[docs/literature_review_academic.md](docs/literature_review_academic.md)` | Главы 1–2, список литературы [1]–[40] |
| `[docs/experiments_chapter.md](docs/experiments_chapter.md)`               | Глава 3                               |
| `[docs/conclusion_chapter.md](docs/conclusion_chapter.md)`                 | Глава 4                               |
| `[docs/appendix.md](docs/appendix.md)`                                     | Приложение А (E3, E8, E13)            |


## Быстрый старт

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Полное воспроизведение основных экспериментов и графиков:

```powershell
# Данные (10k OBD для E2, E13, рис. 3.5):
python -m scripts.prepare_open_bandit --download --behavior-policy random --campaign all --include-context --output-path data/processed/obd_events.csv

.\run_all.ps1
```

`run_all.ps1` прогоняет: pytest → E1/E2/E8b (`run_full_experiments`) → `full_report.json` → E4 → E11 → E14 → gap-сценарии → **E12** → **E13** → рисунки. **E9** (полный OBD, ~1,4M событий) — только если локально есть `data/obd_full/`; иначе шаг пропускается с предупреждением. Команды E9 вручную — в `[docs/internal/experiments_worklog.md](docs/internal/experiments_worklog.md)`.

Рисунки для гл. 3: `outputs/figures/fig_3_*.png` (8 файлов) — **в git**; пересборка: `scripts/generate_coursework_figures.py` или `.\run_all.ps1`.

## Данные (Open Bandit Dataset)

Скачать и конвертировать 10k `random/all` в формат проекта:

```powershell
python -m scripts.prepare_open_bandit --download --behavior-policy random --campaign all --include-context --output-path data/processed/obd_events.csv
```

Полный OBD для E9 — локально в `data/obd_full/` или streaming через `src.experiments.obd_streaming_ope` (см. исходники). CSV не коммитятся (`.gitignore`).

## Структура кода


| Путь                                         | Назначение                                       |
| -------------------------------------------- | ------------------------------------------------ |
| `src/bandits/`                               | Политики: `fixed_ab`, ε-greedy, UCB1, TS, LinUCB |
| `src/environments/`                          | Синтетика, contextual, logged clicks             |
| `src/ope/`                                   | Strict OPE (SNIPS)                               |
| `src/ab_testing/`                            | Inference, IPS, sequential (E11, E14)            |
| `src/experiments/`                           | Прогоны E1–E14, P1, сборка `full_report.json`    |
| `tests/`                                     | pytest                                           |
| `notebooks/03_ablation_and_discussion.ipynb` | Абляции ε, priors, batch_size (рис. 3.6–3.8)     |


### Режимы `compare_ab_vs_bandits`

- `synthetic` — Bernoulli / contextual synthetic (E1, E8b).
- `ope` — strict replay с propensity (E2).
- `batch` — разведочный batch на OBD (**не** strict OPE; E3/E8 — приложение).

### Ключевые команды (по отдельности)

```powershell
python -m src.experiments.run_full_experiments
python -m src.experiments.assemble_full_report --seeds 20
python -m src.experiments.ab_validity --output-path outputs/ab_validity/summary_h20000.csv
python -m src.experiments.inference_solutions_validity --output-dir outputs/inference_valid
python -m src.experiments.sequential_validity_e14 --horizon 20000 --trials 200 --output-dir outputs/sequential_valid
python -m src.experiments.synthetic_scenarios --horizon 5000 --seeds 20 --output-dir outputs/synthetic_scenarios
python -m src.experiments.product_ab_e12 --output-dir outputs/product_ab
python -m src.experiments.obd_pair_selection --output-dir outputs/obd_pair
```

Сводный отчёт после полного прогона: `outputs/extended_full/full_report.json`.

## Ограничения

- `batch`-режим — приближение, не counterfactual OPE; для честной офлайн-оценки — `ope` + propensity.
- DR/CUPED описаны в тексте (§4.3), в коде не реализованы.
- `FixedABPolicy` — только allocation; вывод — `src/ab_testing/`.

