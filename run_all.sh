#!/usr/bin/env bash
# Воспроизведение основных экспериментов курсовой (Linux/macOS)
# Запуск из корня репозитория: bash run_all.sh
#
# Включает: pytest, E1/E2/E8b block (run_full_experiments), full_report,
# E4, E11, E14, synthetic scenarios, E12, E13, figures.
# E9 (full OBD streaming) — только если есть data/obd_full/ (долго, см. README).

set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f "data/processed/obd_events.csv" ]]; then
  echo "WARN: data/processed/obd_events.csv not found. Run prepare_open_bandit first (see README)."
fi

echo "=== pytest ==="
python -m pytest -q

echo "=== E1/E2/E8b block (run_full_experiments) ==="
python -m src.experiments.run_full_experiments

echo "=== assemble report ==="
python -m src.experiments.assemble_full_report --seeds 20

echo "=== E4 ab validity ==="
python -m src.experiments.ab_validity --output-path outputs/ab_validity/summary_h20000.csv

echo "=== E11 inference (IPS-weighted, bootstrap SE) ==="
python -m src.experiments.inference_solutions_validity --trials 100 --bootstrap 200 --output-dir outputs/inference_valid

echo "=== E15 CUPED vs naive A/B (synthetic) ==="
python -m src.experiments.cuped_power_study --trials 100 --output-dir outputs/cuped_power

echo "=== E14 sequential ==="
python -m src.experiments.sequential_validity_e14 --horizon 20000 --trials 200 --output-dir outputs/sequential_valid

echo "=== synthetic gap scenarios (E1 supplement, bootstrap CI) ==="
python -m src.experiments.synthetic_scenarios --horizon 5000 --seeds 20 --bootstrap 1000 --output-dir outputs/synthetic_scenarios

echo "=== E12 pairwise product A/B ==="
python -m src.experiments.product_ab_e12 --output-dir outputs/product_ab

echo "=== E13 OBD pairwise projection ==="
python -m src.experiments.obd_pair_selection --output-dir outputs/obd_pair
python -m src.experiments.obd_pairwise_ope --output-dir outputs/obd_pair --bootstrap 500

e9a="data/obd_full/open_bandit_dataset/random/all/all.csv"
e9b="data/obd_full/open_bandit_dataset/bts/all/all.csv"
if [[ -f "$e9a" && -f "$e9b" ]]; then
  echo "=== E9a full random/all streaming OPE ==="
  python -m src.experiments.obd_streaming_ope \
    --input-path "$e9a" --behavior-policy random --campaign all \
    --seeds 10 --chunksize 100000 --bootstrap 100 \
    --output-dir outputs/obd_full/e9a_random_all_full
  echo "=== E9b bts/all 1M streaming OPE ==="
  python -m src.experiments.obd_streaming_ope \
    --input-path "$e9b" --behavior-policy bts --campaign all \
    --max-events 1000000 --seeds 10 --chunksize 100000 --bootstrap 100 \
    --output-dir outputs/obd_full/e9b_bts_all_1m
else
  echo "=== E9 skipped (no data/obd_full/); see README for full OBD download ==="
fi

echo "=== figures ==="
python -m scripts.generate_coursework_figures

echo "=== Done ==="
echo "Artifacts: outputs/extended_full/, outputs/figures/, outputs/inference_valid/, outputs/product_ab/, outputs/obd_pair/"
