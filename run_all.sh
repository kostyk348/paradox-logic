#!/usr/bin/env bash
# Reproduce every table/figure in results/ and figures/.
set -e
cd "$(dirname "$0")"

echo "== python experiments =="
for f in experiments/0*.py experiments/A_*.py experiments/B_*.py; do
  name=$(basename "$f" .py)
  echo "--- $name"
  python3 "$f" | tee "results/${name}.txt"
done

echo "== figure =="
python3 experiments/plot_results.py | tee -a results/_figures.txt

echo "== C: Living system Hurst (fast) =="
if command -v gcc >/dev/null; then
  gcc -O2 -o /tmp/paradox_living experiments/living.c -lm
  { echo "LivingSystem Hurst (C port):"; /tmp/paradox_living 8 0.05 5000 200 0; } | tee results/C_living_hurst.txt
  gcc -O2 -o /tmp/paradox_null experiments/hurst_null.c -lm
  { echo "Hurst estimator null:"; /tmp/paradox_null 1000 2000; } | tee results/C_hurst_null.txt
else
  echo "gcc not found, skipping C experiments"
fi

echo "done."
