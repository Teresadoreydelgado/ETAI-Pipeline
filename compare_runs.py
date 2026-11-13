"""
Compares two runs of a scoreboard with the corrected paired t-test.
    python compare_runs.py                                  -> last two runs of results/runs.csv
    python compare_runs.py <run_a> <run_b>                  -> two runs by their id
    python compare_runs.py results_adult/runs.csv           -> week 9: another scoreboard (last two runs)
    python compare_runs.py results_adult/runs.csv <a> <b>   -> another scoreboard, two runs by id
"""
import sys

import pandas as pd

from src.evaluate import corrected_paired_test

args = sys.argv[1:]
path = args.pop(0) if args and args[0].endswith(".csv") else "results/runs.csv"  # Week 9
runs = pd.read_csv(path, dtype={"run": str})
if len(args) == 2:
    a = runs[runs["run"] == args[0]].iloc[0]
    b = runs[runs["run"] == args[1]].iloc[0]
else:
    a, b = runs.iloc[-1], runs.iloc[-2]

folds = [c for c in runs.columns if c.startswith("fold_")]
result = corrected_paired_test(a[folds].astype(float), b[folds].astype(float), test_train_ratio=1 / (len(folds) - 1))

print(f"Scoreboard: {path}")
for name, r in [("A", a), ("B", b)]:
    print(f"{name}: run {r['run']}  {r['model']}  feature_selection={r.get('feature_selection')}  "
          f"{r['metric']} = {r['mean']:.3f} ± {r['std']:.3f}")
print(f"\nA - B = {result['mean_diff']:+.4f}   A wins in {result['wins']} folds   "
      f"t = {result['t']:.2f}   p-value = {result['p_value']:.3f}")
print("p < 0.05: a real difference" if result["p_value"] < 0.05 else "p >= 0.05: no real difference")