"""
Compares two runs of the scoreboard (results/runs.csv) with the corrected paired t-test.
    python compare_runs.py                    -> the last two runs
    python compare_runs.py <run_a> <run_b>    -> two runs by their id (the "run" column)
"""
import sys

import pandas as pd

from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv", dtype={"run": str})
if len(sys.argv) == 3:
    a = runs[runs["run"] == sys.argv[1]].iloc[0]
    b = runs[runs["run"] == sys.argv[2]].iloc[0]
else:
    a, b = runs.iloc[-1], runs.iloc[-2]

folds = [c for c in runs.columns if c.startswith("fold_")]
result = corrected_paired_test(a[folds].astype(float), b[folds].astype(float), test_train_ratio=1 / (len(folds) - 1))

for name, r in [("A", a), ("B", b)]:
    print(f"{name}: run {r['run']}  {r['model']}  feature_selection={r.get('feature_selection')}  "
          f"{r['metric']} = {r['mean']:.3f} ± {r['std']:.3f}")
print(f"\nA - B = {result['mean_diff']:+.4f}   A wins in {result['wins']} folds   "
      f"t = {result['t']:.2f}   p-value = {result['p_value']:.3f}")
print("p < 0.05: a real difference" if result["p_value"] < 0.05 else "p >= 0.05: no real difference")