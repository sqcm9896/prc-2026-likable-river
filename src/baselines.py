"""S2: median baselines -> reports/experiments.csv. Usage: python -m src.baselines"""
import csv
import os
import time

import pandas as pd

from .validation import load_dep_dep, report
from .config import HOLDOUT_MONTHS

df = load_dep_dep()
pool, val = df[~df.month.isin(HOLDOUT_MONTHS)].copy(), df[df.month.isin(HOLDOUT_MONTHS)].copy()
val["p_global"] = pool["TAXITIME_SEC_mvt"].median()
val["p_airport"] = val["ADEP_mvt"].map(pool.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].median())
ah = pool.groupby(["ADEP_mvt", "sched_h"])["TAXITIME_SEC_mvt"].median()
val["p_ah"] = val.set_index(["ADEP_mvt", "sched_h"]).index.map(ah)
val["p_ah"] = val["p_ah"].fillna(val["p_airport"]).to_numpy()

new = not os.path.exists("reports/experiments.csv")
with open("reports/experiments.csv", "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["run_id", "data_manifest", "feature_version", "split", "model",
                    "parameters", "RMSE_jan", "RMSE_jul", "combined_RMSE", "runtime",
                    "decision", "notes"])
    for rid, col in [("base_global", "p_global"), ("base_airport", "p_airport"),
                     ("base_airport_hour", "p_ah")]:
        t0 = time.time()
        r = report(val, col)
        w.writerow([rid, "manifest.csv", "n/a", "jan+jul2025", "median", col,
                    round(r["jan"], 1), round(r["jul"], 1), round(r["pooled"], 1),
                    round(time.time() - t0, 1), "keep",
                    f"50/50={r['avg5050']:.1f}"])
        print(rid, r)
print("logged to reports/experiments.csv")
