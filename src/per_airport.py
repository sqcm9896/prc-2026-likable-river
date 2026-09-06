"""Per-airport models for LIRF/EGLL/LFPG + patch-in eval. Usage:
python -m src.per_airport  (uses v1 feats; V2=1 to include congestion)"""
import csv
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .config import MODELS, TARGET
from .train import CATS, NUMS_BASE, NUMS_PERM, CG, STRICT, V2, TAG, build
from .validation import report

APTS = ["LIRF", "EGLL", "LFPG"]


def main():
    t0 = time.time()
    Xp, yp, Xv, yv, valdf, feats = build()
    # BUGFIX 2026-09-06: train.py median-fills NaN numerics before fit, but this
    # script trained on raw NaNs -> saved heads (e.g. cat_v1_LIRF.cbm) mismatch
    # filled eval features (pooled 494.3 vs base 433.6). Replicate train fill.
    meds = {}
    for c in feats:
        if Xp[c].isna().any():
            meds[c] = Xp[c].median()
            Xp[c] = Xp[c].fillna(meds[c])
            Xv[c] = Xv[c].fillna(meds[c])
    ci = [feats.index(c) for c in CATS]
    base = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8,
                             loss_function="RMSE", random_seed=42,
                             verbose=False).load_model(str(MODELS / f"cat_{TAG}.cbm"))
    pv_base = base.predict(Xv)
    from .features import load_full, add_base
    df = add_base(load_full(v2=V2))
    from .config import HOLDOUT_MONTHS
    pool_apt = df.loc[~df.month.isin(HOLDOUT_MONTHS), "ADEP_mvt"].values
    val_apt = valdf["ADEP_mvt"].values
    pv = pv_base.copy()
    for apt in APTS:
        seed = 42 + sum(ord(ch) for ch in apt) % 100  # deterministic (hash() is salted)
        m = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8,
                              loss_function="RMSE", random_seed=seed,
                              verbose=200, early_stopping_rounds=100)
        m.fit(Pool(Xp[pool_apt == apt], yp[pool_apt == apt], cat_features=ci),
              eval_set=Pool(Xv[val_apt == apt], yv[val_apt == apt], cat_features=ci))
        m.save_model(str(MODELS / f"cat_{TAG}_{apt}.cbm"))
        pv[val_apt == apt] = m.predict(Xv[val_apt == apt])
    r = report(pd.DataFrame({"p": pv, TARGET: yv, "month": valdf["month"].values}), "p")
    with open("reports/experiments.csv", "a", newline="") as fh:
        csv.writer(fh).writerow([f"cat_{TAG}+perapt", "manifest.csv", TAG,
                                 "jan+jul2025", "catboost-perapt", "LIRF/EGLL/LFPG heads",
                                 round(r["jan"], 1), round(r["jul"], 1),
                                 round(r["pooled"], 1), round(time.time() - t0, 1), "?",
                                 f"50/50={r['avg5050']:.1f}"])
    print("patched", r)


if __name__ == "__main__":
    main()
