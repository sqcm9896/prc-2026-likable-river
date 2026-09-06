"""Blend saved holdout predictions. Usage: python -m src.ensemble_eval"""
import csv
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb

from .config import MODELS, TARGET
from .train import CATS, build, fill_nans
from .validation import report

t0 = time.time()
Xp, yp, Xv, yv, valdf, feats = build()
Xp, Xv = fill_nans(Xp, Xv, feats)
cat = CatBoostRegressor().load_model(str(MODELS / "cat_v1.cbm"))
lgbm = lgb.Booster(model_file=str(MODELS / "lgbm_v1.txt"))
Xl = Xv.copy()
for c in CATS:
    Xl[c] = Xl[c].astype("category")
pc, pl = cat.predict(Xv), lgbm.predict(Xl)
base = {"month": valdf["month"].values, TARGET: yv}
best = None
for w in [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 1.0]:
    r = report(pd.DataFrame({"p": w * pc + (1 - w) * pl, **base}), "p")
    print(f"w_cat={w}: pooled={r['pooled']:.1f} jan={r['jan']:.1f} jul={r['jul']:.1f}")
    if best is None or r["pooled"] < best[1]:
        best = (w, r["pooled"])
print("best:", best, f"{time.time()-t0:.0f}s")
