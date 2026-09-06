"""Shared holdout evaluator: builds v1 features ONCE (with train.py median-fill),
scores any saved models, sweeps blends + clip policies. Usage:
python -m src.evaluate models/cat_v1.cbm models/lgbm_v1.txt [more...]
"""
import sys

import numpy as np
import pandas as pd

from .config import TARGET
from .train import CATS, build
from .validation import report


def load_filled():
    Xp, yp, Xv, yv, valdf, feats = build()
    for c in feats:
        if Xv[c].isna().any() and pd.api.types.is_numeric_dtype(Xv[c]):
            Xv[c] = Xv[c].fillna(Xp[c].median())
    return Xp, yp, Xv, yv, valdf, feats


def predict(path, Xv, feats):
    if path.endswith(".cbm"):
        from catboost import CatBoostRegressor
        X = Xv.copy()
        for c in CATS:
            X[c] = X[c].astype("string").fillna("MISS")
        return np.asarray(CatBoostRegressor().load_model(path).predict(X[feats]), float)
    if path.endswith(".txt"):
        import lightgbm as lgb
        X = Xv.copy()
        for c in CATS:
            X[c] = X[c].astype("category")
        return np.asarray(lgb.Booster(model_file=path).predict(X[feats]), float)
    raise ValueError(path)


def show(name, p, yv, month, apt):
    p = np.asarray(p, float)
    r = report(pd.DataFrame({"p": p, TARGET: yv, "month": month}), "p")
    s = pd.DataFrame({"s": (p - yv) ** 2, "apt": apt}).groupby("apt")["s"].mean().pow(0.5)
    top = s.sort_values(ascending=False).head(3).round(1).to_dict()
    print(f"{name}: jan={r['jan']:.1f} jul={r['jul']:.1f} pooled={r['pooled']:.1f} "
          f"50/50={r['avg5050']:.1f} worst={top}", flush=True)
    return r


def main(paths):
    Xp, yp, Xv, yv, valdf, feats = load_filled()
    print(f"pool={len(Xp)} val={len(Xv)} feats={len(feats)}", flush=True)
    month, apt = valdf["month"].values, valdf["ADEP_mvt"].values
    preds = {}
    for path in paths:
        preds[path] = predict(path, Xv, feats)
        show(path, preds[path], yv, month, apt)
    if len(preds) == 2:
        (a, pa), (b, pb) = list(preds.items())
        for w in (0.5, 0.6, 0.7, 0.8):
            show(f"blend {a}x{w}+{b}x{1 - w:.1f}", w * pa + (1 - w) * pb, yv, month, apt)
    out = "/tmp/eval_preds.npz"
    np.savez(out, **{f"m{i}": v for i, v in enumerate(preds.values())},
             yv=yv, month=month, apt=np.array(apt))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
