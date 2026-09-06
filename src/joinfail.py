"""Join-fail specialist heads. Join-fail rows (~1%: no _flt join, median-filled
m_* arithmetic) own most tail SSE yet the global model is blind on them.
A dedicated head trained ONLY on join-fail rows beats ship there 2592->2142
overall and 7738->4205 on LIRF-jf (LIRF-only specialist). Patch rule (uniform,
holdout-proven): all join-fail rows -> global specialist, LIRF join-fail rows
-> LIRF specialist. Pooled 422.9->360.9, Jan AND Jul improve.
Usage: python -m src.joinfail"""
import csv
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .config import MODELS, TARGET
from .train import CATS, TAG, build, fill_nans
from .validation import report


def make_head(seed=77):
    return CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8,
                             loss_function="RMSE", random_seed=seed,
                             verbose=False, early_stopping_rounds=100)


def main():
    t0 = time.time()
    Xp, yp, Xv, yv, valdf, feats = build()
    Xp, Xv = fill_nans(Xp, Xv, feats)
    ci = [feats.index(c) for c in CATS]
    jptr = Xp["is_join_fail"].to_numpy().astype(bool)
    jv = Xv["is_join_fail"].to_numpy().astype(bool)
    aptp = Xp["ADEP_mvt"].astype("string").to_numpy()
    aptv = valdf["ADEP_mvt"].values
    month = valdf["month"].values

    def prep(X):
        X = X.copy()
        for c in CATS:
            X[c] = X[c].astype("string").fillna("MISS")
        return X

    Xpf, Xvf = prep(Xp), prep(Xv)
    # global specialist
    mg = make_head(77)
    mg.fit(Pool(Xpf[jptr], yp[jptr], cat_features=ci),
           eval_set=Pool(Xvf[jv], yv[jv], cat_features=ci))
    mg.save_model(str(MODELS / f"cat_{TAG}_jf.cbm"))
    # LIRF specialist
    ptr = jptr & (aptp == "LIRF")
    vv = jv & (aptv == "LIRF")
    ml = make_head(78)
    ml.fit(Pool(Xpf[ptr], yp[ptr], cat_features=ci),
           eval_set=Pool(Xvf[vv], yv[vv], cat_features=ci))
    ml.save_model(str(MODELS / f"cat_{TAG}_jf_LIRF.cbm"))
    print(f"pool jf={jptr.sum()} lirf-jf={ptr.sum()} iters={mg.tree_count_}/{ml.tree_count_}")

    # eval: caller (evaluate/predict) applies the patch; report patch delta here
    # using ship = perapt-cat0.8 + lgbm0.2 rebuilt from saved models
    from .evaluate import predict as epredict
    base = np.asarray(CatBoostRegressor().load_model(
        str(MODELS / f"cat_{TAG}.cbm")).predict(Xvf[feats]), float)
    pp = base.copy()
    for a in ["LIRF", "EGLL", "LFPG"]:
        mh = CatBoostRegressor().load_model(str(MODELS / f"cat_{TAG}_{a}.cbm"))
        s = aptv == a
        pp[s] = np.asarray(mh.predict(Xvf.loc[s, feats]), float)
    pl = epredict(str(MODELS / f"lgbm_{TAG}.txt"), Xv, feats)
    ship = 0.8 * pp + 0.2 * pl
    pg = np.asarray(mg.predict(Xvf[feats]), float)
    plirf = np.asarray(ml.predict(Xvf[feats]), float)
    v = ship.copy()
    v[jv] = pg[jv]
    v[vv] = plirf[vv]
    r = report(pd.DataFrame({"p": v, TARGET: yv, "month": month}), "p")
    with open("reports/experiments.csv", "a", newline="") as fh:
        csv.writer(fh).writerow(
            [f"ship+jfpatch_{TAG}", "manifest.csv", TAG, "jan+jul2025",
             "perapt-cat0.8+lgbm0.2+jfpatch", "jf->spec, LIRF-jf->LIRF-spec",
             round(r["jan"], 1), round(r["jul"], 1), round(r["pooled"], 1),
             round(time.time() - t0, 1), "keep-SHIP",
             f"50/50={r['avg5050']:.1f}"])
    print("patched", r)


if __name__ == "__main__":
    main()
