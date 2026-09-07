"""S3: train CatBoost (RMSE) + LightGBM (huber) on v1, eval holdout. Usage:
python -m src.train  (STRICT=1 for fallback; MODEL=cat|lgbm|both)"""
import csv
import os
import time

import numpy as np
import pandas as pd

from .config import HOLDOUT_MONTHS, MODELS, TARGET
from .features import OOF_COLS, add_base, apply_maps, fit_oof, load_full
from .validation import report

CATS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_ZONE", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
        "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
        "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "ADES_mvt"]
NUMS_BASE = ["month", "dow", "hour", "sin_h", "cos_h", "sin_d", "cos_d",
             "is_weekend", "is_night", "m_hour", "m_dow",
             "sched_lobt", "sched_iobt", "sched_eobt", "eobt_iobt",
             "is_join_fail", "diverted", "adep_agree"]
NUMS_PERM = ["m_aobt", "m_sched", "aobt_sched", "m_lobt", "m_iobt", "m_eobt", "arvt1_m"]
STRICT = os.environ.get("STRICT", "0") == "1"
V2 = os.environ.get("V2", "0") == "1"
TAG = "v2" if V2 else "v1"
if STRICT:
    TAG += "_strict"
TAG += os.environ.get("SUFFIX", "")  # e.g. SUFFIX=_d10 for tuning runs

CG = ["cg_tot_15", "cg_dep_15", "cg_arr_15", "cg_tot_30", "cg_dep_30", "cg_arr_30",
      "cg_tot_60", "cg_dep_60", "cg_arr_60", "cg_tot_120", "cg_dep_120", "cg_arr_120",
      "cg_rwy_60"] + ["log_cg_tot_15", "log_cg_dep_15", "log_cg_arr_15", "log_cg_tot_30",
      "log_cg_dep_30", "log_cg_arr_30", "log_cg_tot_60", "log_cg_dep_60", "log_cg_arr_60",
      "log_cg_tot_120", "log_cg_dep_120", "log_cg_arr_120", "log_cg_rwy_60",
      "cg_mvt_dep_30", "cg_mvt_arr_30", "cg_mvt_dep_60", "cg_mvt_arr_60",
      "log_cg_mvt_dep_30", "log_cg_mvt_arr_30", "log_cg_mvt_dep_60", "log_cg_mvt_arr_60"]


QX = ["nq_takeoff", "nq_takeoff_rwy", "nq_pushback", "wip_at_pushback",
      "log_nq_takeoff", "log_nq_takeoff_rwy", "log_nq_pushback", "log_wip_at_pushback"]
QEXP = os.environ.get("QEXP", "0") == "1"  # experienced-queue features (src/queue_exp.py)


def build():
    df = load_full(v2=V2)
    df = add_base(df)
    if QEXP:
        from .queue_exp import add_queue_exp
        qx = add_queue_exp(df).reset_index(drop=True)
        df = df.reset_index(drop=True)
    pool = df[~df.month.isin(HOLDOUT_MONTHS)].reset_index(drop=True)
    val = df[df.month.isin(HOLDOUT_MONTHS)].reset_index(drop=True)
    oof, maps, counts, gmean = fit_oof(pool)
    te_val = apply_maps(val, maps, counts, gmean)
    te_cols = [c + "_te" for c in OOF_COLS] + [c + "_logn" for c in OOF_COLS]
    nums = NUMS_BASE + ([] if STRICT else NUMS_PERM) + (CG if V2 else [])
    feats = CATS + nums + te_cols + (QX if QEXP else [])
    Xp = pd.concat([pool[CATS + nums].reset_index(drop=True), oof[te_cols].reset_index(drop=True)], axis=1)
    Xv = pd.concat([val[CATS + nums].reset_index(drop=True), te_val[te_cols].reset_index(drop=True)], axis=1)
    if QEXP:
        pmask = ~df.month.isin(HOLDOUT_MONTHS).values
        Xp = pd.concat([Xp, qx[pmask].reset_index(drop=True)], axis=1)
        Xv = pd.concat([Xv, qx[~pmask].reset_index(drop=True)], axis=1)
    for c in CATS:
        Xp[c] = Xp[c].astype("string").fillna("MISS")
        Xv[c] = Xv[c].astype("string").fillna("MISS")
    return Xp, pool[TARGET].to_numpy(float), Xv, val[TARGET].to_numpy(float), val, feats


def fill_nans(Xp, Xv, feats):
    Xp, Xv = Xp.copy(), Xv.copy()
    for c in feats:
        if Xp[c].isna().any():
            med = Xp[c].median()
            Xp[c] = Xp[c].fillna(med)
            Xv[c] = Xv[c].fillna(med)
    return Xp, Xv


def main():
    which = os.environ.get("MODEL", "both")
    t0 = time.time()
    Xp, yp, Xv, yv, valdf, feats = build()
    print(f"pool={len(Xp)} val={len(Xv)} feats={len(feats)} strict={STRICT} tag={TAG}")
    Xp, Xv = fill_nans(Xp, Xv, feats)
    results = {}
    if which in ("cat", "both"):
        from catboost import CatBoostRegressor, Pool
        ci = [feats.index(c) for c in CATS]
        m = CatBoostRegressor(iterations=int(os.environ.get("CAT_ITERS", "1500")), learning_rate=float(os.environ.get("CAT_LR", "0.05")), depth=int(os.environ.get("CAT_DEPTH", "8")),
                               loss_function="RMSE", random_seed=int(os.environ.get("CAT_SEED", "42")), subsample=float(os.environ.get("CAT_SUBSAMPLE", "1.0")), verbose=200,
                               early_stopping_rounds=100)
        m.fit(Pool(Xp, yp, cat_features=ci), eval_set=Pool(Xv, yv, cat_features=ci))
        pv = m.predict(Xv)
        m.save_model(str(MODELS / f"cat_{TAG}.cbm"))
        results["catboost"] = (pv, dict(iterations=m.tree_count_, lr=float(os.environ.get("CAT_LR", "0.05")), depth=int(os.environ.get("CAT_DEPTH", "8")), seed=int(os.environ.get("CAT_SEED", "42")), subsample=float(os.environ.get("CAT_SUBSAMPLE", "1.0"))))
    if which in ("lgbm", "both"):
        import lightgbm as lgb
        for c in CATS:
            Xp[c] = Xp[c].astype("category")
            Xv[c] = Xv[c].astype("category")
        dtr = lgb.Dataset(Xp, yp)
        dval = lgb.Dataset(Xv, yv, reference=dtr)
        params = {"objective": os.environ.get("LGBM_OBJ", "huber"), "alpha": 0.9, "learning_rate": float(os.environ.get("LGBM_LR", "0.05")),
                  "num_leaves": int(os.environ.get("LGBM_LEAVES", "127")), "min_data_in_leaf": int(os.environ.get("LGBM_MIN_DATA", "500")), "verbosity": -1, "seed": int(os.environ.get("LGBM_SEED", "42")),
                  "bagging_fraction": float(os.environ.get("LGBM_BAG", "1.0")), "bagging_freq": 1 if float(os.environ.get("LGBM_BAG", "1.0")) < 1.0 else 0,
                  "feature_fraction": float(os.environ.get("LGBM_FEAT", "1.0"))}
        m = lgb.train(params, dtr, int(os.environ.get("LGBM_ITERS", "1500")), valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        pv = m.predict(Xv)
        m.save_model(str(MODELS / f"lgbm_{TAG}.txt"))
        results["lightgbm-"+os.environ.get("LGBM_OBJ","huber")] = (pv, params)
    with open("reports/experiments.csv", "a", newline="") as fh:
        w = csv.writer(fh)
        for name, (pv, prm) in results.items():
            r = report(pd.DataFrame({"p": pv, TARGET: yv,
                                     "month": valdf["month"].values}), "p")
            tag = TAG
            w.writerow([f"{name}_{tag}", "manifest.csv", tag, "jan+jul2025", name,
                        str(prm)[:200], round(r["jan"], 1), round(r["jul"], 1),
                        round(r["pooled"], 1), round(time.time() - t0, 1), "?",
                        f"50/50={r['avg5050']:.1f}"])
            print(name, tag, r)
            e = pd.DataFrame({"e": (pv - yv) ** 2, "ADEP_mvt": valdf["ADEP_mvt"].values,
                              "month": valdf["month"].values})
            print(e.groupby("ADEP_mvt")["e"].mean().pow(0.5).round(1)
                  .sort_values(ascending=False).to_string())
            imp = None
    print("done")


if __name__ == "__main__":
    main()
