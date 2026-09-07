"""Final: refit on full 2025 DEP, predict ranking DEP, validate, save.
Usage: MODEL=cat|lgbm|both, W_CAT blend weight, PERAPT=1 for LIRF/EGLL/LFPG
heads patched onto cat preds. Output: submissions/likable-river_v<N>.parquet"""
import csv
import datetime
import glob
import hashlib
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .config import DATA, MODELS, SUBMISSIONS, TARGET
from .features import FEAT_COLS, OOF_COLS, add_base, apply_maps, fit_oof
from .submit import validate
from .train import CATS, CG, NUMS_BASE, NUMS_PERM, STRICT, V2, fill_nans

TAG = ("v2" if V2 else "v1") + ("_strict" if STRICT else "") + "_full"
N_VER = int(os.environ.get("SUB_VER", "1"))
W_CAT = float(os.environ.get("W_CAT", "0.7"))


def load_ranking():
    cols = list(dict.fromkeys(FEAT_COLS + ["ADES_mvt", "RUNWAY_mvt", "MVT_ID_mvt"]))
    d = ds.dataset([str(DATA / "ranking.parquet")], format="parquet")
    if V2:
        df = d.to_table(columns=cols).to_pandas()
        from .congestion import add_congestion
        df = add_congestion(df)
    else:
        df = d.to_table(columns=cols, filter=ds.field("PHASE_mvt") == "DEP").to_pandas()
        df = df[df.PHASE_mvt == "DEP"].reset_index(drop=True)
    return df


def main():
    from .features import load_full
    print("loading train + ranking...")
    train = add_base(load_full(v2=V2))
    rk = add_base(load_ranking())
    rk_dep = rk[rk.PHASE_mvt == "DEP"].reset_index(drop=True) if V2 else rk
    ids = rk_dep["MVT_ID_mvt"].to_numpy()
    print(f"train={len(train)} rankDEP={len(rk_dep)}")
    oof, maps, counts, gmean = fit_oof(train)
    te_tr = pd.DataFrame({c + "_te": np.full(len(train), np.nan) for c in OOF_COLS} |
                        {c + "_logn": np.full(len(train), np.nan) for c in OOF_COLS})
    # in-sample encodings for train: use OOF values where available is overkill;
    # refit maps on full train = maps (same); use maps directly (tiny optimistic bias, final fit)
    te_tr = apply_maps(train, maps, counts, gmean)
    te_rk = apply_maps(rk_dep, maps, counts, gmean)
    te_cols = [c + "_te" for c in OOF_COLS] + [c + "_logn" for c in OOF_COLS]
    nums = NUMS_BASE + ([] if STRICT else NUMS_PERM) + (CG if V2 else [])
    feats = CATS + nums + te_cols
    Xtr = pd.concat([train[CATS + nums].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
    Xrk = pd.concat([rk_dep[CATS + nums].reset_index(drop=True), te_rk.reset_index(drop=True)], axis=1)
    for c in CATS:
        Xtr[c] = Xtr[c].astype("string").fillna("MISS")
        Xrk[c] = Xrk[c].astype("string").fillna("MISS")
    Xtr, Xrk = fill_nans(Xtr, Xrk, feats)
    ytr = train[TARGET].to_numpy(float)
    which = os.environ.get("MODEL", "both")
    preds = {}
    if which in ("cat", "both"):
        from catboost import CatBoostRegressor, Pool
        ci = [feats.index(c) for c in CATS]
        m = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8,
                              loss_function="RMSE", random_seed=42, verbose=200)
        m.fit(Pool(Xtr, ytr, cat_features=ci))
        m.save_model(str(MODELS / f"cat_{TAG}.cbm"))
        preds["cat"] = np.asarray(m.predict(Xrk), float)
        if os.environ.get("PERAPT", "0") == "1":
            # holdout-proven: dedicated heads for tail airports (v1: 433.6->423.7)
            tr_apt = train["ADEP_mvt"].values
            rk_apt = rk_dep["ADEP_mvt"].values
            pc = preds["cat"].copy()
            # refit caps = holdout optima (review 2026-09-06; LFPG overfits past ~700)
            for a, cap in [("LIRF", 1500), ("EGLL", 1500), ("LFPG", 700)]:
                h = CatBoostRegressor(iterations=cap, learning_rate=0.05, depth=8,
                                      loss_function="RMSE", random_seed=42, verbose=200)
                h.fit(Pool(Xtr[tr_apt == a], ytr[tr_apt == a], cat_features=ci))
                h.save_model(str(MODELS / f"cat_{TAG}_{a}.cbm"))
                pc[rk_apt == a] = h.predict(Xrk[rk_apt == a])
            preds["cat"] = pc
            print("per-airport heads patched: LIRF/EGLL/LFPG")
        if os.environ.get("JFPATCH", "0") == "1":
            # holdout-proven: join-fail specialists (ship 422.9->359.2).
            # refit iters from holdout optima (global ~360-410, LIRF ~513-889).
            jtr = train["AOBT_3_flt"].isna().to_numpy()
            jrk = rk_dep["AOBT_3_flt"].isna().to_numpy()
            tr_apt = train["ADEP_mvt"].values
            rk_apt = rk_dep["ADEP_mvt"].values
            pj = preds["cat"].copy()
            for nm, mask_tr, iters, seed in [
                    ("jf", jtr, 360, 77),
                    ("jf_LIRF", jtr & (tr_apt == "LIRF"), 890, 78)]:
                h = CatBoostRegressor(iterations=iters, learning_rate=0.05, depth=8,
                                      loss_function="RMSE", random_seed=seed, verbose=200)
                h.fit(Pool(Xtr[mask_tr], ytr[mask_tr], cat_features=ci))
                h.save_model(str(MODELS / f"cat_{TAG}_{nm}.cbm"))
            hg = CatBoostRegressor().load_model(str(MODELS / f"cat_{TAG}_jf.cbm"))
            hl = CatBoostRegressor().load_model(str(MODELS / f"cat_{TAG}_jf_LIRF.cbm"))
            pj[jrk] = hg.predict(Xrk[jrk])
            pj[jrk & (rk_apt == "LIRF")] = hl.predict(Xrk[jrk & (rk_apt == "LIRF")])
            preds["cat"] = pj
            print(f"join-fail specialists patched: {jrk.sum()} rows "
                  f"({(jrk & (rk_apt == 'LIRF')).sum()} LIRF)")
    if which in ("lgbm", "both"):
        import lightgbm as lgb
        Xl, Xr = Xtr.copy(), Xrk.copy()
        for c in CATS:
            Xl[c] = Xl[c].astype("category")
            Xr[c] = Xr[c].astype("category")
        params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 127,
                  "min_data_in_leaf": 500, "verbosity": -1, "seed": 42}
        m = lgb.train(params, lgb.Dataset(Xl, ytr), 1500)
        m.save_model(str(MODELS / f"lgbm_{TAG}.txt"))
        preds["lgbm"] = m.predict(Xr)
    p = W_CAT * preds.get("cat", 0) + (1 - W_CAT) * preds.get("lgbm", 0) \
        if len(preds) == 2 else next(iter(preds.values()))
    p = np.clip(np.asarray(p, float), 0, None)
    out = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": p})
    path = SUBMISSIONS / f"likable-river_v{N_VER}.parquet"
    out.to_parquet(path, index=False)
    validate(str(path))
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    with open(SUBMISSIONS / "manifest.csv", "a", newline="") as fh:
        csv.writer(fh).writerow([path.name, md5, datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 TAG, f"blend w_cat={W_CAT}", len(out),
                                 f"mean={p.mean():.0f} p99={np.quantile(p, .99):.0f}"])
    print(f"SAVED {path} md5={md5}")


if __name__ == "__main__":
    main()
