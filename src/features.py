"""S3: deterministic feature builder v1 (F1+F2+F3+F4). OOF smoothed target means
fit on pool only. `STRICT=1` env drops F4 permissive arithmetic (fallback)."""
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .config import DATA, HOLDOUT_MONTHS, SEED, TARGET

STRICT = os.environ.get("STRICT", "0") == "1"
M_SMOOTH = 50  # smoothing mass for target means
MIN_COUNT = 20  # RARE threshold

FEAT_COLS = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt",
             "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
             "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
             "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
             "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "LOBT_flt", "IOBT_flt",
             "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt", "BLOCK_TIME_UTC_mvt",
             "PHASE_mvt", TARGET]

OOF_COLS = ["RUNWAY_mvt", "STAND_ZONE", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
            "ADES_mvt", "ADEP_mvt"]


def stand_zone(s, adep=None):
    """Terminal/apron zone from STAND. Airport-aware: leading letters (EDDF 'B41'->B)
    else first digit (EGLL '537'->5, LIRF '409'->4); prefixed with airport to avoid
    cross-airport collisions (EGLL_5). Fixes 100% UNK at EGLL/LIRF (review 2026-09-06)."""
    s = s.astype("string")
    alpha = s.str.extract(r"^([A-Z]+)", expand=False)
    digit = s.str.extract(r"^(\d)", expand=False)
    zone = alpha.fillna(digit).fillna("UNK")
    if adep is not None:
        zone = adep.astype("string").fillna("UNK") + "_" + zone
    return zone.fillna("UNK")


def add_base(df):
    df = df.copy()
    df["month"] = df["SCHED_TIME_UTC_mvt"].dt.month
    df["dow"] = df["SCHED_TIME_UTC_mvt"].dt.dayofweek
    df["hour"] = df["SCHED_TIME_UTC_mvt"].dt.hour
    df["sin_h"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_h"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_d"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_d"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype("int8")
    df["m_hour"] = df["MVT_TIME_UTC_mvt"].dt.hour  # clock only
    df["m_dow"] = df["MVT_TIME_UTC_mvt"].dt.dayofweek
    df["STAND_ZONE"] = stand_zone(df["STAND_mvt"].astype("string"), df["ADEP_mvt"])
    df["is_join_fail"] = df["AOBT_3_flt"].isna().astype("int8")
    df["diverted"] = (df["ADES_flt"].notna() & df["ADES_FILED_flt"].notna()
                      & (df["ADES_flt"] != df["ADES_FILED_flt"])).astype("int8")
    df["adep_agree"] = (df["ADEP_mvt"] == df["ADEP_flt"]).astype("int8")
    for a, b, n in [("SCHED_TIME_UTC_mvt", "LOBT_flt", "sched_lobt"),
                    ("SCHED_TIME_UTC_mvt", "IOBT_flt", "sched_iobt"),
                    ("SCHED_TIME_UTC_mvt", "EOBT_1_flt", "sched_eobt"),
                    ("EOBT_1_flt", "IOBT_flt", "eobt_iobt")]:
        df[n] = (df[a] - df[b]).dt.total_seconds()
    if not STRICT:
        for a, b, n in [("MVT_TIME_UTC_mvt", "AOBT_3_flt", "m_aobt"),
                        ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "m_sched"),
                        ("AOBT_3_flt", "SCHED_TIME_UTC_mvt", "aobt_sched"),
                        ("MVT_TIME_UTC_mvt", "LOBT_flt", "m_lobt"),
                        ("MVT_TIME_UTC_mvt", "IOBT_flt", "m_iobt"),
                        ("MVT_TIME_UTC_mvt", "EOBT_1_flt", "m_eobt"),
                        ("ARVT_1_flt", "MVT_TIME_UTC_mvt", "arvt1_m")]:
            df[n] = (df[a] - df[b]).dt.total_seconds()
    return df


def fit_oof(pool, seed=SEED):
    """K-fold OOF smoothed means + full-pool maps + global mean."""
    rng = np.random.RandomState(seed)
    folds = rng.randint(0, 5, size=len(pool))
    global_mean = float(pool[TARGET].mean())
    oof = pd.DataFrame(index=pool.index)
    maps, counts = {}, {}
    for c in OOF_COLS:
        col_oof = np.full(len(pool), np.nan)
        for k in range(5):
            tr, va = folds != k, folds == k
            st = pool.loc[tr].groupby(pool.loc[tr, c], observed=True)[TARGET].agg(["mean", "size"])
            enc = (st["mean"] * st["size"] + global_mean * M_SMOOTH) / (st["size"] + M_SMOOTH)
            col_oof[va] = pool.loc[va, c].map(enc).fillna(global_mean).to_numpy()
        oof[c + "_te"] = col_oof
        st = pool.groupby(c, observed=True)[TARGET].agg(["mean", "size"])
        maps[c] = (st["mean"] * st["size"] + global_mean * M_SMOOTH) / (st["size"] + M_SMOOTH)
        counts[c] = st["size"]
        oof[c + "_logn"] = np.log1p(pool[c].map(st["size"]).fillna(0))
    return oof, maps, counts, global_mean


def apply_maps(df, maps, counts, global_mean):
    out = pd.DataFrame(index=df.index)
    for c in OOF_COLS:
        out[c + "_te"] = df[c].map(maps[c]).fillna(global_mean)
        out[c + "_logn"] = np.log1p(df[c].map(counts[c]).fillna(0))
    return out


def load_full(v2=False):
    d = ds.dataset(sorted(glob.glob(str(DATA / "training_*.parquet"))), format="parquet")
    cols = FEAT_COLS + (["ADES_mvt", "RUNWAY_mvt", "SCHED_TIME_UTC_mvt"] if False else [])
    cols = list(dict.fromkeys(FEAT_COLS + ["ADES_mvt", "RUNWAY_mvt"]))
    filt = None if v2 else ds.field("PHASE_mvt") == "DEP"
    df = d.to_table(columns=cols, filter=filt).to_pandas()
    if v2:
        from .congestion import add_congestion
        df = add_congestion(df)
        df = df[df.PHASE_mvt == "DEP"].reset_index(drop=True)
    df["month"] = df["BLOCK_TIME_UTC_mvt"].dt.month
    return df
