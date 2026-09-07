"""AOBT-anchored pushback queue: true pushback sequence counts.

Counts prior departures by actual off-block (AOBT_3_flt, fallback SCHED when
null) in trailing 15/30/60-min windows, per airport, strictly past-only
(excludes self). Same code runs on train and ranking (no targets, no future).
"""
import numpy as np
import pandas as pd

WINDOWS = [15, 30, 60]


def add_queue(dep: pd.DataFrame) -> pd.DataFrame:
    req = ["ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt"]
    miss = [c for c in req if c not in dep.columns]
    if miss:
        raise KeyError(f"add_queue missing cols: {miss}")
    is_missing = dep["AOBT_3_flt"].isna().to_numpy()
    aobt = pd.to_datetime(dep["AOBT_3_flt"])
    sched = pd.to_datetime(dep["SCHED_TIME_UTC_mvt"])
    fallback = aobt.where(aobt.notna(), sched)
    # unit-agnostic epoch seconds (parquet is datetime64[us]; astype int64 //1e9
    # assumes ns and yields kiloseconds -> 10-day windows; avoid that here)
    t_float = (fallback - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    still_nat = np.isnan(t_float)
    t_sec = np.where(still_nat, 0, t_float).astype(np.int64)
    t_sec[still_nat] = 0
    apt = dep["ADEP_mvt"].astype("string").fillna("MISS").to_numpy()
    n = len(dep)
    q15 = np.zeros(n, dtype=np.int32)
    q30 = np.zeros(n, dtype=np.int32)
    q60 = np.zeros(n, dtype=np.int32)
    # per-airport trailing counts via searchsorted (past-only, exclude self)
    for a in np.unique(apt):
        m = np.where(apt == a)[0]
        tm = t_sec[m]
        order = np.argsort(tm, kind="stable")
        ts = tm[order]
        pos = np.arange(len(ts))
        for w, dest in ((15, q15), (30, q30), (60, q60)):
            lo = np.searchsorted(ts, ts - w * 60, side="left")
            cnt = (pos - lo).astype(np.int32)
            dest[m[order]] = cnt
    out = pd.DataFrame(index=dep.index)
    out["q_aobt_15"] = q15
    out["q_aobt_30"] = q30
    out["q_aobt_60"] = q60
    out["log_q_aobt_15"] = np.log1p(q15.astype(float))
    out["log_q_aobt_30"] = np.log1p(q30.astype(float))
    out["log_q_aobt_60"] = np.log1p(q60.astype(float))
    out["is_aobt_missing"] = is_missing.astype("int8")
    return out
