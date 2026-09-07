"""Wake-vortex separation pressure: heavies (H/J) that took off ahead of you force
ATC spacing holds. Counts trailing departures with WK_TBL_CAT_flt in {H, J}
in 15/30-min MVT-anchored windows per airport and per runway (past-only,
same code runs on train and ranking). Permissive lineage only (uses MVT clock
and _flt wake cats)."""
import numpy as np
import pandas as pd

HEAVY = {"H", "J"}
WINDOWS = [15, 30]


def add_wake(dep):
    """dep: DEP-only frame with ADEP_mvt, RUNWAY_mvt, MVT_TIME_UTC_mvt,
    WK_TBL_CAT_flt. Returns DataFrame aligned to dep.index."""
    df = dep.copy()
    df["mvt_i"] = df["MVT_TIME_UTC_mvt"].astype("int64") // 10**9
    df["is_heavy"] = df["WK_TBL_CAT_flt"].astype("string").isin(HEAVY).astype("int8")
    df["rwy"] = df["RUNWAY_mvt"].astype("string").fillna("MISS")
    out = pd.DataFrame(index=df.index)
    out["wk_heavy_15"] = 0
    out["wk_heavy_30"] = 0
    out["wk_heavy_rwy_15"] = 0
    out["wk_leader_heavy"] = 0
    for apt, g in df.groupby("ADEP_mvt", sort=False):
        o = g.sort_values("mvt_i")
        t = o["mvt_i"].to_numpy()
        h = o["is_heavy"].to_numpy()
        csh = np.concatenate([[0], np.cumsum(h)])
        for w in WINDOWS:
            lo = np.searchsorted(t, t - w * 60, side="left")
            hi = np.arange(len(t))
            out.loc[o.index, f"wk_heavy_{w}"] = csh[hi + 1] - csh[lo] - h
        out.loc[o.index, "wk_leader_heavy"] = np.concatenate([[0], h[:-1]])
        for rw, g2 in o.groupby(o["rwy"].values, sort=False):
            t2 = g2["mvt_i"].to_numpy()
            h2 = g2["is_heavy"].to_numpy()
            cs = np.concatenate([[0], np.cumsum(h2)])
            lo = np.searchsorted(t2, t2 - 15 * 60, side="left")
            hi = np.arange(len(t2))
            out.loc[g2.index, "wk_heavy_rwy_15"] = cs[hi + 1] - cs[lo] - h2
    for c in ["wk_heavy_15", "wk_heavy_30", "wk_heavy_rwy_15"]:
        out["log_" + c] = np.log1p(out[c].clip(lower=0))
    return out.astype("float64")
