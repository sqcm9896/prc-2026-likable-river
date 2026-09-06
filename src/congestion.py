"""v2: chronological trailing congestion counts (SCHED-anchored, strictly past).
No targets used; same code runs on train and ranking. Windows: 15/30/60/120 min
per airport (dep/arr/total) + per runway (dep)."""
import numpy as np
import pandas as pd

WINDOWS = [15, 30, 60, 120]


def add_congestion(df_all):
    df = df_all.copy()
    df["apt"] = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    df["sched_i"] = df["SCHED_TIME_UTC_mvt"].astype("int64") // 10**9
    df["mvt_i"] = df["MVT_TIME_UTC_mvt"].astype("int64") // 10**9
    df["is_dep"] = (df["PHASE_mvt"] == "DEP").astype("int8")
    df = df.sort_values(["apt", "sched_i"]).reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    for apt, g in df.groupby("apt", sort=False):
        t = g["sched_i"].to_numpy()
        isd = g["is_dep"].to_numpy()
        cs_all = np.concatenate([[0], np.cumsum(np.ones(len(t)))])
        cs_dep = np.concatenate([[0], np.cumsum(isd)])
        cs_arr = cs_all - cs_dep
        for w in WINDOWS:
            lo = np.searchsorted(t, t - w * 60, side="left")
            hi = np.arange(len(t))
            out.loc[g.index, f"cg_tot_{w}"] = cs_all[hi + 1] - cs_all[lo] - 1
            out.loc[g.index, f"cg_dep_{w}"] = cs_dep[hi + 1] - cs_dep[lo] - isd
            out.loc[g.index, f"cg_arr_{w}"] = cs_arr[hi + 1] - cs_arr[lo] - (1 - isd)
        # MVT-anchored flow (takeoffs/landings already happened -> knowable; permissive)
        g2 = g.sort_values("mvt_i")
        tm = g2["mvt_i"].to_numpy()
        dm = g2["is_dep"].to_numpy()
        csd = np.concatenate([[0], np.cumsum(dm)])
        csa = np.concatenate([[0], np.cumsum(1 - dm)])
        for w in [30, 60]:
            lo = np.searchsorted(tm, tm - w * 60, side="left")
            hi = np.arange(len(tm))
            out.loc[g2.index, f"cg_mvt_dep_{w}"] = csd[hi + 1] - csd[lo] - dm
            out.loc[g2.index, f"cg_mvt_arr_{w}"] = csa[hi + 1] - csa[lo] - (1 - dm)
        rws = g["RUNWAY_mvt"].astype("string").fillna("MISS").to_numpy()
        for rw in np.unique(rws):
            m = (rws == rw)
            tm = t[m]
            dm = isd[m]
            cs = np.concatenate([[0], np.cumsum(dm)])
            lo = np.searchsorted(tm, tm - 3600, side="left")
            idx = np.where(m)[0]
            out.loc[g.index[idx], "cg_rwy_60"] = cs[np.arange(len(tm)) + 1] - cs[lo] - dm
    out["cg_rwy_60"] = out["cg_rwy_60"].fillna(0)
    for c in out.columns:
        out["log_" + c] = np.log1p(out[c].clip(lower=0))
    df = pd.concat([df, out], axis=1)
    return df.sort_index()
