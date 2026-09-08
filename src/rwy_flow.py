"""Runway flow features: trailing takeoff rates per runway/airport + config instability.
All anchored on SCHED (observable), past-only via MVT times (observable permissive).
- rwy_flow_W: takeoffs on YOUR runway in [sched-W, sched), W=15/30/60m
- apt_flow_W: takeoffs at airport in window (all rwys)
- n_rwy_active_60: distinct runways with takeoff in trailing 60m (config spread)
- rwy_share_60: your-runway share of airport flow (config preference)
"""
import numpy as np
import pandas as pd

WINDOWS = [15, 30, 60]


def _epoch(dt: pd.Series) -> np.ndarray:
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def add_rwy_flow(dep: pd.DataFrame) -> pd.DataFrame:
    for c in ["ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]:
        if c not in dep.columns:
            raise KeyError(f"add_rwy_flow missing {c}")
    s = _epoch(pd.to_datetime(dep["SCHED_TIME_UTC_mvt"]))
    t = _epoch(pd.to_datetime(dep["MVT_TIME_UTC_mvt"]))
    apt = dep["ADEP_mvt"].astype("string").fillna("MISS").to_numpy()
    rwy = dep["RUNWAY_mvt"].astype("string").fillna("MISS").to_numpy()
    key = np.array([a + "|" + r for a, r in zip(apt, rwy)])
    n = len(dep)
    cols = {}
    for w in WINDOWS:
        cols[f"rwy_flow_{w}"] = np.zeros(n, dtype=np.int32)
        cols[f"apt_flow_{w}"] = np.zeros(n, dtype=np.int32)
    cols["n_rwy_active_60"] = np.zeros(n, dtype=np.int32)
    cols["rwy_share_60"] = np.zeros(n, dtype=np.float64)
    for a in np.unique(apt):
        md = np.where(apt == a)[0]
        mt = np.where(apt == a)[0]
        tt = np.sort(t[mt])
        sd = np.argsort(s[md], kind="stable")
        sts = s[md][sd]
        for w in WINDOWS:
            lo = np.searchsorted(tt, sts - w * 60, side="left")
            hi = np.searchsorted(tt, sts, side="left")
            cols[f"apt_flow_{w}"][md[sd]] = (hi - lo).astype(np.int32)
        # per-runway flows
        for r in np.unique(rwy[md]):
            mdr = md[rwy[md] == r]
            mtr = mt[rwy[mt] == r]
            trt = np.sort(t[mtr])
            sdr = np.argsort(s[mdr], kind="stable")
            stsr = s[mdr][sdr]
            for w in WINDOWS:
                lo = np.searchsorted(trt, stsr - w * 60, side="left")
                hi = np.searchsorted(trt, stsr, side="left")
                cols[f"rwy_flow_{w}"][mdr[sdr]] = (hi - lo).astype(np.int32)
        # config spread: distinct runways active trailing 60m (event-based sweep)
        ev = np.sort(t[mt])
        evr = rwy[mt][np.argsort(t[mt], kind="stable")]
        from collections import deque
        dq: deque = deque()
        cnt: dict = {}
        j = 0
        out = np.zeros(len(sts), dtype=np.int32)
        for i in range(len(sts)):
            while j < len(ev) and ev[j] < sts[i]:
                r0 = evr[j]
                dq.append((ev[j], r0))
                cnt[r0] = cnt.get(r0, 0) + 1
                j += 1
            bound = sts[i] - 3600
            while dq and dq[0][0] < bound:
                _, r0 = dq.popleft()
                cnt[r0] -= 1
                if cnt[r0] == 0:
                    del cnt[r0]
            out[i] = len(cnt)
        cols["n_rwy_active_60"][md[sd]] = out
    a60 = cols["apt_flow_60"].astype(float)
    cols["rwy_share_60"] = np.where(a60 > 0, cols["rwy_flow_60"].astype(float) / a60, 0.0)
    return pd.DataFrame({k: v for k, v in cols.items()}, index=dep.index)


RWYC = ([f"rwy_flow_{w}" for w in WINDOWS] + [f"apt_flow_{w}" for w in WINDOWS]
        + ["n_rwy_active_60", "rwy_share_60"])
