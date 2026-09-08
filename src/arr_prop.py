"""ARR propagation features: late-inbound cascade signals for DEP rows.

For each DEP row (airport X, anchor = SCHED_TIME_UTC_mvt, always observed):
per-airport ARR landings with landing time in [anchor-W, anchor), W in 30/60/120m:
  n_arr_W, mean/max(arr_delay=MVT-SCHED)_W, n_late15(>900s)_W.
Past-only, sched-anchored (no targets, no future, no DEP clock input).
Same code runs on train and ranking (ARR fully observed in both).
"""
import numpy as np
import pandas as pd
from collections import deque

WINDOWS = [30, 60, 120]

ARRC = [f"{p}_{w}" for w in WINDOWS
        for p in ("n_arr", "mean_adly", "max_adly", "n_late15")]


def _epoch(dt: pd.Series) -> np.ndarray:
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def add_arr_prop(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    for c in ["ADEP_mvt", "SCHED_TIME_UTC_mvt"]:
        if c not in dep.columns:
            raise KeyError(f"add_arr_prop dep missing {c}")
    for c in ["ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]:
        if c not in arr.columns:
            raise KeyError(f"add_arr_prop arr missing {c}")
    s = _epoch(pd.to_datetime(dep["SCHED_TIME_UTC_mvt"]))
    apt_d = dep["ADEP_mvt"].astype("string").fillna("MISS").to_numpy()
    l = _epoch(pd.to_datetime(arr["MVT_TIME_UTC_mvt"]))
    a = _epoch(pd.to_datetime(arr["SCHED_TIME_UTC_mvt"]))
    apt_a = arr["ADES_mvt"].astype("string").fillna("MISS").to_numpy()
    dly = l - a
    n = len(dep)
    cols = {}
    for w in WINDOWS:
        cols[f"n_arr_{w}"] = np.zeros(n, dtype=np.int32)
        cols[f"mean_adly_{w}"] = np.zeros(n, dtype=np.float64)
        cols[f"max_adly_{w}"] = np.zeros(n, dtype=np.float64)
        cols[f"n_late15_{w}"] = np.zeros(n, dtype=np.int32)
    for x in np.unique(apt_d):
        md = np.where(apt_d == x)[0]
        ma = np.where(apt_a == x)[0]
        if len(ma) == 0:
            continue
        lt = np.sort(l[ma])
        # order arr by landing time; need delays aligned to sorted landings
        o = np.argsort(l[ma], kind="stable")
        lts = l[ma][o]
        dlys = dly[ma][o]
        sd = np.argsort(s[md], kind="stable")
        sts = s[md][sd]
        for w in WINDOWS:
            lo = np.searchsorted(lts, sts - w * 60, side="left")
            hi = np.searchsorted(lts, sts, side="left")  # strictly past
            cnt = hi - lo
            cols[f"n_arr_{w}"][md[sd]] = cnt.astype(np.int32)
            P = np.concatenate([[0.0], np.cumsum(dlys)])
            P2 = np.concatenate([[0], np.cumsum((dlys > 900).astype(np.int64))])
            tot = P[hi] - P[lo]
            cols[f"mean_adly_{w}"][md[sd]] = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
            cols[f"n_late15_{w}"][md[sd]] = (P2[hi] - P2[lo]).astype(np.int32)
            # range-max over sliding window via deque (queries sorted, O(n+m))
            mx = np.zeros(len(sts))
            dq: deque = deque()
            j = 0
            for i in range(len(sts)):
                while j < len(lts) and lts[j] < sts[i]:
                    while dq and dlys[dq[-1]] <= dlys[j]:
                        dq.pop()
                    dq.append(j)
                    j += 1
                bound = sts[i] - w * 60
                while dq and lts[dq[0]] < bound:
                    dq.popleft()
                if dq and hi[i] > lo[i]:
                    mx[i] = dlys[dq[0]]
            cols[f"max_adly_{w}"][md[sd]] = mx
    out = pd.DataFrame({k: v for k, v in cols.items()}, index=dep.index)
    return out
