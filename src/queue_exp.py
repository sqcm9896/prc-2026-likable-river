"""Experienced-queue features (MIT Simaiakis/Balakrishnan NQ(l) idea): the queue
*you personally sat in*, measured over your own (pushback, takeoff) interval.
All inputs are observable at inference under the permissive policy
(MVT/AOBT/SCHED/RUNWAY present in ranking); no targets used. Past/future rows
are all observable, so no embargo needed — but counts never touch y.

- nq_takeoff: takeoffs with MVT in (AOBT_i, MVT_i), per airport and per runway
- nq_pushback: pushbacks with AOBT in (AOBT_i, MVT_i), per airport
- wip_at_pushback: aircraft with AOBT_j < AOBT_i < MVT_j (surface inventory
  when you push), per airport
AOBT missing (join-fail) -> SCHED fallback; join-fail rows keep is_join_fail=1
from the main pipeline to gate on.
"""
import numpy as np
import pandas as pd


def add_queue_exp(dep):
    """dep: DEP-only frame with ADEP_mvt, RUNWAY_mvt, MVT_TIME_UTC_mvt,
    AOBT_3_flt, SCHED_TIME_UTC_mvt. Returns DataFrame aligned to dep.index."""
    df = dep.copy()
    df["mvt_i"] = df["MVT_TIME_UTC_mvt"].astype("int64") // 10**9
    aobt = df["AOBT_3_flt"]
    fallback = aobt.isna()
    df["obt_i"] = np.where(fallback,
                           df["SCHED_TIME_UTC_mvt"].astype("int64") // 10**9,
                           aobt.astype("int64") // 10**9)
    df["rwy"] = df["RUNWAY_mvt"].astype("string").fillna("MISS")
    out = pd.DataFrame(0.0, index=df.index,
                       columns=["nq_takeoff", "nq_takeoff_rwy", "nq_pushback",
                                "wip_at_pushback"])
    for apt, g in df.groupby("ADEP_mvt", sort=False):
        t = np.sort(g["mvt_i"].to_numpy())
        # takeoffs in (obt_i, mvt_i): strict both ends excludes self
        o = g.sort_values("mvt_i")
        ot = o["mvt_i"].to_numpy()
        oob = o["obt_i"].to_numpy()
        lo2 = np.searchsorted(ot, oob, side="right")
        hi2 = np.arange(len(ot))
        out.loc[o.index, "nq_takeoff"] = (hi2 - lo2).clip(min=0)
        # pushbacks in interval: order by obt
        op = g.sort_values("obt_i")
        pt = op["obt_i"].to_numpy()
        pm = op["mvt_i"].to_numpy()
        # per-row: #{j: obt_i < obt_j < mvt_i}
        lo3 = np.searchsorted(pt, op["obt_i"].to_numpy(), side="right")
        hi3 = np.searchsorted(pt, pm, side="left")
        out.loc[op.index, "nq_pushback"] = (hi3 - lo3).clip(min=0)
        # WIP at pushback: #{j: obt_j < obt_i < mvt_j} = pushbacks before - takeoffs at/before
        mv_all = np.sort(t)
        n_pushed = np.searchsorted(pt, op["obt_i"].to_numpy(), side="left")
        n_taken = np.searchsorted(mv_all, op["obt_i"].to_numpy(), side="left")
        out.loc[op.index, "wip_at_pushback"] = (n_pushed - n_taken).clip(min=0)
        # per-runway takeoffs in interval
        for rw, g2 in o.groupby(o["rwy"].values, sort=False):
            rt = g2["mvt_i"].to_numpy()
            ro = g2["obt_i"].to_numpy()
            rlo = np.searchsorted(rt, ro, side="right")
            rhi = np.arange(len(rt))
            out.loc[g2.index, "nq_takeoff_rwy"] = (rhi - rlo).clip(min=0)
    for c in list(out.columns):
        out["log_" + c] = np.log1p(out[c].clip(lower=0))
    return out.astype("float64")
