"""S1: calendar splits (Jan+Jul 2025 holdout) + RMSE. Usage:
python -m src.validation make-splits"""
import glob
import sys

import numpy as np
import pyarrow.dataset as ds

from .config import DATA, HOLDOUT_MONTHS


def load_dep_dep():
    d = ds.dataset(sorted(glob.glob(str(DATA / "training_*.parquet"))), format="parquet")
    df = d.to_table(columns=["TAXITIME_SEC_mvt", "ADEP_mvt", "BLOCK_TIME_UTC_mvt",
                             "SCHED_TIME_UTC_mvt"],
                    filter=ds.field("PHASE_mvt") == "DEP").to_pandas()
    df["month"] = df["BLOCK_TIME_UTC_mvt"].dt.month
    df["sched_h"] = df["SCHED_TIME_UTC_mvt"].dt.hour
    return df


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def report(df, pred, target="TAXITIME_SEC_mvt"):
    out = {}
    for name, s in [("jan", df[df.month == 1]), ("jul", df[df.month == 7]),
                    ("pooled", df)]:
        out[name] = rmse(s[pred], s[target])
    out["avg5050"] = (out["jan"] + out["jul"]) / 2
    return out


if __name__ == "__main__":
    assert sys.argv[1] == "make-splits"
    df = load_dep_dep()
    pool, val = df[~df.month.isin(HOLDOUT_MONTHS)], df[df.month.isin(HOLDOUT_MONTHS)]
    print(f"pool={len(pool)} val={len(val)} jan={(val.month == 1).sum()} jul={(val.month == 7).sum()}")
    val.to_parquet("reports/holdout_val.parquet", index=False)
    print("wrote reports/holdout_val.parquet")
