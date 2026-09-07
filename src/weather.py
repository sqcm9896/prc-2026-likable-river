"""Open ERA5 weather (via Open-Meteo archive API, openly licensed, retrieved
2026-09-06; see reports/weather/SOURCE.md): hourly temp/dewpoint/precip/
snowfall/wind/gusts/cloud per airport, joined on (ADEP, scheduled hour) --
strictly pre-departure info. Blunt operational flags: freezing/de-icing risk,
fog risk (dewpoint depression), precip, high wind."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

WDIR = ROOT / "reports" / "weather"
CACHE = WDIR / "weather_era5.parquet"
VARS = ["temperature_2m", "dew_point_2m", "precipitation", "snowfall",
        "wind_speed_10m", "wind_gusts_10m", "cloud_cover"]


def build_cache():
    rows = []
    for f in sorted(WDIR.glob("*.json")):
        d = json.loads(f.read_text())
        h = d["hourly"]
        n = len(h["time"])
        df = pd.DataFrame({"apt": f.stem, "time": pd.to_datetime(h["time"])})
        for v in VARS:
            df[v] = pd.to_numeric(h[v], errors="coerce")
        assert len(df) == n
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(CACHE, index=False)
    print(f"cache {CACHE} rows={len(out)}")
    return out


def add_weather(df):
    """df: frame with ADEP_mvt + SCHED_TIME_UTC_mvt. Returns features aligned
    to df.index. Missing hours -> neutral defaults (no precip, mild)."""
    if not CACHE.exists():
        build_cache()
    w = pd.read_parquet(CACHE)
    sh = pd.to_datetime(df["SCHED_TIME_UTC_mvt"], utc=True).dt.tz_localize(None).dt.floor("h")
    wh = pd.to_datetime(w["time"]).dt.tz_localize(None).dt.floor("h")
    key = df["ADEP_mvt"].astype("string") + "|" + sh.dt.strftime("%Y-%m-%d %H:%M")
    wkey = w["apt"].astype("string") + "|" + wh.dt.strftime("%Y-%m-%d %H:%M")
    w = w.set_index(wkey)
    hit = key.isin(w.index).mean()
    print(f"weather join hit rate: {hit:.4f}")
    assert hit > 0.99, "weather join failed"
    t = w["temperature_2m"]
    feats = pd.DataFrame(index=df.index)
    feats["wx_t"] = key.map(t).astype("float64").fillna(10.0)
    feats["wx_dpd"] = (key.map(w["temperature_2m"]) - key.map(w["dew_point_2m"])).astype("float64").fillna(5.0)
    feats["wx_precip"] = key.map(w["precipitation"]).astype("float64").fillna(0.0)
    feats["wx_snow"] = key.map(w["snowfall"]).astype("float64").fillna(0.0)
    feats["wx_wind"] = key.map(w["wind_speed_10m"]).astype("float64").fillna(10.0)
    feats["wx_gust"] = key.map(w["wind_gusts_10m"]).astype("float64").fillna(20.0)
    feats["wx_cloud"] = key.map(w["cloud_cover"]).astype("float64").fillna(50.0)
    feats["wx_freezing"] = (feats["wx_t"] < 0).astype("float64")
    feats["wx_deice"] = ((feats["wx_t"] < 2) & ((feats["wx_precip"] > 0) | (feats["wx_snow"] > 0))).astype("float64")
    feats["wx_fog"] = (feats["wx_dpd"] < 2.5).astype("float64")
    feats["wx_windy"] = (feats["wx_gust"] > 50).astype("float64")
    return feats
