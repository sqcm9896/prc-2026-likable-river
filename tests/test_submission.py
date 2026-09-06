"""Zero-tolerance submission tests (run: pytest tests/ from likable-river/)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.submit import N_EXPECTED, validate


def test_template_contract():
    t = pq.read_table("../data/submitting.parquet").to_pandas()
    r = pq.read_table("../data/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt"]).to_pandas()
    dep = r.loc[r.PHASE_mvt == "DEP", "MVT_ID_mvt"]
    assert len(t) == N_EXPECTED and list(t.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    assert (t["MVT_ID_mvt"].values == dep.values).all()


def test_validator_accepts_good_rejects_bad(tmp_path):
    t = pq.read_table("../data/submitting.parquet").to_pandas()
    good = t.copy()
    good["TAXITIME_SEC_mvt"] = 900.0
    gp = tmp_path / "good.parquet"
    good.to_parquet(gp)
    validate(str(gp))  # must not raise
    bad = good.copy()
    bad.loc[0, "TAXITIME_SEC_mvt"] = -5.0
    bp = tmp_path / "bad.parquet"
    bad.to_parquet(bp)
    try:
        validate(str(bp))
    except AssertionError:
        return
    raise SystemExit("validator failed to reject negatives")
