"""Submission gate: strict validator. Raises on any violation."""
import pyarrow.parquet as pq

from .config import DATA

N_EXPECTED = 344841


def validate(path, template=None):
    template = template or str(DATA / "submitting.parquet")
    t = pq.read_table(template).to_pandas()
    p = pq.read_table(path).to_pandas()
    assert list(p.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"], p.columns.tolist()
    assert len(p) == len(t) == N_EXPECTED, (len(p), len(t))
    assert (p["MVT_ID_mvt"].values == t["MVT_ID_mvt"].values).all(), "ID set/order"
    v = p["TAXITIME_SEC_mvt"]
    assert v.notna().all() and __import__("numpy").isinf(v.to_numpy()).sum() == 0
    assert (v >= 0).all(), "negatives present"
    assert str(v.dtype) in ("float64", "int64", "int32")
    print(f"SUBMISSION VALID: {path} ({len(p)} rows, range {v.min():.0f}-{v.max():.0f}s)")
