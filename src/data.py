"""S0: manifest + schema audit + data-contract asserts. Usage:
python -m src.data manifest | audit | contract  (cwd = likable-river/)"""
import glob
import hashlib
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

from .config import AIRPORTS, DATA

TRAIN = sorted(glob.glob(str(DATA / "training_*.parquet")))
RANK = str(DATA / "ranking.parquet")
SUB = str(DATA / "submitting.parquet")


def cmd_manifest():
    rows = []
    for f in TRAIN + [RANK, SUB]:
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 20), b""):
                h.update(ch)
        pf = pq.ParquetFile(f)
        rows.append({"file": os.path.basename(f), "rows": pf.metadata.num_rows,
                     "cols": len(pf.schema_arrow.names),
                     "bytes": os.path.getsize(f), "md5": h.hexdigest()})
    man = pd.DataFrame(rows)
    man.to_csv("reports/manifest.csv", index=False)
    print(man.to_string())
    print("wrote reports/manifest.csv")


def cmd_audit():
    lines = []
    tot = 0
    for f in TRAIN:
        pf = pq.ParquetFile(f)
        n = pf.metadata.num_rows
        tot += n
        t = pq.read_table(f, columns=["ADEP_mvt", "ADES_mvt", "PHASE_mvt"]).to_pandas()
        d = sorted(t.loc[t.PHASE_mvt == "DEP", "ADEP_mvt"].unique())
        a = sorted(t.loc[t.PHASE_mvt == "ARR", "ADES_mvt"].unique())
        lines.append(f"{os.path.basename(f)}: {n} rows, DEP airports={len(d)}, ARR airports={len(a)}")
        assert d == a == AIRPORTS, f
    r = pq.read_table(RANK, columns=["MVT_TIME_UTC_mvt", "PHASE_mvt"]).to_pandas()
    r["ym"] = r["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    lines.append(f"train total: {tot}")
    lines.append("ranking months: " + str(r.groupby(["ym", "PHASE_mvt"]).size().to_dict()))
    with open("reports/data_audit.md", "w") as fh:
        fh.write("# data_audit (generated)\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
    print("\n".join(lines))
    print("wrote reports/data_audit.md")


def cmd_contract():
    seen = []
    for f in TRAIN + [RANK]:
        v = pq.read_table(f, columns=["MVT_ID_mvt"]).to_pandas()["MVT_ID_mvt"]
        assert v.isna().sum() == 0 and v.nunique() == len(v), f
        seen.append(v)
    all_ids = pd.concat(seen, ignore_index=True)
    assert all_ids.nunique() == len(all_ids), "cross-file dupes"
    rk = pq.read_table(RANK, columns=["MVT_ID_mvt", "PHASE_mvt"]).to_pandas()
    sb = pq.read_table(SUB).to_pandas()
    dep_ids = rk.loc[rk.PHASE_mvt == "DEP", "MVT_ID_mvt"]
    assert len(sb) == len(dep_ids) == 344841
    assert set(sb["MVT_ID_mvt"]) == set(dep_ids)
    assert (sb["MVT_ID_mvt"].values == dep_ids.values).all(), "template order"
    assert sb["TAXITIME_SEC_mvt"].isna().all()
    tr = pq.read_table(TRAIN[0], columns=["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                                           "TAXITIME_SEC_mvt", "PHASE_mvt"]).to_pandas()
    d = tr[tr.PHASE_mvt == "DEP"]
    assert (((d["MVT_TIME_UTC_mvt"] - d["BLOCK_TIME_UTC_mvt"]).dt.total_seconds()
             == d["TAXITIME_SEC_mvt"]).all())
    rd = pq.read_table(RANK, columns=["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
                                       "MVT_TIME_UTC_mvt", "PHASE_mvt"]).to_pandas()
    rd = rd[rd.PHASE_mvt == "DEP"]
    assert rd["BLOCK_TIME_UTC_mvt"].isna().all() and rd["TAXITIME_SEC_mvt"].isna().all()
    assert rd["MVT_TIME_UTC_mvt"].notna().all()
    print("CONTRACT GREEN: ids unique, template 1:1+ordered, TAXI identity, blanking")


if __name__ == "__main__":
    {"manifest": cmd_manifest, "audit": cmd_audit, "contract": cmd_contract}[sys.argv[1]]()
