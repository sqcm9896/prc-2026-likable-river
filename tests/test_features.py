"""Feature unit tests: stand-zone extractor must recover terminal info at numeric-stand airports."""
import pandas as pd

from src.features import stand_zone


def test_stand_zone_numeric_airports():
    s = pd.Series(["537", "320", "218R", "409", "605", "583"], dtype="string")
    a = pd.Series(["EGLL", "EGLL", "EGLL", "LIRF", "LIRF", "EDDM"], dtype="string")
    z = stand_zone(s, a)
    assert (z != "UNK").all()
    assert z.tolist() == ["EGLL_5", "EGLL_3", "EGLL_2", "LIRF_4", "LIRF_6", "EDDM_5"]


def test_stand_zone_alpha_and_missing():
    s = pd.Series(["B41", "V119", None], dtype="string")
    a = pd.Series(["EDDF", "EDDF", "EDDF"], dtype="string")
    assert stand_zone(s, a).tolist() == ["EDDF_B", "EDDF_V", "EDDF_UNK"]
