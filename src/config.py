"""Shared config: paths, seeds, constants. No secrets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data"  # shared read-only competition parquets
REPORTS = ROOT / "reports"
SUBMISSIONS = ROOT / "submissions"
MODELS = ROOT / "models"

SEED = 42
HOLDOUT_MONTHS = (1, 7)  # Jan+Jul 2025 validate; mirrors Jan+Jul 2026 test
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
TARGET = "TAXITIME_SEC_mvt"
ID_COL = "MVT_ID_mvt"
