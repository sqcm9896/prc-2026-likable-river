# PRC Data Challenge 2026 — team likable-river

Predict departure taxi-out (seconds) for 10 European airports. Metric: RMSE.
Test: Jan + Jul 2026 (`ranking.parquet`, 344,841 DEP rows to predict).

## Layout
- `instructions.md` — rules + policies (source of truth).
- `plan/` — final plan, steps, actions, review, testing, red-team challenge.
- `src/` — pipeline code (S0–S3 + S9 implemented: config/data/validation/
  baselines/features/train/predict/submit; per_airport fill-bug fixed, heads
  pending retrain).
- `tests/` — submission validator tests (implemented, pytest 2 passed).
- `reports/` — EDA + experiment log (`experiments.csv`).
- `submissions/` — local copies + manifest (ignored except manifest).
- `models/` — trained artifacts (ignored).
- `../data/` — shared read-only competition parquets (never copy, never commit).
- `eda/01_full_eda.ipynb` — executed full-data EDA (asserts pass) + `FINDINGS.md`.

## Data contract (verified 2026-09-06)
Train 4,167,797 rows (DEP 2,085,047 / ARR 2,082,750), 30 cols, `TAXI=MVT−BLOCK`
exact (DEP) / `TAXI=BLOCK−MVT` exact (ARR). Ranking DEP: BLOCK+TAXI 100% blank,
MVT/SCHED + all `_flt` times present (~98.5%). Template IDs 1:1 + order-identical.
Full numbers: `eda/FINDINGS.md` (notebook assertions green).

## Reproduce
Requirements: Python 3.12, `pip install -r requirements.txt`, ~2 GB RAM, ~1 h for
a full refit. Data: place the 14 competition parquets in `../data/` (team bucket
`prc-2026-datasets`; checksums in `reports/manifest.csv`, audit via
`python -m src.data`).

Full pipeline (ship config):
`python -m src.train` → `python -m src.per_airport` → `python -m src.joinfail` →
`SUB_VER=3 W_CAT=0.8 PERAPT=1 JFPATCH=1 python -m src.predict` → validator +
manifest run automatically. Upload: `mc cp submissions/likable-river_v3.parquet
<mc-alias>/prc-2026-likable-river/` and confirm the remote ETag equals the manifest
md5. Strict (operational, no MVT/AOBT) track: prefix training with `STRICT=1`
(best strict: CatBoost 508.6 pooled). Every run appends to `reports/experiments.csv`;
a feature ships only if Jan AND Jul holdout RMSE improve.
Current FINAL: perapt-catboost×0.8 + lgbm-L2×0.2 + join-fail patch (v1 feats),
holdout pooled 359.2 (jan 355.2 / jul 362.4). `submissions/likable-river_v3.parquet`
(full refit) validated 2026-09-06, UPLOADED 2026-09-07 (ETag verified). v2 + v1
frozen kept as backups. Model details: MODEL_CARD.md. Paper seed: JOAS_seed.md.
License: GNU GPLv3 (LICENSE).
