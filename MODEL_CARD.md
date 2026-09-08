# Model Card — likable-river taxi-out predictor (PRC Data Challenge 2026)

## Model details
- **Task:** predict departure taxi-out time (seconds) = `MVT_TIME_UTC − BLOCK_TIME_UTC`
  for 344,841 DEP rows (Jan+Jul 2026) at 10 European airports.
- **Ship config (v11, uploaded 2026-09-09, scoring after UTC reset):** 0.7 × v1-lineage
  ship + 0.3 × queue-lineage ship (each = CatBoost×0.8 + LightGBM-L2×0.2 with
  ARR-propagation + runway-flow junior features, per-airport heads, join-fail
  specialists) + physics-identity floor (joined m_aobt>1800 → max(pred,
  m_aobt−3600)) + negative-cap. Full-2025 refits.
  Scored ships: v10 **316.05**, v9 316.19, v5 336.46 (leaderboard RMSE).
  Previous ships v3 (345.57) and v4-strict (470.85) kept as fallback/insurance.
- **Hyperparameters:** CatBoost iters 1500 (LFPG head capped 700), lr 0.05, depth 8,
  RMSE loss; LightGBM-L2 leaves 127, min_data 500, lr 0.05. All decisions logged
  in `reports/experiments.csv`.

## Training / validation data
- Train: 2025 full year, 4,167,797 rows (DEP 2,085,047 / ARR 2,082,750), 30 cols.
- Validation: Jan+Jul 2025 holdout (pool 1,740,628 / val 344,419); report Jan / Jul /
  pooled RMSE; a feature ships only if Jan AND Jul improve.
- Target DEP stats: mean 991 / median 912 / std 546 / p99 2,339s.

## Results (holdout RMSE, seconds)
| model | Jan | Jul | pooled |
|---|---|---|---|
| global median | 610.7 | 754.0 | 693.7 |
| airport×hour median | 584.0 | 718.6 | 662.0 |
| honest MVT−AOBT proxy | 559.5 | 720.8 | 653.8 |
| CatBoost v1 | 397.1 | 460.9 | 433.6 |
| + per-airport heads | 390.2 | 448.9 | 423.7 |
| + lgbm blend (0.8/0.2) | 387.4 | 449.5 | 422.9 |
| **+ join-fail specialists (SHIP)** | **355.2** | **362.4** | **359.2** |
| strict (no MVT/AOBT) CatBoost | 469.5 | 538.0 | 508.6 |

Per-airport ship slices: LEMD 191 … EGLL 280, LFPG 573, LIRF 800.
MAE 163 / median-AE 111 / ±2min 53% / ±5min 88%.

## Key findings
- 86% of >2h LIRF extremes are join-fail rows (~1%: no Network Manager `_flt`
  join); `MVT−SCHED` delay (always observed) carries the signal; dedicated heads
  cut LIRF-join-fail RMSE 7,738→4,205.
- Rejected with evidence: v2 congestion, depth-10, XGBoost (481.8), wake-ahead
  counts, ERA5 weather, zone-feature lineage switch, all bagging variants.
- Upper-tail prediction clipping hurts (+90.7); train-target clipping is vacuous.

## Limitations & risks
- **Leakage policy PERMISSIVE** (team decision 2026-09-06): uses unblanked
  `MVT_TIME_UTC_mvt` arithmetic + `AOBT_3_flt`. If organizers blank these or
  require operational purity, fall back to the strict track (508.6).
- Seed sensitivity: pure-seed spread ≈ ±6 pooled on base models; frozen numbers
  carry seed luck. Month drift 2026 vs 2025 observed (±~180s schedule-delay medians).
- Tail rows (data errors up to ~24h) dominate RMSE; model is calibrated by
  predicted decile but systematically underpredicts the far tail.

## Reproduce
See README. `pip install -r requirements.txt` → `python -m src.train` →
`SUB_VER=3 W_CAT=0.8 PERAPT=1 JFPATCH=1 python -m src.predict` → validator +
manifest run automatically. Strict track: prefix commands with `STRICT=1`.

## License / prize path
GNU GPLv3 (LICENSE). Open data only (competition parquets + Open-Meteo ERA5,
documented in `reports/weather/SOURCE.md`). Original work; no leaderboard probing.
