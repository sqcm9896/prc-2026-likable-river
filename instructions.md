# instructions.md — PRC Data Challenge 2026, team likable-river (source of truth)

## Task
Predict departure **taxi-out** (seconds) = `MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`
for DEP rows in `ranking.parquet`. Verified: identity holds exactly on all
2,085,047 train DEP rows. Metric: **RMSE**, best submission wins.
Timeline: 1 Sep → 11 Oct 2026 23:59:59 CET. Prize €5,000 top-3.

## Data (shared, read-only: `../data/`)
- 12× `training_2025-MM.parquet` (2025 full year, 4,167,797 rows, 30 cols).
- `ranking.parquet` (689,534 rows: Jan 2026 305,590 + Jul 2026 383,944;
  DEP 344,841 to predict, ARR 344,693 context, fully observed).
- `submitting.parquet` (344,841×2 template, IDs 1:1 + order-identical to ranking DEP).
- Target distribution DEP: mean 991 / median 912 / std 546 / p90 1,471 /
  p99 2,339 / max 131,167. Per-airport medians: LSZH 711, EHAM 742,
  EDDM 774, EDDF 837, LEBL 906, LFPG 954, LTFM 963, LEMD 985, LIRF 1,025,
  EGLL 1,319.

## Rule points (status as of 2026-09-06)
1. **Airport count: RESOLVED = 10.** Site scope sentence says "11" but its own
   Table 1 lists 10 and every one of the 13 data files contains exactly
   EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM. Treat "11" as site typo;
   flag if organizers announce otherwise.
2. **RMSE weighting Jan vs Jul: OPEN.** Report pooled + per-month RMSE locally.
3. **Submission cap: OPEN.** No numeric limit published; "monitoring" only.
   Ask on Discord; enforce local budget (baseline + genuine iterations only).
4. **Leakage policy: PERMISSIVE (user decision 2026-09-06).** All provided
   columns usable, incl. `AOBT_3_flt`, `MVT_TIME_UTC_mvt` arithmetic
   (`MVT−AOBT/SCHED/LOBT/IOBT/EOBT_1`). Rationale: organizers left them
   unblanked. Document everything; never touch `BLOCK_TIME`/`TAXITIME` of test DEP.
5. **Naive `MVT−AOBT` proxy RMSE ≈ 385s** (R² 0.29, medianAbs AOBT-vs-BLOCK 175s)
   — strong feature, not exact leak. AOBT 97.7% :00s (2.3% nonzero seconds).
   Month-matched drift: MVT−SCHED Jan +184s / Jul −181s (2026 vs 2025);
   MVT−AOBT stable. >2h outliers are 82% LIRF; negatives 81% LSZH.
6. **Batch congestion from ranking rows: ALLOWED** (permissive). Chronological
   only; no validation-target peeking; embargo lookback features.
7. **External data (e.g. METAR): allowed only if** openly licensed +
   documented (source/version/retrieval date/as-of policy). Low priority.
8. **ARR rows: context only**, never targets.
9. **Rules may change; leaderboard widget broken** (Observable migration 1 Sep).
   Watch Discord `#prc-data-competition` + REST API
   (`competitionId bb3693e1-26bc-4a9e-8619-4fe78b4eab0c`).

## Hard policies
- Validation: train ≤ month T, validate Jan+Jul 2025 (same calendar months as
  test), report Jan / Jul / combined RMSE. Never select on one month alone.
- Outliers: NO train-clip (proven vacuous). Squared-error objective (Huber
  REJECTED: lgbm-huber 668.4 vs l2 448.8 — Huber optimizes the wrong target
  for RMSE). Cap only negatives, dedicated LIRF-tail strategy
  (LIRF = 7.7% rows, 59.6% of SSE).
- IDs: `MVT_ID_mvt` is float64 — join/order as given, output template order.
  Unique within + across files, but ranges interleave (Oct/Nov) — never infer
  time order from IDs.
- Submission validator must pass before every upload (exact ID set/order,
  2 cols in order, finite non-negative, no nulls, read-back check).
- Encoders/scalers fit on train window only; OOF target means (m≈25–100) +
  `log(count)` + RARE(<20)→parent back-off; garbage-bucket unseen levels
  (29 novel STAND, 62 ADES, 8 types, 29 operators seen in ranking).
- Retune hyperparameters per feature set; log every run in
  `reports/experiments.csv`.
- Prize path: public GitHub, GPLv3, reproducible docs, open data only, original work.
- No leaderboard probing; credentials never in repo (`data/*.parquet` ignored).

## Status (2026-09-06, tail fixed, NOTHING uploaded)
- Ship config FINAL: perapt-catboost ×0.8 + lgbm-L2 ×0.2 + join-fail patch
  (join-fail rows → jf specialist, LIRF join-fail → LIRF specialist).
  Holdout pooled 359.2 (jan 355.2 / jul 362.4). Was 422.9 before the tail fix.
- Why it works: 86% of >2h LIRF extremes are join-fail rows where the global
  model is blind (median-filled m_*); m_sched (present, 10,769s vs 1,615s median)
  carries the signal; dedicated heads learn it (jf 2592→2142, LIRF-jf 7738→4205).
- `submissions/likable-river_v3.parquet` = full-2025 refit (`SUB_VER=3 PERAPT=1
  JFPATCH=1 W_CAT=0.8`, 5,290 rows patched), validator green, manifest row written.
  v2 (no jfpatch) + v1 frozen kept as backups.
- Review response (2026-09-06, `recommendations.md`): stand_zone fixed airport-aware
  (was 100% UNK at EGLL/LIRF) BUT _sz lineage 361.2 loses to v1 359.2 both months →
  fix kept, ship stays v1 (Jan+Jul rule). Strict insurance track retuned 512.4→508.6
  (cat-alone; blends/perapt mixed → dropped). XGBoost 481.8 dropped (adds ~0 to ship).
  Refit iter caps aligned (LFPG 700, jf 360/890). pytest config added (4 passed).
- NOTE: a second worker is active in this repo (wrote `src/ensemble_eval.py`,
  retrained `cat_v2.cbm` → 430.5 verified, refactored `train.fill_nans` +
  `predict.py` retrain path). Coordinate before editing shared files.
