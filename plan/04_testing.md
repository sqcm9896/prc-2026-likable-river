# 04 — Testing (validation protocol + submission gates)

## Offline validation
- Primary: train Feb–Jun+Aug–Dec 2025 (+Jan/Jul excluded), validate Jan 2025
  and Jul 2025 separately; report RMSE_jan, RMSE_jul, combined (pooled), plus
  per-airport tables. Iteration: rolling-origin + embargo on lookbacks.
- Secondary: small random holdout as overfit sanity only (never for selection).
- Target: raw seconds RMSE. Aux `log1p`-target runs allowed but judged on raw
  back-transformed RMSE only.

## Submission gate (`pytest tests/test_submission.py`, zero tolerance)
- Exactly 2 cols in template order; row count == 344,841; `MVT_ID` set+order
  == template; no dupes/drops; all finite, non-negative, no nulls; plausible
  range vs train; file reads back and re-passes; name
  `likable-river_v<N>.parquet`; manifest row (checksum, run_id, timestamp).

## Experiment log
Append-only `reports/experiments.csv`: run_id, data_manifest, feature_version,
split, model, parameters, RMSE_jan, RMSE_jul, combined_RMSE, runtime,
decision, notes.
