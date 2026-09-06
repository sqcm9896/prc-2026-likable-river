# 02 — Actions (concrete commands; run in order; log every run)

1. `python -m src.data manifest --data ../data` → manifest (sizes, checksums).
2. `python -m src.data audit` → `reports/data_audit.md` (schema, nulls by
   PHASE, airports=10, timestamps, ID checks, TAXI identity, blanking).
3. `python -m src.validation make-splits` → Jan+Jul 2025 holdout (+embargo).
4. `python -m src.baselines` → median tables + RMSE table.
5. `python -m src.features --version v1` (deterministic builder, clock+static+
   schedule+permissive arithmetic+missingness).
6. `python -m src.train --model catboost --features v1` (+ lightgbm compare);
   log to `reports/experiments.csv`
   (run_id, manifest, features, split, params, RMSE_jan, RMSE_jul, combined,
   runtime, decision, notes).
7. Residual slices by airport/month/hour/runway/type/quantile → `reports/`.
8. `python -m src.features --version v2` (+congestion) → retrain → compare.
9. `python -m src.features --version v3` (+priors) → retrain → compare.
10. Optuna tuning on frozen features → global-vs-per-airport test.
11. Optional weather join (license logged) / ensemble blend.
12. Clip bounds test → freeze → `python -m src.train --full` →
    `python -m src.submit` → `pytest tests/test_submission.py` →
    upload `likable-river_v<N>.parquet` → manifest row → leaderboard check.
