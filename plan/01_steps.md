# 01 — Steps (exit criteria per step; stop if a criterion fails)

- [x] S0 manifest: byte sizes + checksums + row counts for all 14 parquets.
      Exit: manifest file + `reports/data_audit.md`.
- [x] S1 contract asserts (IDs unique/disjoint, template 1:1, TAXI identity,
      blanking 100%, dtype check float64 IDs). Exit: asserts green.
- [x] S2 baselines on Jan+Jul 2025 holdout (median tables). Exit: baseline RMSE
      table (Jan/Jul/combined, overall + per-airport).
- [x] S3 core GBM run (00 §2). Exit: beats baseline on combined holdout;
      artifact + params + feature version + residual slices saved.
- [x] S4 congestion family. Exit: improves Jan AND Jul (or documented trade-off).
      DONE 2026-09-06: v2 worse both months (cat 436.4 / lgbm 450.3) → DROPPED.
- [x] S5 priors/interactions (OOF means, runway×stand). OOF in v1 done; per-airport
      heads RETRAINED with fill fix: v1 patch 433.6→423.7 (both months improve);
      v2 heads REJECTED (430.5→435.8, overfit on congestion feats).
- [x] S6 tuning + global-vs-per-airport. FINAL: perapt-cat×0.8 + lgbm-L2×0.2 =
      422.9 pooled. lgbm grid (4 runs) keeps defaults; cat depth-10 rejected.
- [ ] S7 optional weather/ensemble. Exit: improves rolling-split RMSE or dropped.
      (untouched — GBM tuning exhausted first per stronger-lever rule).
- [ ] S7 optional weather/ensemble. Exit: improves rolling-split RMSE or dropped.
- [x] S8 clip policy validation. DONE 2026-09-06: negatives-only (no-op, safe);
      upper-tail cap REJECTED (+90.7 pooled).
- [~] S9 refit full-2025, predict, validate, submit `likable-river_v<N>.parquet`.
      DONE locally 2026-09-06: `likable-river_v3.parquet` (full refit global+perapt
      heads+jf specialists, PERAPT=1 JFPATCH=1 W_CAT=0.8, validator green, manifest
      row). v2 (no jfpatch) + v1 frozen kept as backups.
      UPLOAD explicitly held per user (no submissions yet).
- [ ] S10 publish: GPLv3 LICENSE, README repro, model card, JOAS seed.
      Exit: public repo complete.
