# 00 — Final plan (merged: plan.md + plan 1.md + brain.md + data audits, 2026-09-06)

## Decisions (user-confirmed)
1. **Permissive leakage**: all provided columns usable incl. AOBT_3 and
   MVT-derived arithmetic. 2. **Validation**: Jan+Jul 2025 holdout, report
   Jan/Jul/combined. 3. **Outliers**: explicit clip, validated offline.
4. **Airports: 10, resolved** (site "11" = typo). 5. No implementation yet —
   structure + plans only.

## Pipeline (in order, keep a feature family only if Jan AND Jul improve)
0. **Contract**: manifest + schema audit of `../data/`; assert 30 cols, ID
   uniqueness/disjointness, template⊆ranking-DEP 1:1, TAXI=MVT−BLOCK exact.
1. **Baselines**: global median → airport×month×hour median (+shrinkage) →
   +permissive `MVT−AOBT` (expect ~385s RMSE region).
2. **Core model**: CatBoost (native cats) vs LightGBM on clock (hour/dow/month,
   sin/cos, holiday) + static (airport/runway/stand-zone/type/wake/operator/
   market/rule/flight-type/route/diversion) + schedule deltas
   (SCHED/LOBT/IOBT/EOBT_1 vs each other) + permissive arithmetic
   (MVT−AOBT/SCHED/…) + missingness flags. OOF smoothed target means
   (m≈25–100) + log(count) + RARE back-off; garbage-bucket unseen.
3. **Congestion** (chronological + embargo): trailing 15/30/60/120-min
   DEP/ARR counts per airport/runway from SCHED/MVT clocks; queue proxies;
   `log1p` + `qcut(5)` + `is_peak`. ARR rows as context.
4. **Interactions for linear head only** (airport×hour, runway×ops,
   airline×airport); GBM finds the rest — skip blind crossing.
5. **Tuning**: Optuna on Jan+Jul RMSE; retune per feature set; global vs
   per-airport comparison on same split (EGLL likely needs own head).
6. **Optional**: METAR weather (licensed+logged) → ensemble blend (weights on
   holdouts, confirmed on rolling split) → refit full-2025 → clip →
   template-ordered submit → manifest.
7. **Publication**: GPLv3, model card, repro docs, JOAS seed.

## Why this order
Congestion + permissive time-arithmetic attack the two biggest known signals
(AOBT proxy R² 0.29; airport medians span 711→1,319s); everything else is
incremental. Clip policy exists because 584 rows >2h dominate RMSE².
Evidence: `eda/FINDINGS.md` (executed notebook, assertions pass).
