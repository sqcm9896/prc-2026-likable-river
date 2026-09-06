# 06 — Feature engineering plan (the full spec; implements brain.md §§2–5 + data audits)

Target: `TAXITIME_SEC_mvt` (DEP, raw seconds, RMSE). Policy: PERMISSIVE —
every provided column usable except test-blanked `BLOCK_TIME_UTC_mvt` /
`TAXITIME_SEC_mvt` themselves. Features build in versions; a version ships
only if Jan AND Jul 2025 holdout RMSE improve (or documented trade-off).

## F1 — Clocks (from `SCHED_TIME_UTC_mvt`, `MVT_TIME_UTC_mvt` as clock only)
`month, dow, hour, minute_bin(15m), sin/cos(hour·2π/24, dow·2π/7)`,
`is_weekend, is_night/curfew, holiday (per country of ADEP), season`.
`MVT_TIME` never enters arithmetic in v1 (takeoff = post-offblock); its
hour/dow feed congestion joins. `SCHED` is the anchor clock (0% null).

## F2 — Static categoricals (core of taxi-out, brain Ch5)
`ADEP, RUNWAY, STAND→stand-zone (terminal/apron prefix) + raw STAND,
AIRCRAFT_TYPE_mvt + _flt, WK_TBL_CAT, OPERATOR (anon hash), MARKET_SEGMENT,
FLIGHT_RULE_mvt/_flt, FLIGHT_TYPE, ADES route, diversion flag
(ADES_flt≠ADES_FILED_flt, 0.11%), origin ADEP_flt agreement flag.`
Encoding: GBM-native cats where possible; else smoothed OOF
`mean(taxi|level)`, m≈25–100, + `log(count)` + `is_rare`; RARE(<20 rows) →
parent back-off (stand→terminal→airport). PRIORITY by proven eta² (02 EDA):
RUNWAY (0.133) > OPERATOR (0.115) ≈ ADEP (0.112) > TYPE (0.068) > WAKE (0.036)
> MARKET (0.024) > hour (0.006). CALLSIGN/FLIGHT_mvt (~60% novel in
ranking): hash-or-drop, never plain categorical. Unseen-level garbage buckets
mandatory (ranking brings 29 novel STAND, 62 ADES, 8 types, 29 operators).
Interactions airport×hour, runway×hour handcrafted; no blind crossing (O(n²)).

## F3 — Schedule / plan deltas (pre-departure, SAFE)
`SCHED−LOBT, SCHED−IOBT, SCHED−EOBT_1, EOBT_1−IOBT, LOBT−IOBT`
(schedule adherence; LOBT/IOBT equal SCHED ~70% — the residuals carry signal),
plus missingness flags for the 1.5% joint `_flt`-missing block (indicator, not
row drop — drops would break template 1:1). `is_join_fail` is a FIRST-CLASS
extreme-flag feature (challenge-proven: join-fail RMSE 3,979 vs 447 joined;
112/150 val >2h extremes lack joins) — interact it with airport (esp. LIRF).

## F4 — Permissive time arithmetic (the AOBT decision)
`MVT−AOBT_3 (naive proxy, RMSE≈385s alone), MVT−SCHED, AOBT−SCHED,
MVT−LOBT/IOBT/EOBT_1, ARVT_1−MVT (DEP: +5,800s median, plan-level block time).
` Document as post-event-derived. Guard: distribution check train-vs-ranking
(stable: 960→978s median); keep strict-F1–F3 fallback model scored alongside.
`ARVT_3` excluded from DEP features (pure future, +6,218s after MVT).

## F5 — Congestion / flow (brain Ch2 counts), version v2
Trailing 15/30/60/120-min DEP+ARR counts per airport and per runway, anchored
on SCHED/MVT clocks, built chronologically with embargo (validation rows never
see future rows or validation targets). `log1p` + `qcut(5)` + `is_peak` flags
alongside raw counts (binarize when gamed/noisy). ARR ranking rows are context.
Queue proxies: scheduled demand in window, delay tail share.

## F6 — Historical priors (OOF aggregates), version v3
Smoothed out-of-fold `mean(taxi|airport×runway×hour-bin, stand-zone, route,
operator, aircraft family)` + pointwise variances; fit on train window only,
K-fold OOF + Laplace noise + leave-one-out. Recompute monthly (concept drift).

## F7 — Optional / usually skipped for GBM
PCA only on correlated flow numerics; k-means "regime" one-hots
(k≈20–100 on scaled hour+congestion+delay) under a linear head; log-target aux
head judged on raw RMSE only. Weather (METAR: wind/vis/precip/temp/pressure,
station map + license + as-of policy logged) — time-boxed, killed fast if flat.

## Version map
- v1 = F1+F2+F3 (+F4 permissive arithmetic; strict fallback without F4).
- v2 = v1 + F5. v3 = v2 + F6. v4 = weather/ensemble candidates.
- Per run: retune hyperparams (brain Ch4 — never compare on defaults), log
  RMSE_jan/RMSE_jul/combined + per-airport slices to `reports/experiments.csv`.
