# Challenge rounds — adversarial review of EDA results (2026-09-06, build mode)

Scripts (independent code paths, kept in `/var/.../T/opencode/`):
`roundA_verify.py` (pyarrow.dataset + pyarrow.compute, no pandas concat),
`roundB_sensitivity.py` (Jan+Jul 2025 holdout: pool 1,740,630 / val 344,417).

## Round A — independent recomputation: ALL COUNTS PASS, medians confirmed
30/30 column check, row counts, identity (0 mismatches), all outlier thresholds
(369 / 19 / 1,105 / 74,167 / 4,126 / 584), min −12, max 131,167, LIRF-480,
LSZH-298, rank months, blanking, AOBT :00 = 0.977 — all PASS.
One FAIL class, investigated and dismissed: `pc.approximate_median` returned
918.9 overall vs 912.0. Third path (numpy exact median) gives **912.0** —
the FINDINGS number stands; Arrow's t-digest approx is just coarse on skewed
ints. Lesson: never report approximate quantiles; fine for features.

## Round B — holdout baselines (honest: SCHED-hour only, no BLOCK features)
| model | Jan | Jul | pooled | 50/50 |
|---|---|---|---|---|
| global median | 610.7 | 754.0 | 693.7 | 682.3 |
| airport median | 586.7 | 723.8 | 666.1 | 655.2 |
| airport×SCHED-hour | 584.0 | 718.6 | 662.0 | 651.3 |
Permissive MVT−AOBT proxy on same holdout: 428.2 (vs strict 441.1*) —
permissive buys little once airport medians exist; its value is per-row,
not aggregate. (*strict-on-subset number is confounded — see below.)

## Round B — three big findings (all change the plan)
1. **Join-fail rows are the tail**: val rows without `_flt` (1.5%, n=5,372)
   RMSE 3,979 vs 447 joined; 112/150 val >2h extremes lack joins; their mean
   taxi is 1,458 vs 1,005. The earlier "strict 441" was this confound.
   Consequence: `is_join_fail` is a first-class feature (it flags extremes),
   and join-fail rows need their own treatment, not just a flag.
2. **LIRF is the competition**: 7.7% of val rows, **59.6% of SSE** under
   airport-median. Per-airport RMSE: LIRF 1,853 / LFPG 701 / EGLL 503 /
   rest 313–416. EGLL global→airport head: 686 → 503 (confirmed necessary,
   but LIRF matters more). Consequence: per-airport (at least LIRF/EGLL/LFPG)
   modeling is mandatory; LIRF tail strategy decides the leaderboard.
3. **Clip reframed**: clipping TRAINING targets (1,800–7,200) moves neither
   median (666.1) nor mean (686.7–687.7) holdout RMSE — the val tail dominates
   regardless. Clipping predictions is equally vacuous (medians never exceed
   bounds). Consequence: drop "clip training" idea; instead (a) robust
   objective (Huber) for GBM, (b) cap only impossible predictions (negatives),
   (c) attack the LIRF tail directly with dedicated features/model.

## Minor
- Jul holdout much harder than Jan (718 vs 584); test is Jul-heavy (192k vs
  153k) — pooled RMSE is the honest target, 50/50 flatters by ~11 pts.
- Val-Jul MVT−SCHED drop (−181s) vs Jan rise (+184s) confirmed on holdout too.
