# EDA findings — full-data scan (`eda/01_full_eda.ipynb`, 5 execute-review loops, 2026-09-06)

Notebook asserts every number below (last cell `ALL EDA ASSERTIONS PASS`).
Figures in `eda/figures/`. Loop history: L1 path fix → L2 ID-overlap catch →
L3 full pass → L4 challenged AOBT precision + month-matched shift → L5 green.

## Scope (verified, not assumed)
- Train 4,167,797 rows (DEP 2,085,047 / ARR 2,082,750); ranking 689,534
  (Jan-2026 305,590 + Jul-2026 383,944; DEP 344,841 / ARR 344,693).
- 30 identical cols in 13 files; template 2 cols, IDs 1:1 + order-identical.
- **10 airports, all files**: EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM.
- IDs unique within + across all files (4,857,331/4,857,331) but ranges
  interleave (Oct/Nov) — never assume disjoint or sortable-by-ID time order.

## Target
- `TAXI=MVT−BLOCK` exact on 2,085,047/2,085,047 DEP; ARR `TAXI=BLOCK−MVT`
  exact (1 nominal mismatch = the single null row).
- DEP: mean 991 / median 912 / std 546 / p90 1,471 / p99 2,339 / max 131,167.
  Medians: LSZH 711 < EHAM 742 < EDDM 774 < EDDF 837 < LEBL 906 < LFPG 954 <
  LTFM 963 < LEMD 985 < LIRF 1,025 < EGLL 1,319.
- Outliers: 369 negatives (all −12…−1; LSZH 298, LFPG 61); 19 zeros;
  1,105 <60s; 74,167 >30min; 4,126 >60min; **584 >2h, of which LIRF 480 (82%)**.
  ~24h clusters + 131,167s max are data errors → clip policy mandatory.
- Seasonality weak (median 902–948); hour effect ±182s, dow ±28s on medians.
- Month-matched shift (NEW): MVT−SCHED median Jan 1,267→1,451 (+184s more
  delay in Jan-26) but Jul 1,684→1,503 (−181s); MVT−AOBT stable (+37/+8s).
  Airport shares train-vs-rank stable (≤0.8pp).

## Leakage / permissive-policy evidence
- Ranking DEP: BLOCK+TAXI 100% blank; MVT/SCHED 0% null; `_flt` block 1.53% null
  (train DEP 1.08%, ARR 0.74%).
- AOBT: **97.7% :00s, 2.3% nonzero seconds** (not strictly minute-precision).
  BLOCK−AOBT median +51s / medAbs 175s; ≤60s only 21%.
- Naive MVT−AOBT proxy: RMSE 384.9s, corr 0.536 — top feature, not a leak.
- Join-fail ~1% (DEP 1.1–1.5%): missingness flags required, never row drops.
- Agreement: ADEP 99.4%, ADES 99.3%, TYPE 99.7%. Novel in ranking DEP:
  STAND 30, ADES 62, TYPE 8, OPERATOR 30, RUNWAY 0. Callsign reused
  (~56 rows/cs full-year) → hash-or-drop.

## Modeling consequences (wired into plan/00, 04, 06)
1. Per-airport heads at least for EGLL (median +45%) and LIRF (tail).
2. Clip bounds validated offline; consider robust aux loss (tail dominates RMSE²).
3. Weight pooled RMSE by test mass (Jul 384k > Jan 306k), not 50/50.
4. Strict F1–F3 fallback scored alongside permissive F4 (Jan/Jul delay drift).
5. `MVT_ID` float64 exact-match only; never re-sort template order.

## Statistical EDA (`eda/02_statistical_eda.ipynb`, assertions pass)
- Shape: skew 48.4, kurtosis 7,016 — extreme right tail. log1p overcorrects
  (skew −0.61); NOT lognormal (normaltest rejects, p≈0). Model raw target for
  RMSE; consider sqrt/log1p only as aux heads judged on raw RMSE.
- Factor effect sizes (eta², all KW p≈0 — meaningless at n=2M, eta² is the
  story): RUNWAY 0.133 > OPERATOR 0.115 ≈ ADEP 0.112 > TYPE 0.068 >
  WAKE 0.036 > MARKET 0.024 > hour 0.006 > month/dow/flight-type ≈ 0.
  FE priority follows this order (factors overlap — runway nests in airport).
- Spearman vs target: MVT−AOBT 0.615 (pearson 0.536 — monotonic > linear),
  MVT−SCHED 0.388, AOBT−SCHED 0.153, LOBT/EOBT−SCHED ≈ 0, BLOCK−SCHED −0.017:
  delay LEVEL does not drive taxi — queue/clock context does.
- Shift (KS, n=100k samples): MVT−AOBT D=0.029, MVT−SCHED D=0.041, p≈0 —
  significant only by sample size; shapes match, median drift only.
- Tail: top 1% rows own 58% of SSE; top 0.1% own 46%. LIRF SSE share 51.5%
  full-train (59.6% Jan+Jul holdout — tail concentration varies by month).

## Challenge addendum (Round A+B, `eda/CHALLENGE.md` — supersedes 2–3 above)
- Round A re-verified every number above via an independent path; one artifact
  dismissed (Arrow approximate-median 918.9 vs exact 912.0 — FINDINGS stands).
- `is_join_fail` is a first-class extreme-flag feature (join-fail RMSE 3,979
  vs 447; 112/150 val extremes lack joins). Join-fail rows need own treatment.
- LIRF = 7.7% rows but 59.6% of SSE: per-airport modeling mandatory
  (LIRF/EGLL/LFPG first), LIRF tail strategy decides the leaderboard.
- "Clip training" idea DROPPED (moves neither median 666.1 nor mean ~687
  holdout RMSE). Instead: robust GBM objective (Huber), cap only negatives,
  attack LIRF tail directly.
- Holdout baselines for S2 (SCHED-hour honest): global 693.7 / airport 666.1 /
  airport×hour 662.0 pooled; MVT−AOBT proxy 428.2. Jul (718) >> Jan (584).

## Tail-fix addendum (2026-09-06, ship 422.9 → 359.2 pooled)
- Diagnosis (holdout, ship model): 54 rows with actual >3h own 68.6% of LIRF SSE;
  joined LIRF rows already fine (RMSE 613 ≈ LFPG). 98/114 LIRF >2h rows (86%) are
  join-fail: no `_flt` features, median-filled `m_*` arithmetic — the global model
  is blind there. Signal: `m_sched` is always observed and reads 10,769s median on
  extremes vs 1,615s otherwise. Model underpredicts extremes (top-actual-decile
  bias −798) while calibrated by predicted decile — it needs a dedicated head.
- Fix (`src/joinfail.py`): global join-fail specialist (jf rows 2,592→2,142) +
  LIRF-only specialist (LIRF-jf 7,738→4,205). Uniform patch rule (all jf → spec,
  LIRF-jf → LIRF-spec); hard patch beats blends. LIRF slice 1,128→800.
- Caveat / future work: on EDDF/EGLL/LEMD join-fail slices the base model still
  beats the specialist (e.g. EDDF 380 vs 526, n=599) — kept the uniform rule to
  avoid small-slice overfit; revisit with more data or per-airport specialists.
