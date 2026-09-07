# JOAS paper seed — Predicting Departure Taxi-Out Times with Gradient Boosting
and Tail Specialists (PRC Data Challenge 2026, team likable-river)

DRAFT seed, 2026-09-07. Target venue: Journal of Open Aviation Science (JOAS),
PRC Data Challenge 2026 special track. To be finalized after leaderboard
confirmation. Numbers below are Jan+Jul 2025 holdout RMSE (seconds) unless noted.

## Proposed title
Specialist-Head Gradient Boosting for Airport Departure Taxi-Out Prediction:
Handling the Join-Failure Tail

## Abstract (skeleton)
Taxi-out time is a key post-operations quantity for congestion, fuel and CO₂
analysis, yet hard to predict: at ten major European airports the distribution
is heavy-tailed (median 912 s, p99 2,339 s) and 1.5% of departures lack a
Network Manager flight-plan join. We show (i) naive median baselines score
662–694 s RMSE and an honest `MVT−AOBT` proxy 654 s; (ii) CatBoost/LightGBM on
clock, static, schedule-delta and OOF-encoded features reach 433.6 s;
(iii) 86% of >2 h extremes at Rome Fiumicino are join-fail rows where global
models are blind, and dedicated join-fail specialists cut pooled RMSE to
359.2 s (Jan 355.2 / Jul 362.4). Full-2025 refit submitted to the official
leaderboard. Code/data: GPLv3 repo (reproduce in README), open data only.

## 1. Introduction
- Operational motivation: post-ops congestion intervals, excess fuel/CO₂ vs
  unconstrained taxi (challenge rationale).
- Task: DEP taxi-out = takeoff − off-block; 10 airports; train 2025 (4.17M rows),
  test Jan+Jul 2026 (344,841 DEP); metric RMSE; best submission counts.

## 2. Data & EDA highlights (eda/FINDINGS.md)
- `TAXI = MVT−BLOCK` exact on all train DEP; 30 cols; IDs unique, template 1:1.
- Tail: top 1% rows = 58% of SSE; 584 rows >2 h, 82% at LIRF; negatives cluster LSZH.
- Factor effects (eta²): RUNWAY .133 > OPERATOR .115 ≈ ADEP .112 > TYPE .068.
- Drift 2026 vs 2025: MVT−SCHED medians Jan +184 s / Jul −181 s; MVT−AOBT stable.

## 3. Method
- Validation: Jan+Jul 2025 holdout (calendar-matched), Jan/Jul/pooled reporting,
  ship-only-if-both-improve rule.
- Features v1: clocks (cyclical), static cats (native GBM) + OOF smoothed means
  (m=50) + log-counts, schedule deltas, permissive MVT−AOBT/SCHED arithmetic,
  `is_join_fail` extreme flag. Airport-aware stand zones (EGLL_5-style).
- Models: CatBoost (d8, lr .05) + LightGBM-L2, per-airport heads (LIRF/EGLL/LFPG),
  join-fail specialists (global + LIRF-only), 0.8/0.2 blend, negative-cap only.
- Rejected with evidence (experiments.csv): congestion v2, depth-10, XGBoost,
  wake-ahead counts, ERA5 weather, train/upper-tail clipping, all bagging.

## 4. Results
- Table: baselines → 433.6 → 423.7 → 422.9 → **359.2** (per-airport slices,
  MAE 163, ±2 min 53%). Strict (operational, no MVT/AOBT) track: 508.6.
- Ablations: each rejected variant with Jan/Jul deltas (from experiments.csv).
- Leaderboard: [TO FILL after results].

## 5. Discussion
- Leakage position: permissive use of unblanked MVT/AOBT documented; operational
  fallback provided; call for organizer clarification (cite Discord status).
- Limitations: seed sensitivity (±6 base), 2026 drift, far-tail underprediction,
  ERA5 too coarse to help.

## References (to complete)
- PRC DC 2024 data paper, JoAS 3(2) 2025, doi:10.59490/joas.2025.8252.
- PRC DC 2025 fuel-burn outcome + Kačar & Ćulibrk (ERA5/METAR) Zenodo 2026.
- team_likable_jelly 2024 repo (seed-averaging precedent).
- OpenAP, METAR/ERA5 via Open-Meteo (reports/weather/SOURCE.md).
