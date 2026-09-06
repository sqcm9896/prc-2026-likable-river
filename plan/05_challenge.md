# 05 — Red-team challenge (argue against this plan; resolve before S9)

1. **Permissive risk**: AOBT/MVT arithmetic may encode 2025-specific reporting
   lag that shifts in Jan/Jul 2026 (median AOBT−BLOCK +51s could drift).
   Mitigation: monitor `MVT−AOBT` distribution train-vs-ranking (done: 960→978s
   median, stable); keep a strict-feature fallback model scored on holdouts.
2. **EGLL dominance**: EGLL median 1,319s (+45% over mean); a global model may
   smear it. Forced comparison global vs per-airport (S6) — do not skip.
3. **Tail wagging RMSE**: 0.2% rows >60min + LIRF extremes dominate RMSE².
   Clip too tight → bias on real long taxis; too loose → one 131,167s-style
   error costs ~thousands of RMSE points. Validate bounds, consider
   Huber/quantile aux heads.
4. **Seasonality trap**: train medians span only 902–948, but Jul-2026 ranking
   is the heavier month (383,944 vs 305,590 rows); weight pooled RMSE by actual
   test mass, not 50/50.
5. **plan-1 "submit early" vs fair-play**: cap submissions; v1 = genuine
   baseline, not a probe. Best-RMSE counting does not excuse pattern-probing.
6. **Brain skips**: hashing/PCA/k-means add nothing for GBM — do not revive
   without holdout proof. CALLSIGN/FLIGHT (~60% novel) stay hashed-or-dropped.
7. **Join-fail 1.5%**: DEP rows without `_flt` need missingness-indicator path,
   not row drops (drops would break template 1:1).
8. **Float64 IDs**: exact float equality on MVT_ID is safe < 2^53, but never
   cast to int32; never re-sort template order.
9. **Weather ROI**: likely marginal vs congestion; time-box it, kill fast.
10. **Rule-change clause**: re-check site/Discord before S9; keep strict-model
    fallback publishable even if organizers narrow the information policy.
