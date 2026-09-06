# 03 — Review (gate each phase; reviewer answers every line)

- Leakage re-audit: no `BLOCK`/`TAXI` of test DEP anywhere; new features pass
  availability-in-ranking + pre/post-offblock classification; permissive
  derivations listed explicitly with justification.
- Validation honesty: time-based splits only; encoders/scalers fit on train
  window; OOF means never see validation fold; embargo respected on lookbacks.
- Score integrity: Jan + Jul + combined reported; per-airport breakdown (EGLL,
  LIRF tails watched); no selection on one month; hyperparameters retuned per
  feature set (no default-comparison).
- Robustness: clip bounds justified; unseen-level buckets hit in ranking;
  prediction histogram vs train (median ~912, p99 ~2,339); no absurd mass.
- Repro: seed fixed; run row in experiments.csv; artifact + feature version +
  manifest + commit recorded; clean-checkout regen possible.
- Compliance: submission budget respected; no probing; external data licensed;
  credentials absent from repo; GPLv3/original-work path intact.
