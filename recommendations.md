# Project Review, Adversarial Challenge & Recommendations
**Target Project:** PRC Data Challenge 2026 — Team `likable-river`  
**Evaluation Date:** September 6, 2026  
**Auditor:** Antigravity (Google DeepMind)  
**Scope:** Architecture, Feature Engineering, Validation Honesty, Regulatory/Competition Risk, Software Quality, and Advanced Machine Learning Strategy.

---

## Executive Summary & Scorecard

The `likable-river` repository demonstrates structured thinking, systematic experimentation logging, and impressive local score progression (moving from baseline medians of ~662s RMSE down to a reported 359.2s RMSE). The team has correctly diagnosed that extreme outliers and join failures at Rome Fiumicino (LIRF) dominate the sum of squared errors ($SSE$), and has created dedicated specialists to address them.

However, beneath the headline RMSE of **359.2s**, the project harbors **critical vulnerabilities, subtle data leakage in validation, a silent bug blinding models to parking stands at 5 out of 10 airports, and severe competition disqualification risks**.

| Dimension | Rating | Verdict / Critical Finding |
| :--- | :---: | :--- |
| **Competition Compliance & Leakage** | ⚠️ High Risk | Permissive use of post-departure timestamps (`MVT_TIME` + `AOBT_3`) creates an existential disqualification risk if organizers enforce operational realism. |
| **Validation Integrity** | ❌ Compromised | The Jan+Jul 2025 "holdout" is contaminated by early-stopping target peeking and hardcoded tree-count harvesting. |
| **Refit & Generalization** | ❌ Flawed | `predict.py` switches from Out-Of-Fold target encoding to in-sample empirical encoding, and doubles tree counts on airports that overfit (e.g., LFPG). |
| **Feature Engineering Quality** | ⚠️ Severe Bug | `stand_zone` regex `^([A-Z]+)` produces **100% UNK** for Heathrow (EGLL) and Rome (LIRF), completely stripping parking terminal signals. |
| **Code Modularity & Cleanliness** | ⚠️ Fragile | Duplicated dataset prep across 5 scripts; global mutable state driven by shell environment variables; pytest fails without `python -m`. |
| **ML Architecture Maturity** | ⚠️ Unfinished | Relies on basic GBDT blend (CatBoost + LightGBM); lacks target decomposition, asymmetric tail loss, multi-seed bagging, and neural embeddings. |

---

## 1. Adversarial Challenge & "Grilling" the Core Assumptions

### 1.1 The Permissive Leakage Trap (`MVT_TIME_UTC_mvt` + `AOBT_3_flt`)
- **The Grilling Question:** *Are you predicting taxi-out time, or are you doing post-hoc telemetry reconciliation?*
- **The Reality:** 
  Taxi-out is defined as $\text{Takeoff} - \text{Off-Block} = \text{MVT\_TIME} - \text{BLOCK\_TIME}$. In `ranking.parquet`, the organizers left `MVT_TIME_UTC_mvt` unblanked (actual takeoff time) and left `AOBT_3_flt` unblanked (actual off-block time from Network Manager). The subtraction $\text{MVT} - \text{AOBT}$ achieves an RMSE of ~385s alone without machine learning.
- **The Risk:**
  1. **Disqualification:** The challenge rules explicitly state: *"Optimize the RMSE used by the official leaderboard, but do not chase leaderboard noise or use target leakage."* Furthermore, prize winners must submit a paper to the *Journal of Open Aviation Science* (JOAS). An operational aviation journal will immediately reject a methodology that relies on actual takeoff time to predict pre-takeoff taxi duration.
  2. **Unannounced Blanking:** The organizers may realize `AOBT_3_flt` was inadvertently left unblanked and reissue `ranking.parquet`. If they do, your primary pipeline (`v3`) instantly collapses, dropping your score from 359s to >512s.
- **Verdict:** Banking solely on permissive `v3` without an officially clarified Discord confirmation is an unacceptable single point of failure.

---

### 1.2 Circular Validation & Early-Stopping Contamination
- **The Grilling Question:** *Why do you call Jan+Jul 2025 an "untouched holdout" when your training scripts stop early on it and your prediction script hardcodes iterations derived from it?*
- **The Reality:**
  - In `train.py` line 79: `m.fit(..., eval_set=Pool(Xv, yv), early_stopping_rounds=100)`.
  - In `per_airport.py` line 47: `eval_set=Pool(Xv[val_apt == apt], yv[val_apt == apt])`.
  - In `joinfail.py` lines 48 & 55: early-stopping directly on `Xvf[jv]` and `Xvf[vv]`.
  - In `predict.py` lines 94–95: `("jf", jtr, 400, 77), ("jf_LIRF", ..., 700, 78)`.
- **The Consequence:**
  The reported 359.2s RMSE is not an out-of-sample evaluation—it is an **in-sample validation minimum**. When a model evaluates itself against the very dataset used to decide when to stop gradient boosting, the score is artificially optimistic.

---

### 1.3 The Refit Discrepancy & In-Sample Target Encoding Leak (`predict.py`)
- **The Grilling Question:** *Why does your final submission model (`predict.py`) train on completely different data dynamics than your validated model (`train.py`)?*
- **The Reality:**
  1. **Tree Count Explosion:** On holdout, LFPG early-stopped at **695 trees** because deeper iterations overfit. In `predict.py`, the LFPG dedicated head is fitted with `iterations=1500` without early stopping! Inspection of `models/cat_v1_full_LFPG.cbm` confirms it has **1500 trees**—more than double its optimal depth.
  2. **Target Encoding Mismatch:** In `train.py`, target encodings for the training pool are generated using 5-fold Out-Of-Fold (`oof`) cross-validation with noise. In `predict.py`, lines 50–51 apply full in-sample target encodings: `te_tr = apply_maps(train, maps, counts, gmean)`. Trees splitting on `RUNWAY_mvt_te` in the final refit see zero cross-fold variance, causing them to memorize small-sample categories.

---

### 1.4 The Stand Zone Regex Disaster (`features.py`)
- **The Grilling Question:** *Did anyone actually inspect the extracted `STAND_ZONE` feature on European airports?*
- **The Reality:**
  Line 29 of `features.py` defines:
  ```python
  def stand_zone(s):
      return s.str.extract(r"^([A-Z]+)", expand=False).fillna("UNK")
  ```
  At major European airports, stands are predominantly numeric (e.g. Heathrow stands `537`, `328`; Rome stands `409`, `605`; Munich stands `583`, `245`). Because the regex only matches leading letters `^[A-Z]+`, numeric stands produce `NaN` and get assigned `"UNK"`.
- **Empirical Audit:**
  - **EGLL (Heathrow): 100.0% UNK**
  - **LIRF (Rome Fiumicino): 100.0% UNK**
  - **EDDM (Munich): 99.3% UNK**
  - **LEBL (Barcelona): 99.2% UNK**
  - **LEMD (Madrid): 69.2% UNK**
- **Impact:**
  Because raw `STAND_mvt` was excluded from both `CATS` and `OOF_COLS`, **the models at Heathrow, Rome, Munich, and Barcelona have ZERO spatial gate/terminal information.** At Heathrow, taxi-out from Terminal 5 (west) vs Terminal 2 (central) differs by several kilometers of taxiway, but your model treats every flight at Heathrow as parking in the same unknown spot.

---

### 1.5 Month Discrepancies & Raw Numeric Month Extrapolation
- **Discrepancy:** `validation.py` defines month via `BLOCK_TIME_UTC_mvt.dt.month`, whereas `features.py` defines month via `SCHED_TIME_UTC_mvt.dt.month`. There are 84 cross-month boundary flights in the dataset where these diverge.
- **Tree Extrapolation:** `month` is included as a raw numeric feature (`NUMS_BASE = ["month", ...]`). In the training pool (Feb–Jun, Aug–Dec), `month` takes values $\{2, 3, 4, 5, 6, 8, 9, 10, 11, 12\}$. In ranking test data, `month` is strictly $\{1, 7\}$. Decision trees cannot extrapolate trend lines outside their training intervals; splitting on raw numeric month creates arbitrary step-functions for test months.

---

### 1.6 Brittle Specialist Hard-Patching (`joinfail.py`)
- The improvement from 422.9s to 359.2s relies on hard-replacing predictions for `is_join_fail == 1` rows with a dedicated specialist model.
- However, as acknowledged in `FINDINGS.md`, on EDDF, EGLL, and LEMD, the base model actually beats the specialist (e.g., EDDF 380s vs 526s).
- Hard binary routing based on a 1.5% sample slice introduces high variance. If July 2026 exhibits even a slight shift in missingness etiology (e.g. a specific low-cost carrier failing NM joins while parking at near gates), this unregularized patch will degrade.

---

## 2. Code Quality & Software Engineering Audit

### 2.1 Modularity & The DRY Principle (Don't Repeat Yourself)
The codebase exhibits significant copy-paste drift:
- `build()` and `fill_nans()` are duplicated or reimplemented across `train.py`, `per_airport.py`, `joinfail.py`, `predict.py`, and `evaluate.py`.
- In `per_airport.py`, an earlier bugfix was required because `train.py` had updated its median-filling logic while `per_airport.py` had not.
- **Recommendation:** Centralize feature processing and dataset assembly into a single deterministic `Pipeline` class.

### 2.2 Global State & Environment Variable Antipattern
Pipeline execution is heavily governed by mutable environment variables:
```python
STRICT = os.environ.get("STRICT", "0") == "1"
V2 = os.environ.get("V2", "0") == "1"
MODEL = os.environ.get("MODEL", "both")
```
- Module-level constants like `TAG` are evaluated upon import. If `V2` is toggled inside a Python session or imported across modules, `TAG` does not update dynamically.
- CLI execution should use standard `argparse` or `typer` interfaces with explicit configuration objects, not ambient environment variables.

### 2.3 Testing & CI Readiness
- Running `pytest` fails with `ModuleNotFoundError: No module named 'src'` because `likable-river/` lacks a `pyproject.toml` or `pytest.ini` with `pythonpath = ["."]`.
- Existing tests (`test_submission.py`) only verify row counts and negative numbers. There are zero unit tests for feature transformations, target encodings, or split boundaries.

### 2.4 Version Control & License Hygiene
- The directory is not initialized as a git repository (`fatal: not a git repository`).
- The challenge requires prize winners to publish an open-source repository under **GNU GPLv3**. Currently, no `LICENSE` file exists in the directory.

---

## 3. Engineering & Operational Recommendations

### Priority 0: Immediate Risk Mitigation & Competition Integrity
1. **Clarify the Leakage Policy on Discord Immediately:**
   Post in the official OpenSky `#prc-data-competition` Discord channel to verify whether `MVT_TIME_UTC_mvt` arithmetic and `AOBT_3_flt` are acceptable for final scoring and the JOAS paper.
2. **Establish a Dual-Track Submission Strategy:**
   - **Track A (Permissive / Benchmark):** `likable-river_v3.parquet` (current 359s lineage).
   - **Track B (Strict / Operational):** A dedicated, fully tuned strict model (no `MVT` arithmetic, no `AOBT_3`), ready to submit instantly if organizers disqualify Track A.
3. **Initialize Git & Add GPLv3 License:**
   Run `git init`, add `.gitignore`, commit clean baseline code, and add `LICENSE` (GPLv3).

---

### Priority 1: Feature Engineering & Bug Fixes
1. **Fix `stand_zone` for Numeric Airports (High ROI):**
   Replace the simplistic regex with an airport-aware terminal/apron extractor:
   ```python
   def extract_stand_zone(stand_col, adep_col):
       # For EGLL, stand '537' -> Terminal '5', stand '320' -> Terminal '3'
       # For EDDM, stand '219' -> Pier '2'
       # For EDDF, stand 'B41' -> Concourse 'B'
       # Combine airport + extracted zone: e.g. "EGLL_T5", "EDDF_B"
   ```
   *Expected Gain:* Recovering spatial gate information for EGLL and LIRF will significantly reduce per-airport variance without any leakage risk.
2. **Fix `month` Feature Representation:**
   Drop raw numeric `month` from tree splits. Instead, represent seasonality via:
   - Categorical month (with unseen handling).
   - Cyclical encoding ($\sin/\cos$ of day of year).
   - Or omit month entirely from tree splits and rely on calendar-invariant congestion and weather features.
3. **Refactor Congestion Features (`F5`):**
   The rejection of `v2` congestion was caused by dumping 35 collinear raw and log counts into depth-8 trees.
   - Reduce to 3 high-signal features: `dep_queue_pressure_30m` (DEP count / active runway count), `arr_arrival_rate_30m`, and `rwy_load_15m`.
   - Normalize counts by typical airport throughput.

---

### Priority 2: Validation Rigor & Refit Alignment
1. **Implement Nested Out-Of-Fold Validation:**
   - Split the 2025 non-holdout pool into 5 temporal folds (or rolling-origin quarterly splits).
   - Use inner folds for early-stopping and hyperparameter tuning.
   - Keep Jan+Jul 2025 strictly for final offline evaluation—**never pass it as `eval_set` during model training**.
2. **Align Refit Logic in `predict.py`:**
   - In `predict.py`, enforce true K-fold OOF target encodings when fitting on full 2025 data.
   - Cap per-airport head iterations to the true optimal tree count discovered during validation (e.g. stop LFPG at ~700 trees instead of 1500).
3. **Replace Hard Binary Patching with Blended Specialists:**
   Instead of `v[jv] = pg[jv]` (binary switch), use soft logistic mixing or a stacking meta-regressor on join-fail rows to avoid abrupt boundary degradation.

---

### Priority 3: Engineering, Tooling & Testing
1. **Add `pyproject.toml`:**
   ```toml
   [tool.pytest.ini_options]
   pythonpath = ["."]
   testpaths = ["tests"]
   ```
   This ensures `pytest` runs cleanly without manual `python -m` wrappers.
2. **Add Comprehensive Pipeline Tests:**
   - `test_features.py`: Assert `extract_stand_zone` returns non-UNK on EGLL, EDDM, and LIRF.
   - `test_splits.py`: Assert train and validation date intervals are strictly disjoint.
   - `test_leakage.py`: Assert strict pipeline contains zero references to `AOBT_3`, `MVT_TIME`, or `BLOCK_TIME`.
3. **Unified CLI with Configuration Dataclasses:**
   Replace `os.environ` flags with structured config files (YAML or `dataclasses`) driven by CLI commands:
   ```bash
   python -m src.pipeline train --config configs/v1_strict.yaml
   python -m src.pipeline predict --config configs/v3_permissive.yaml
   ```

---

## 4. Super-Model Blueprint: Winning Machine Learning Architecture

To move from a baseline GBDT blend to a podium-winning solution in the PRC Data Challenge 2026, the machine learning architecture must directly exploit the physical dynamics of airport ground operations, solve the extreme quadratic RMSE tail penalty, and leverage multi-paradigm ensembling.

```
┌─────────────────────────────────────────────────────────────┐
│                       Raw Features                          │
└──────┬──────────────┬────────────────┬──────────────┬───────┘
       │              │                │              │
       ▼              ▼                ▼              ▼
  ┌──────────┐  ┌───────────┐   ┌────────────┐  ┌───────────┐
  │ CatBoost │  │ LightGBM  │   │  XGBoost   │  │  Tabular  │
  │(Oblivious│  │(Leaf-wise │   │(Depth-wise │  │  ResNet / │
  │  Trees)  │  │   Trees)  │   │ Histogram) │  │  MLP Embed│
  └────┬─────┘  └─────┬─────┘   └─────┬──────┘  └─────┬─────┘
       │              │               │               │
       └──────────────┼───────────────┴───────────────┘
                      ▼
        ┌───────────────────────────┐
        │  Level-2 Stacking Regressor│  (Constrained Non-Negative Ridge)
        └─────────────┬─────────────┘
                      ▼
              Final Prediction
```

### 4.1 Target Decomposition: Two-Stage Kinematic + Congestion Modeling
Taxi-out duration is physically the sum of two distinct processes:
$$\text{TaxiOut} = \text{UnimpededTaxiTime}(Stand, Runway, Aircraft) + \text{QueueDelay}(Congestion, Slots, Weather)$$

1. **Stage 1: Unimpeded Taxi Time (Kinematic Baseline)**
   - Train a dedicated model or estimate unimpeded taxi time ($U$) using flights during unconstrained off-peak periods (e.g., 02:00–05:00 UTC or periods with congestion count = 0).
   - $U$ is purely geometric: distance from stand/apron to runway threshold, pushback heading, and aircraft taxi speed.
   - For rare combinations, back off hierarchically: `[ADEP, RUNWAY, TERMINAL] → [ADEP, RUNWAY] → [ADEP]`.
2. **Stage 2: Queue & Congestion Delta Regressor**
   - Target for Stage 2: $\Delta = \text{TaxiOut} - U$.
   - By modeling $\Delta$ (which is lower variance and strictly non-negative), tree splits focus exclusively on congestion, runway sequencing, wake turbulence separations, and slot holding, rather than re-learning physical runway distances on every split.
   - Final Prediction: $\hat{y} = U + \hat{\Delta}$.

---

### 4.2 Loss Function Engineering: Tail-Calibrated Asymmetric RMSE Optimization
In an RMSE competition, top 1% outliers represent over 50% of the total loss. However, standard Huber loss collapses (giving 668s vs 448s) because it optimizes the median rather than the conditional expectation $E[Y|X]$.
- **Asymmetric Weighted MSE:**
  Train with a custom gradient where positive residuals (underpredicting long taxis) carry higher weight than negative residuals:
  $$L(y, \hat{y}) = \begin{cases} (y - \hat{y})^2 & \text{if } y \le \hat{y} \\ (1 + \alpha)(y - \hat{y})^2 & \text{if } y > \hat{y} \end{cases} \quad (\alpha \in [0.15, 0.30])$$
  This directly corrects the observed top-decile negative bias (where the baseline underpredicts extreme delays by ~798s).

```python
def asymmetric_rmse_objective(preds, train_data):
    labels = train_data.get_label()
    diff = preds - labels
    alpha = 0.25  # Tune between 0.15 and 0.30 on holdout
    # When diff < 0 (model underpredicts, label > pred), apply higher penalty
    grad = np.where(diff < 0, 2.0 * (1.0 + alpha) * diff, 2.0 * diff)
    hess = np.where(diff < 0, 2.0 * (1.0 + alpha), 2.0)
    return grad, hess
```

- **Hurdle / Mixture Model for Tail Extremes:**
  - Binary Classifier: Predict probability $P(\text{delay} > 3600s)$.
  - Bulk Regressor: Trained on $y \le 3600s$.
  - Tail Regressor: Dedicated model trained with sample weights on severe holding events.
  - Final Prediction: $\hat{y} = (1 - P) \cdot \hat{y}_{\text{bulk}} + P \cdot \hat{y}_{\text{tail}}$.

---

### 4.3 The "Holy Trinity" + Tabular Deep Learning Ensemble
Currently, the pipeline uses only CatBoost and LightGBM. A winning tabular ensemble requires four orthogonal model families whose error residuals are uncorrelated ($r < 0.85$):

1. **CatBoost (Symmetric Trees):**
   - Best for high-cardinality categorical combinations (`RUNWAY x STAND_ZONE`, `OPERATOR x ADEP`).
   - Hyperparameters: `subsample=0.8`, `l2_leaf_reg=5`, `depth=7-8`.
2. **LightGBM (Leaf-wise Asymmetric Trees):**
   - Catches deep hierarchical splits and extreme values in numeric features (`m_sched`, `sched_lobt`).
   - Hyperparameters: `num_leaves=127`, `min_child_samples=100`, `feature_fraction=0.7`, `bagging_fraction=0.8`.
3. **XGBoost (Histogram / Exact Depth-wise):**
   - Currently absent from the repo. XGBoost with `tree_method='hist'` and `grow_policy='lossguide'` discovers orthogonal decision boundaries to CatBoost's oblivious trees.
   - Set `max_depth=9`, `colsample_bytree=0.7`, `subsample=0.8`.
4. **Tabular Deep Learning (Entity Embeddings + ResNet MLP):**
   - Neural architectures yield residuals that have low correlation ($r < 0.85$) with tree-based predictions, making them exceptionally powerful in ensembling.
   - Architecture:
     - Categorical entity embeddings (dimension $\approx \min(50, (cardinality+1)/2)$ for `ADEP`, `RUNWAY`, `STAND`, `OPERATOR`, `AIRCRAFT_TYPE`).
     - Dense numeric input with batch norm and Swish activations.
     - 3-block ResNet tabular MLP with skip connections and dropout ($p=0.15$).

---

### 4.4 Spatial & Interaction Embeddings (Matrix Factorization)
- Build a learned interaction embedding between Stand Clusters and Runways:
  $$E_{\text{spatial}} = \mathbf{e}_{\text{stand}} \cdot \mathbf{e}_{\text{runway}}$$
- Alternatively, perform truncated SVD / NMF on the co-occurrence matrix of `(STAND x RUNWAY)` median taxi times to extract a 4-dimensional latent spatial distance vector for every airport.

---

### 4.5 Multi-Seed Bagging & Variance Reduction
- Individual tree runs suffer from stochastic split variance.
- Retrain each GBDT model across **5 random seeds** (e.g., seeds 42, 1337, 2026, 77, 999) with `subsample=0.8` and `feature_fraction=0.8`.
- Averaging 5 seeds of CatBoost + 5 seeds of LightGBM + 5 seeds of XGBoost provides an immediate, consistent **1.5 to 3.0s RMSE reduction** purely by canceling out variance in leaf estimates:
  $$\hat{y}_{\text{bag}} = \frac{1}{5} \sum_{s=1}^5 \hat{y}_{\text{model}, s}$$

---

### 4.6 Out-Of-Fold (OOF) Stacking Meta-Learner
Replace the manual scalar weight grid (`w * pc + (1-w) * pl`) with Level-2 Stacking:
1. Generate 5-fold out-of-fold prediction vectors for all training flights from each base model (CatBoost, LightGBM, XGBoost, TabNet/ResNet).
2. Train a constrained Level-2 Meta-Learner:
   - Non-negative Ridge Regression or ElasticNet:
     $$\min_{\mathbf{w} \ge 0, \sum w_i = 1} \sum \left(y_i - \sum_m w_m \hat{y}_{im}\right)^2$$
   - Airport-conditioned stacking: Let weights vary per airport ($\mathbf{w}_{\text{EGLL}}$, $\mathbf{w}_{\text{LIRF}}$, $\mathbf{w}_{\text{LFPG}}$), allowing CatBoost to take higher weight on complex gate layouts while LightGBM takes higher weight on numeric schedule deltas.

---

### 4.7 Post-Processing & Monotonic Boundary Calibration
- **Zero-Floor and Flight Duration Cap:**
  Ensure $\hat{y} \ge \max(60, \text{airport\_min\_taxi})$. Taxi times below 60 seconds at major hubs (e.g. Heathrow or Charles de Gaulle) are physical impossibilities.
- **Quantile-Guided Tail Shrinkage:**
  Train a LightGBM quantile regressor at $\alpha = 0.99$. If the point prediction $\hat{y}_{\text{point}} > \hat{y}_{q99}$, shrink the excess prediction toward $\hat{y}_{q99}$ with a shrinkage factor $\gamma = 0.85$ to prevent isolated over-predictions from causing catastrophic quadratic error explosions on ranking data:
  $$\text{If } \hat{y}_{\text{point}} > \hat{y}_{q99}: \quad \hat{y}_{\text{final}} = \hat{y}_{q99} + \gamma (\hat{y}_{\text{point}} - \hat{y}_{q99})$$

---

## 5. Implementation Roadmap & Priority Matrix

| Phase | Milestone | Specific ML Action | Expected Impact |
| :--- | :--- | :--- | :---: |
| **Phase 1** | **Diversity & Bagging** | Integrate **XGBoost** (`tree_method='hist'`) and implement **5-seed bagging** across CatBoost + LightGBM + XGBoost. | **-3.0 to -5.0s RMSE** |
| **Phase 2** | **Target Decomposition** | Decouple taxi-out into **Unimpeded Taxi ($U$) + Congestion ($\Delta$)**; model $\Delta$ with queue features. | **-6.0 to -10.0s RMSE** |
| **Phase 3** | **Tail Loss Optimization** | Deploy **Asymmetric Weighted MSE** ($\alpha=0.25$) to eliminate the 798s negative bias on peak congestion. | **-8.0 to -15.0s RMSE** (LIRF/EGLL) |
| **Phase 4** | **Spatial Topology** | Extract **Stand-Runway SVD embeddings** and fix the `stand_zone` parsing for EGLL, LIRF, EDDM, LEBL. | **-4.0 to -8.0s RMSE** |
| **Phase 5** | **Deep Stacking** | Train a **Tabular ResNet with Entity Embeddings** and stack all models using **Airport-Conditioned Ridge**. | **-4.0 to -7.0s RMSE** |

---

## 6. Conclusion

The `likable-river` team has established a competitive foundation with sophisticated error slicing and targeted mitigation of tail loss at LIRF. However, transitioning from an experimental prototype to a bulletproof, competition-winning solution requires eliminating circular early-stopping, fixing the critical stand-zone parsing bug, aligning refit tree counts, formalizing a strict operational backup track, and deploying this multi-paradigm machine learning super-model architecture.
