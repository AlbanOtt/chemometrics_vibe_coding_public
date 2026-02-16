# Phase 6: Proxy Biomarker Discovery

## Context

Phases 1-5 and 7 are implemented. Phase 6 (`src/biomarkers.py`) is the only remaining gap — currently an empty placeholder with just a docstring. This phase answers the core translational question: **which skin metabolite features track plasma diphenhydramine levels closely enough to serve as non-invasive drug monitoring proxies?**

## Deliverables

1. **`src/biomarkers.py`** — Core analysis module (9 functions)
2. **`tests/test_biomarkers.py`** — Test suite (~20 tests)
3. **`reports/biomarkers_tutorial.qmd`** — Quarto tutorial report

## Data Flow

```
pickle data (per location: forearm/forehead)
    │
    ▼
pair_skin_plasma_samples()          ← reuses search_features_by_mz from drug_detection
    │── skin_indices, y_plasma_dph, subjects
    │
    ▼
X_skin_paired = X_processed[skin_indices]
    │
    ├──► correlate_skin_features_to_target()   → correlation DataFrame + FDR
    │
    └──► pls_regression_loso_cv()              → R², Q², RMSECV, VIP
              │
              ▼
         select_biomarker_candidates()          → ranked candidates (corr + VIP + FDR)
              │
              ▼
         Plot functions (volcano, pred vs obs, heatmap, summary)
```

## Implementation: `src/biomarkers.py`

### Constants
- `DEFAULT_FDR_THRESHOLD = 0.05`
- `DEFAULT_CORRELATION_THRESHOLD = 0.5`
- `DEFAULT_N_COMPONENTS_PLS = 2`
- `DEFAULT_N_TOP_CANDIDATES = 20`

### Functions

**1. `pair_skin_plasma_samples(peak_areas_raw, metadata, feature_metadata, target_mz, tolerance_ppm) -> dict`**
- Find DPH feature via `search_features_by_mz` (reuse from `drug_detection`)
- Split metadata into plasma/skin by `sample_type`
- Inner-join on `(subject, timepoint)` to pair samples
- Extract plasma DPH intensity from raw peak areas
- Returns: `skin_indices`, `y_plasma_dph`, `subjects`, `timepoints`, `n_pairs`, `dph_feature_id`, `status`, `pairing_df`
- Edge case: DPH not found → return `{"status": "dph_not_found", "n_pairs": 0}`

**2. `correlate_skin_features_to_target(X_skin, y_target, feature_ids, fdr_threshold) -> pd.DataFrame`**
- Spearman correlation per feature column against y_target
- FDR correction via `statsmodels.multipletests(method="fdr_bh")`
- Returns DataFrame: `feature_id, rho, pvalue, pvalue_fdr, abs_rho, significant` (sorted by abs_rho desc)

**3. `select_biomarker_candidates(correlation_results, vip_scores, feature_ids, fdr_threshold, vip_threshold, correlation_threshold) -> pd.DataFrame`**
- Join correlation results with VIP scores
- Filter: FDR < threshold AND VIP > threshold AND |rho| > threshold
- Rank by composite score: `abs_rho * vip`
- Returns DataFrame: `feature_id, rho, pvalue_fdr, abs_rho, vip, candidate, rank_score`

**4. `pls_regression_loso_cv(X, y, groups, n_components) -> dict`**
- Fit `PLSRegression` on all data; compute training R²
- LOSO-CV via `LeaveOneGroupOut` (groups=subjects, 7 folds)
- Compute Q² = 1 - SS_res/SS_tot, RMSECV, bias from CV predictions
- VIP from full model via `calculate_vip` (reuse from `multivariate`)
- Returns: `model, r2, q2, rmsecv, bias, vip, y_pred_cv, y_actual, subjects_cv, n_components, n_subjects`

**5. `run_biomarker_discovery(X_processed, metadata, feature_metadata, feature_ids, peak_areas_raw, ...) -> dict`**
- Orchestrator: chains functions 1→2→4→3
- Returns nested dict with `pairing`, `correlation`, `pls_result`, `candidates`, `status`

**6-9. Plot functions** (all follow `ax: Axes | None` pattern):
- `plot_correlation_volcano(correlation_results, ...) -> Axes` — rho vs -log10(FDR)
- `plot_predicted_vs_observed(pls_result, ...) -> Axes` — CV pred vs actual, colored by subject
- `plot_candidate_heatmap(X_skin_paired, candidates, ...) -> Axes` — top candidates across subject/timepoint
- `plot_biomarker_summary(discovery_result, location_name) -> Figure` — 2x2 multi-panel

### Key Imports to Reuse (no modifications to existing files)
- `src.drug_detection`: `TARGET_COMPOUNDS`, `search_features_by_mz`
- `src.multivariate`: `VIP_THRESHOLD`, `calculate_vip`

### Key Design Decisions
- **y = raw peak areas** (not Pareto-scaled) for plasma DPH — preserves absolute intensity for meaningful regression
- **X = Pareto-scaled skin features** from X_processed — PLS benefits from scaled features
- **VIP from PLS regression** (not Phase 5 PLS-DA) — VIP should reflect relevance to plasma drug prediction, not time class discrimination
- **LOSO-CV** (not k-fold) — 7 subjects, prevents within-subject data leakage

## Tests: `tests/test_biomarkers.py`

Follow patterns from `test_multivariate.py`: `matplotlib.use("Agg")`, `@pytest.fixture()` with parens, synthetic data with `default_rng(42)`.

**Fixture:** `synthetic_paired_data` — 5 subjects, 6 timepoints, 50 features, one DPH-like feature at m/z 256.1696, 3 features intentionally correlated with DPH curve.

**Test groups:**
- Pairing (5 tests): status, n_pairs, shapes, DPH-not-found edge case, y non-negative
- Correlation (4 tests): columns, length, sort order, signal detection in top 10
- Candidate selection (3 tests): columns, filter correctness, sort order
- PLS regression (4 tests): dict keys, prediction length, R² > 0, few-subjects edge case
- Orchestrator (2 tests): full pipeline, DPH-not-found
- Plot functions (4 tests): one per plot function

## Quarto Report: `reports/biomarkers_tutorial.qmd`

**Structure** (follows existing report patterns):
1. Learning Objectives (callout-note)
2. Background — translational goal, proxy biomarkers
3. Setup — standard imports, pickle loading
4. Sample Pairing — `pair_skin_plasma_samples` for both locations
5. Univariate Correlation Screening — volcano plots (1x2), top 10 table
6. PLS Regression Model — predicted vs observed (1x2), metrics table
7. Biomarker Candidate Selection — VIP bars, candidate table, heatmap
8. Location Comparison — forearm vs forehead side-by-side
9. Summary — key findings, next steps

## Verification

1. `uv run --frozen ruff check src/biomarkers.py tests/test_biomarkers.py`
2. `uv run --frozen ruff format --check src/biomarkers.py tests/test_biomarkers.py`
3. `uv run --frozen pytest tests/test_biomarkers.py -v`
4. `uv run quarto render reports/biomarkers_tutorial.qmd` (requires pickle file)

## Beads Tracking

Create a beads issue for this phase before starting implementation.
