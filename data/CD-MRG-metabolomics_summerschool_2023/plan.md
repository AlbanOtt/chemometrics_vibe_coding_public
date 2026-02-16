# Metabolomics Analysis Plan: Diphenhydramine Pharmacokinetics

> **After approval:** Create Beads issue with `bd create "Metabolomics analysis: Diphenhydramine pharmacokinetics in plasma vs skin"`

## Overview

Analyze LC-MS metabolomics data from skin (forearm, forehead) and plasma samples to investigate non-invasive drug monitoring of diphenhydramine.

**Research Questions:**
1. Can diphenhydramine be detected in skin with similar pharmacokinetics as plasma?
2. What metabolites show interesting time trends in plasma/skin?
3. Which skin metabolites can serve as proxies for plasma drug levels?
4. Is forearm or forehead better for monitoring?

---

## Data Summary

| File | Description |
|------|-------------|
| `data/CD-MRG-metabolomics_summerschool_2023/forearm_iimn_gnps_quant.csv` | 7171 features, 89 samples (plasma + forearm skin) |
| `data/CD-MRG-metabolomics_summerschool_2023/forehead_iimn_gnps_quant.csv` | 6378 features, 89 samples (plasma + forehead skin) |
| `data/CD-MRG-metabolomics_summerschool_2023/metadata.txt` | Sample metadata (7 subjects, 6 timepoints: 0-720 min) |

**Data structure:** Feature metadata (m/z, RT, neutral mass) + peak areas per sample. Data already processed through GNPS.

---

## Implementation Plan

### Phase 1: Project Setup

**Add packages:**
```bash
uv add seaborn statsmodels jupyter
```

**Create directory structure:**
```
src/
  __init__.py
  preprocessing.py
  eda.py
  drug_detection.py
  multivariate.py
  biomarkers.py
  location_comparison.py
reports/
  metabolomics_analysis.qmd
```

### Phase 2: Data Loading & Preprocessing (`src/preprocessing.py`)

1. **Load GNPS quant tables** - Separate feature metadata from peak areas
2. **Load metadata** - Parse subject, timepoint, sample type, location
3. **Filter blanks** - Remove blank samples and blank-dominated features
4. **Handle zeros** - Replace with min/2 per feature, filter >50% zero features
5. **Normalize** - PQN normalization (robust to biological variation)
6. **Scale** - Pareto scaling (balanced importance)
7. **Match samples** - Create plasma-skin pairs by subject/timepoint

### Phase 3: Exploratory Data Analysis (`src/eda.py`)

1. **Dataset summary** - Feature/sample counts, missing values, dynamic range
2. **Intensity distributions** - Box plots by sample type
3. **PCA quality control** - Score plots colored by sample type/subject/timepoint
4. **Outlier detection** - Hotelling's T2 statistic

### Phase 4: Diphenhydramine Detection (`src/drug_detection.py`)

**Target masses ([M+H]+):**
- Diphenhydramine: 256.1696 Da
- N-desmethyl-DPH: 242.1539 Da
- DPH N-oxide: 272.1645 Da

1. **Search by m/z** - 10 ppm tolerance
2. **Extract PK curves** - Concentration vs time per subject
3. **Calculate PK parameters** - Cmax, Tmax, AUC
4. **Compare plasma vs skin** - Correlation, time lag

### Phase 5: Multivariate Analysis (`src/multivariate.py`)

1. **PCA time trajectory** - Connect timepoints per subject
2. **PLS-DA** - Early vs late timepoints (discriminate drug effect)
3. **VIP scores** - Identify important features
4. **Time-trending features** - Spearman correlation with timepoint
5. **S-plot** - Biomarker selection (covariance vs correlation)

### Phase 6: Proxy Biomarker Discovery (`src/biomarkers.py`)

1. **Correlate skin features to plasma diphenhydramine**
2. **Select candidates** - VIP > 1, FDR-corrected p < 0.05, high correlation
3. **Build PLS regression** - Predict plasma drug from skin metabolites
4. **Evaluate performance** - R2, Q2 (LOSO-CV), RMSECV

### Phase 7: Location Comparison (`src/location_comparison.py`)

1. **Feature overlap** - Shared vs unique features between forearm/forehead
2. **Detection rate comparison** - Diphenhydramine signal in each location
3. **Model performance comparison** - Which location predicts plasma better?
4. **Generate recommendation**

### Phase 8: Quarto Report (`reports/metabolomics_analysis.qmd`)

**Structure:**
1. Executive Summary
2. Introduction (background, research questions)
3. Methods (study design, data processing, statistical analysis)
4. Results
   - Data quality and PCA
   - Diphenhydramine detection and PK curves
   - Multivariate analysis and time-trending features
   - Proxy biomarker candidates
   - Forearm vs forehead comparison
5. Discussion
6. Supplementary Materials

---

## Key Technical Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Normalization | PQN | Robust to biological variation between plasma/skin |
| Scaling | Pareto | Balanced importance, doesn't overweight noise |
| Missing values | Min/2 imputation | Standard metabolomics approach |
| Cross-validation | LOSO (Leave-One-Subject-Out) | Prevents data leakage with 7 subjects |
| Mass tolerance | 10 ppm | Standard for high-resolution MS |

---

## Critical Files to Modify/Create

**Create:**
- `src/preprocessing.py` - Data loading and cleaning functions
- `src/eda.py` - Exploratory analysis and visualization
- `src/drug_detection.py` - Diphenhydramine/metabolite search
- `src/multivariate.py` - PCA, PLS-DA, VIP calculation
- `src/biomarkers.py` - Correlation analysis and proxy selection
- `src/location_comparison.py` - Forearm vs forehead comparison
- `reports/metabolomics_analysis.qmd` - Final Quarto report

**Reference (reuse code patterns from):**
- `.claude/skills/chemometrics-ms-metabolomics/skill.md` - PCA, PLS-DA, VIP, normalization implementations

---

## Verification Plan

1. **Data loading** - Verify feature counts match (7171 forearm, 6378 forehead)
2. **Preprocessing** - Check no NaN values after imputation, normalized data centered around 0
3. **Drug detection** - Confirm diphenhydramine feature found at m/z ~256.17
4. **PCA** - Verify plasma/skin separation in score plots
5. **PLS-DA** - Check Q2 > 0 (permutation test p < 0.05)
6. **Report** - Render Quarto to HTML successfully

**Run commands:**
```bash
uv run python -c "from src.preprocessing import load_gnps_quant_table; print('OK')"
uv run quarto render reports/metabolomics_analysis.qmd
```
