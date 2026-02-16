# NIR Moisture Prediction Project

## Domain Context

This project analyzes Near-Infrared (NIR) spectra of pharmaceutical tablets to predict moisture content using chemometric methods. Moisture is a critical quality attribute affecting tablet stability, dissolution, and shelf life.

**Analytical method:** NIR spectroscopy (1100-2500 nm, 2 nm resolution)
**Reference method:** Karl Fischer titration (% w/w moisture)
**Sample type:** Pharmaceutical tablets from multiple production batches
**Regulatory context:** ICH Q2(R1) guideline for analytical procedure validation

## Data Structure

### Input Files

- `data/nir_spectra.csv`: NIR absorbance spectra
  - Rows: Samples (n=80)
  - Columns: Wavelengths (1100-2500 nm, 700 wavelengths)
  - Index: Sample IDs
  - Units: Absorbance (dimensionless)

- `data/reference_moisture.csv`: Reference moisture values
  - Columns: ['sample_id', 'moisture_pct', 'batch', 'measurement_date']
  - Units: moisture_pct is % w/w (0-5% typical range)
  - Batch: Production batch identifier (for group CV if needed)

- `data/metadata.csv` (optional): Sample metadata
  - Columns: ['sample_id', 'tablet_weight_mg', 'hardness_n', 'storage_condition']

### Output Structure

```
results/
├── figures/
│   ├── 01_spectra_overview.png
│   ├── 02_pca_scores.png
│   ├── 03_cv_components.png
│   ├── 04_predicted_vs_actual.png
│   ├── 05_loadings_plot.png
│   └── 06_residuals.png
├── models/
│   ├── pls_moisture_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── reports/
│   ├── analysis_report.md
│   └── validation_metrics.csv
└── data_processed/
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    └── y_test.csv
```

## Analysis Requirements

### Data Splitting

1. **Train/test split:** 80/20 ratio
2. **Stratification:** Use stratified sampling based on moisture quartiles to ensure representative distribution
3. **Random seed:** Always use `random_state=42` for reproducibility
4. **Batch awareness:** If using batch information, consider GroupKFold to avoid batch effects

```python
# Example code structure
from sklearn.model_selection import train_test_split
import pandas as pd

# Stratify by moisture quartiles
y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_bins, random_state=42
)
```

### Preprocessing

1. **Primary method:** Standard Normal Variate (SNV)
   - Corrects for multiplicative scatter effects
   - Applied per-spectrum (row-wise)
   - Alternative: Multiplicative Scatter Correction (MSC)

2. **Wavelength selection (optional):**
   - If needed, focus on water absorption bands: 1400-1500 nm, 1900-2000 nm
   - Or use full spectrum with proper regularization

3. **Outlier detection:**
   - Use Mahalanobis distance or Hotelling's T² before modeling
   - Remove only clear outliers (D > critical value at α=0.05)
   - Document any removed samples

```python
# SNV preprocessing
def snv(spectra):
    return (spectra - spectra.mean(axis=1, keepdims=True)) / \
           spectra.std(axis=1, keepdims=True, ddof=1)

X_train_snv = snv(X_train)
X_test_snv = snv(X_test)
```

### Modeling Approach

1. **Primary model:** Partial Least Squares (PLS) Regression
   - Industry standard for NIR calibration
   - Use `sklearn.cross_decomposition.PLSRegression`

2. **Component selection:**
   - Use 10-fold cross-validation on training set
   - Test range: 1-20 components (or n_samples // 4, whichever is smaller)
   - Select based on maximum R²_CV or minimum RMSECV
   - Watch for plateau (adding components doesn't improve)

3. **Alternative models (for comparison):**
   - Ridge Regression (linear baseline)
   - Support Vector Regression (SVR) if non-linear relationships suspected
   - Random Forest (for comparison, may not be optimal for NIR)

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold

# Determine optimal components
cv = KFold(n_splits=10, shuffle=True, random_state=42)
n_components_range = range(1, 21)
cv_scores = []

for n in n_components_range:
    pls = PLSRegression(n_components=n)
    scores = cross_val_score(pls, X_train_snv, y_train, cv=cv, scoring='r2')
    cv_scores.append(scores.mean())

optimal_n = n_components_range[np.argmax(cv_scores)]
print(f"Optimal components: {optimal_n}")
```

### Validation Requirements

1. **Cross-validation:** 10-fold on training set
2. **Test set:** Hold-out 20% for final unbiased evaluation
3. **Metrics to report:**
   - **Primary:** RMSEP (Root Mean Square Error of Prediction)
   - **Secondary:** R² (coefficient of determination), RPD (Ratio of Performance to Deviation)
   - **Diagnostic:** Bias, SEP (Standard Error of Prediction)

4. **Performance thresholds (ICH-inspired):**
   - RMSEP < 0.3% w/w (acceptable for this application)
   - R² > 0.90 (excellent), > 0.85 (good), > 0.80 (acceptable)
   - RPD > 2.5 (excellent quantitative), > 2.0 (good), > 1.8 (screening)
   - |Bias| < 0.1% w/w (no systematic error)

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Test set evaluation
y_pred_test = pls_model.predict(X_test_snv).ravel()

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)
rpd = y_test.std() / rmse
bias = (y_pred_test - y_test).mean()

print(f"RMSEP: {rmse:.3f} % w/w")
print(f"R²:    {r2:.3f}")
print(f"RPD:   {rpd:.2f}")
print(f"Bias:  {bias:.3f} % w/w")

# Interpretation
if rpd > 2.5:
    print("✅ Excellent quantitative prediction capability")
elif rpd > 2.0:
    print("✓ Good quantitative predictions")
elif rpd > 1.8:
    print("⚠ Fair predictions (suitable for screening)")
else:
    print("❌ Poor prediction capability - model not suitable")
```

### Model Interpretation

1. **PLS loadings plot:**
   - Show which wavelengths contribute to each component
   - Identify water absorption bands (expected around 1450, 1940 nm)

2. **VIP scores:**
   - Calculate Variable Importance in Projection
   - Threshold: VIP > 1 indicates important wavelengths

3. **Predicted vs Actual plot:**
   - Include 1:1 line (perfect prediction)
   - Show ±RMSEP error bands
   - Color-code by batch if available

4. **Residual analysis:**
   - Plot residuals vs predicted values
   - Check for patterns (indicates model inadequacy)
   - Q-Q plot to assess normality

## Skills to Use

Load these skills for domain-specific guidance:

1. **chemometrics-ml-selection**: Guide for choosing ML methods
   - Decision framework based on data size, problem type
   - PLS vs SVM vs RF comparisons
   - When to use each method

2. **chemometrics-validation**: Validation strategies and metrics
   - Cross-validation schemes
   - Performance metrics (RMSEP, RPD, etc.)
   - Small dataset handling

3. **chemometrics-preprocessing** (if available): Spectral preprocessing
   - SNV, MSC, derivatives
   - Baseline correction
   - When to use each method

4. **scikit-learn** (from K-Dense Scientific Skills): General ML
   - Workflow patterns
   - Pipeline construction
   - Best practices

## Style Guide

### Code Conventions

- Use **numpy** for numerical operations
- Use **pandas** for data frames (easier inspection)
- Use **matplotlib** or **seaborn** for plotting (save as PNG, 300 DPI)
- Use **scikit-learn** for ML models (consistent API)
- Always set `random_state=42` for reproducibility

### Variable Naming

- `X`: Spectra (samples × wavelengths)
- `y`: Reference values (1D array)
- `X_train`, `X_test`: Split data
- `X_snv`: Preprocessed spectra
- `pls_model` or `pls`: PLS regression object
- `y_pred`: Predicted values

### Documentation

- Add docstrings to any custom functions
- Comment non-obvious preprocessing steps
- Save key parameters in `model_metadata.json`
- Generate markdown report in `results/reports/analysis_report.md`

### Plotting

- Figure size: `figsize=(10, 6)` for standard plots
- DPI: 300 for publication quality, 100 for draft
- Color scheme: Use `viridis` or `RdBu_r` for continuous, `Set2` for categorical
- Always label axes with units: "Moisture Content (% w/w)", "Wavelength (nm)", "Absorbance"
- Include grid: `plt.grid(alpha=0.3)`
- Save to `results/figures/` with descriptive names

```python
# Example plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, s=50, label='Test Set')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Measured Moisture (% w/w)', fontsize=12)
plt.ylabel('Predicted Moisture (% w/w)', fontsize=12)
plt.title(f'PLS Prediction Performance (R²={r2:.3f}, RMSEP={rmse:.3f})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/04_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Common Pitfalls to Avoid

1. ❌ **Fitting scaler on full dataset before splitting**
   - This leaks information from test set into training
   - Always fit preprocessing on training set only

2. ❌ **Not checking for data leakage in cross-validation**
   - Use Pipeline to ensure preprocessing is done within CV folds
   - Or manually preprocess within each fold

3. ❌ **Using default hyperparameters without tuning**
   - Always determine optimal n_components via CV
   - Don't just use n_components=10 by default

4. ❌ **Reporting only R² without RMSEP or RPD**
   - R² doesn't tell you prediction error in original units
   - RPD is standard in chemometrics for calibration quality

5. ❌ **Not verifying model performance meets requirements**
   - Check if RMSEP < 0.3% and RPD > 2.0 before concluding success
   - Compare to analytical method precision

6. ❌ **Ignoring physical constraints**
   - Moisture predictions should be in 0-100% range
   - If predictions violate this, model may be inappropriate

## Example Workflow

Here's the complete workflow you should follow:

1. **Load data** → Check dimensions, missing values
2. **Explore** → Plot spectra, check reference value distribution, detect outliers
3. **Split** → 80/20 stratified by moisture quartiles (random_state=42)
4. **Preprocess** → Apply SNV to training and test sets separately
5. **Model selection** → PLS with CV to find optimal components
6. **Train** → Fit final PLS model on full training set with optimal components
7. **Evaluate** → Predict on test set, calculate RMSEP, R², RPD, Bias
8. **Interpret** → Loadings, VIP scores, predicted vs actual plot
9. **Validate** → Check performance against thresholds (RMSEP < 0.3, RPD > 2.5)
10. **Document** → Save model, generate report with all metrics and plots

## Additional Notes

- If analysis takes a different direction based on intermediate results, document the reasoning
- If model performance is poor, consider:
  - Different preprocessing (try MSC, derivatives)
  - Non-linear models (SVR, Random Forest)
  - Wavelength selection (focus on water bands)
  - Check for batch effects or outliers
- Always save intermediate results for reproducibility
- If using Claude Code, leverage the chemometrics skills for guidance at each step

## Questions or Clarifications

If uncertain about any aspect of the analysis:

1. Consult the chemometrics-ml-selection skill for method choice
2. Consult the chemometrics-validation skill for metrics interpretation
3. Refer to the Trinh et al. 2021 paper in assets/ for ML best practices
4. Ask for clarification before proceeding with uncertain decisions

---

**Ready to start?** Load this CLAUDE.md file in your working directory and begin the analysis with Claude Code CLI!
