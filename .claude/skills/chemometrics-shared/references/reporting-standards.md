# Reporting Standards

## Minimum Requirements for Publications

1. **Data splitting strategy** — "Data were split 80/20 into training and test sets using stratified random sampling"
2. **Cross-validation scheme** — "Model selection was performed using 10-fold cross-validation repeated 10 times"
3. **Performance metrics** — Report multiple: RMSEP, R², RPD (regression) or Sensitivity, Specificity, F1 (classification)
4. **Sample sizes** — "Training: n=64; Test: n=16"
5. **Hyperparameter tuning** — "Optimal number of PLS components (n=12) was determined via cross-validation"
6. **Reproducibility** — State random seeds, software versions

## Example Methods Section

```markdown
Partial Least Squares (PLS) regression was used to build a calibration
model relating NIR spectra (X) to moisture content (y). Data were split
80/20 into training (n=64) and test (n=16) sets using stratified random
sampling based on quartiles of the moisture distribution (random_state=42).

Standard Normal Variate (SNV) preprocessing was applied to correct for
multiplicative scatter effects. The optimal number of PLS components was
determined via 10-fold cross-validation on the training set, maximizing
the R² score. The selected model (10 components) was trained on the full
training set and evaluated on the held-out test set.

Performance was assessed using Root Mean Square Error of Prediction
(RMSEP), coefficient of determination (R²), and Ratio of Performance to
Deviation (RPD). Model interpretation was performed using Variable
Importance in Projection (VIP) scores and PLS loadings.

All analyses were performed in Python 3.10 using scikit-learn 1.3.0,
numpy 1.24.0, and pandas 2.0.0.
```
