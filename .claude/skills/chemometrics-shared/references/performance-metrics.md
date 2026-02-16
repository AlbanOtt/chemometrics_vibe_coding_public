# Performance Metrics

## Regression Metrics

### RMSEP (Root Mean Square Error of Prediction)

The most important metric for chemometrics calibration.

```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSEP = {rmse:.3f} {units}")

# RMSECV (from cross-validation)
y_pred_cv = cross_val_predict(model, X, y, cv=10)
rmsecv = np.sqrt(mean_squared_error(y, y_pred_cv))
print(f"RMSECV = {rmsecv:.3f}")
```

### R-squared (Coefficient of Determination)

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
# Cross-validation R²
scores = cross_val_score(model, X, y, cv=10, scoring='r2')
print(f"R²_CV = {scores.mean():.3f} ± {scores.std():.3f}")
```

**Caution:** R² can be misleading with small datasets or biased predictions.

### RPD (Ratio of Performance to Deviation)

```python
rpd = y_test.std() / rmse
```

Interpretation: see SKILL.md quick reference table.

### Bias and SEP (Standard Error of Prediction)

```python
import numpy as np

bias = (y_pred - y_test).mean()
sep = np.sqrt(np.sum((y_pred - y_test - bias)**2) / (len(y_test) - 1))
# Relationship: RMSE² = Bias² + SEP²
```

### Complete Regression Report

```python
def regression_report(y_true, y_pred, units=''):
    """Comprehensive regression performance report."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    n = len(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    sep = np.sqrt(np.sum((y_pred - y_true - bias)**2) / (n - 1))
    rpd = y_true.std() / rmse
    residuals = y_pred - y_true

    print(f"Regression Performance Report (n={n})")
    print("=" * 50)
    print(f"RMSE:  {rmse:.3f} {units}")
    print(f"MAE:   {mae:.3f} {units}")
    print(f"R²:    {r2:.3f}")
    print(f"RPD:   {rpd:.2f}")
    print(f"Bias:  {bias:.3f} {units}")
    print(f"SEP:   {sep:.3f} {units}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'rpd': rpd, 'bias': bias, 'sep': sep}
```

## Classification Metrics

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')

# Normalize (show percentages)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
```

### Sensitivity and Specificity

```python
from sklearn.metrics import confusion_matrix, classification_report

# Binary classification
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate, Recall
specificity = tn / (tn + fp)  # True Negative Rate
precision = tp / (tp + fp)    # Positive Predictive Value

# Multi-class
print(classification_report(y_test, y_pred, target_names=class_names))
```

**Which matters more?**
- Medical/Safety: High sensitivity (don't miss positives)
- Quality control: Balanced sensitivity/specificity
- Rare events: Focus on precision (avoid false alarms)

### F1-Score

```python
from sklearn.metrics import f1_score

f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
```

### ROC AUC

```python
from sklearn.metrics import roc_auc_score, roc_curve

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# AUC > 0.9: Excellent, > 0.8: Good, > 0.7: Fair, = 0.5: Random
```

### Complete Classification Report

```python
def classification_report_extended(y_true, y_pred, y_prob=None, class_names=None):
    """Comprehensive classification performance report."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    import numpy as np

    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Classification Performance Report (n={n})")
    print("=" * 60)
    print(f"Accuracy:          {acc:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")

    if y_prob is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC AUC:           {auc:.3f}")

    print("\nPer-Class Metrics:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
```
