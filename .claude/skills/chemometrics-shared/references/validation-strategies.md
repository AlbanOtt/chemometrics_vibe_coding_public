# Cross-Validation Strategies

## 1. Train/Test Split

**When:** Large datasets (n > 200), final evaluation after CV tuning.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Classification: stratified split (preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Regression: stratified by binned y values
y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_bins, random_state=42
)
```

**Best practices:**
- Use 70/30 or 80/20 splits
- Always set `random_state` for reproducibility
- Stratify whenever possible
- Never use test set until final evaluation
- Report both training and test performance

## 2. K-Fold Cross-Validation

**When:** Medium datasets (50 < n < 200), model comparison, hyperparameter tuning.

```python
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold
import numpy as np

# Regression: Standard K-Fold
cv = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
print(f"R² = {scores.mean():.3f} ± {scores.std():.3f}")

# Classification: Stratified K-Fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)

# Multiple metrics at once
scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
results = cross_validate(
    model, X, y, cv=cv, scoring=scoring,
    return_train_score=True, n_jobs=-1
)
print(f"Test R²: {results['test_r2'].mean():.3f}")
print(f"Train R²: {results['train_r2'].mean():.3f}")
print(f"Test RMSE: {np.sqrt(-results['test_neg_mean_squared_error'].mean()):.3f}")
```

**Best practices:**
- k=5 or k=10 (10 is standard for reporting)
- Always shuffle unless time series
- Use StratifiedKFold for classification
- Report mean +/- std across folds
- Check train vs test scores for overfitting

## 3. Leave-One-Out Cross-Validation (LOOCV)

**When:** Very small datasets (n < 30), linear models.
**Avoid:** Large datasets, high-variance models, non-linear models.

```python
from sklearn.model_selection import LeaveOneOut, ShuffleSplit

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
print(f"LOO R² = {scores.mean():.3f}")

# More stable alternative: repeated random sub-sampling
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
print(f"Repeated random R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

**Caution:** LOOCV has high variance — consider repeated random sub-sampling as alternative.

## 4. Monte Carlo Cross-Validation

**When:** Small to medium datasets, want more stable estimates than LOOCV.

```python
from sklearn.model_selection import ShuffleSplit
import numpy as np

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
print(f"Monte Carlo CV R² = {scores.mean():.3f} ± {scores.std():.3f}")
print(f"95% CI: [{np.percentile(scores, 2.5):.3f}, {np.percentile(scores, 97.5):.3f}]")
```

## 5. Time Series Cross-Validation

**When:** Process monitoring, sequential measurements, any time-ordered data.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
print(f"Time Series CV R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

## 6. Group Cross-Validation

**When:** Multiple measurements from same sample, batch effects, multi-site studies.

```python
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

# Leave-one-group-out
logo = LeaveOneGroupOut()
scores = cross_val_score(model, X, y, groups=groups, cv=logo, scoring='r2')

# Group K-Fold
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='r2')
print(f"Group CV R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

**Critical:** Always use group CV when samples are not independent!

## Small Dataset Strategies

### Repeated Cross-Validation

```python
from sklearn.model_selection import RepeatedKFold
import numpy as np

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
print(f"R² = {scores.mean():.3f} ± {scores.std():.3f}")
print(f"95% CI: [{np.percentile(scores, 2.5):.3f}, {np.percentile(scores, 97.5):.3f}]")
```

### Nested Cross-Validation

Unbiased estimate when tuning hyperparameters:

```python
from sklearn.model_selection import GridSearchCV, KFold

param_grid = {'n_components': range(1, 21)}
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(PLSRegression(), param_grid, cv=inner_cv, scoring='r2')

outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=outer_cv, scoring='r2')
print(f"Nested CV R² = {scores.mean():.3f} ± {scores.std():.3f}")
```
