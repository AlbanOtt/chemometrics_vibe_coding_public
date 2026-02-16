# ML Method Details for Chemometrics

Detailed descriptions, usage guidance, and code examples for each core ML method used in chemometrics.

## 1. Partial Least Squares (PLS)

**Best for:** Regression with high-dimensional, collinear data

**Characteristics:**
- Handles multicollinearity naturally
- Works well with n < p (more variables than samples)
- Provides interpretable loadings and VIP scores
- Industry standard for spectroscopy calibration

**When to use:**
- NIR/IR/Raman calibration models
- Small to medium datasets (n = 20-200)
- Need interpretable results (loading plots)
- Linear or near-linear relationships

**Python example:**
```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

# Determine optimal number of components
n_components_range = range(1, min(20, X_train.shape[0] // 2))
cv_scores = []

for n in n_components_range:
    pls = PLSRegression(n_components=n)
    scores = cross_val_score(pls, X_train, y_train,
                             cv=5, scoring='r2')
    cv_scores.append(scores.mean())

optimal_n = n_components_range[np.argmax(cv_scores)]

# Final model
pls_model = PLSRegression(n_components=optimal_n)
pls_model.fit(X_train, y_train)

# Interpret: VIP scores
vip = np.sqrt(len(pls_model.x_weights_) *
              np.sum(pls_model.x_weights_**2 *
                     np.array([pls_model.x_scores_[:, i].var()
                               for i in range(optimal_n)]), axis=1) /
              np.sum([pls_model.x_scores_[:, i].var()
                      for i in range(optimal_n)]))
```

## 2. Support Vector Machines (SVM/SVR)

**Best for:** Small datasets with complex, non-linear relationships

**Characteristics:**
- Excellent for n < 100 samples
- Handles non-linear patterns via kernels (RBF, polynomial)
- Robust to outliers (with soft margin)
- Less interpretable than PLS

**When to use:**
- Non-linear calibrations
- Classification with clear decision boundaries
- Small datasets where neural networks would overfit
- When robustness to outliers is needed

**Python example:**
```python
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# CRITICAL: SVM requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regression
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.2]
}

svr = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5,
                    scoring='r2', n_jobs=-1)
svr.fit(X_train_scaled, y_train)

# Classification
svc = GridSearchCV(SVC(kernel='rbf', probability=True),
                    param_grid={'C': [0.1, 1, 10, 100],
                               'gamma': ['scale', 'auto', 0.001, 0.01]},
                    cv=5, scoring='f1_weighted', n_jobs=-1)
svc.fit(X_train_scaled, y_train)
```

## 3. Random Forest (RF)

**Best for:** Complex patterns with moderate to large datasets

**Characteristics:**
- Handles non-linear relationships naturally
- Provides feature importance measures
- Robust to overfitting (with many trees)
- No need for feature scaling
- Works well with mixed data types

**When to use:**
- n > 50 samples
- Mixed features (spectral + metadata)
- Need feature importance ranking
- Robust baseline model for comparison
- Classification or regression

**Python example:**
```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Regression
rf_reg = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,  # Grow trees fully
    min_samples_split=5,  # Prevent overfitting on small data
    min_samples_leaf=2,
    max_features='sqrt',  # Decorrelate trees
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train, y_train)

# Feature importance
importances = rf_reg.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 20 most important features
plt.figure(figsize=(10, 6))
plt.bar(range(20), importances[indices[:20]])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Top 20 Feature Importances')
```

## 4. Gaussian Processes (GP)

**Best for:** Very small datasets with uncertainty quantification

**Characteristics:**
- Excellent for n < 50 samples
- Provides prediction uncertainty (confidence intervals)
- Flexible through kernel choice
- Computationally expensive (O(n^3))

**When to use:**
- Very small datasets (n = 10-50)
- Need uncertainty estimates for predictions
- Expensive experiments (optimization, DoE)
- Smooth, continuous functions expected

**Python example:**
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Define kernel (RBF + noise)
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + \
         WhiteKernel(noise_level=1e-5)

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42
)

gp.fit(X_train, y_train)

# Predict with uncertainty
y_pred, y_std = gp.predict(X_test, return_std=True)

# 95% confidence intervals
ci_lower = y_pred - 1.96 * y_std
ci_upper = y_pred + 1.96 * y_std

# Plot predictions with uncertainty
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Predictions')
plt.errorbar(y_test, y_pred, yerr=1.96*y_std, fmt='none',
             alpha=0.3, label='95% CI')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', label='Ideal')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
```

## 5. Neural Networks

**Best for:** Large datasets with complex, non-linear patterns

**Characteristics:**
- Powerful for n > 500 samples
- Can learn highly complex patterns
- Risk of overfitting on small data
- Requires careful regularization (dropout, early stopping)
- Less interpretable (black box)

**When to use:**
- Large spectral datasets (n > 500)
- Hyperspectral imaging
- Deep learning architectures (CNNs for images)
- When maximum predictive performance is priority
- Sufficient data for train/validation/test split

**Python example:**
```python
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# CRITICAL: Scale features for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simple MLP
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50, 20),  # 3 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.01,  # L2 regularization
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,  # Prevent overfitting
    validation_fraction=0.2,
    n_iter_no_change=20,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# Monitor training
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_, label='Training Loss')
plt.plot(mlp.validation_scores_, label='Validation Score')
plt.xlabel('Iteration')
plt.ylabel('Loss / Score')
plt.legend()
plt.title('Neural Network Training Progress')
```

## 6. k-Nearest Neighbors (k-NN)

**Best for:** Small datasets, local patterns, baseline model

**Characteristics:**
- Simple, interpretable
- No training phase (lazy learning)
- Works for both regression and classification
- Sensitive to feature scaling and noise
- Good baseline model

**When to use:**
- Quick baseline comparison
- Local similarity-based predictions
- Small to medium datasets
- Classification of similar samples
- When interpretability through neighbors is useful

**Python example:**
```python
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# CRITICAL: k-NN requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal k using cross-validation
k_range = range(1, min(21, X_train.shape[0] // 2))
cv_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_scaled, y_train,
                             cv=5, scoring='r2')
    cv_scores.append(scores.mean())

optimal_k = k_range[np.argmax(cv_scores)]

# Final model
knn = KNeighborsRegressor(n_neighbors=optimal_k, weights='distance')
knn.fit(X_train_scaled, y_train)
```

## Advanced Topics

### Ensemble Methods

Combine multiple models for improved performance:

```python
from sklearn.ensemble import VotingRegressor, StackingRegressor

# Voting (average predictions)
voting = VotingRegressor([
    ('pls', PLSRegression(n_components=10)),
    ('rf', RandomForestRegressor(n_estimators=500)),
    ('svr', SVR(kernel='rbf'))
])

# Stacking (meta-model learns to combine)
stacking = StackingRegressor(
    estimators=[
        ('pls', PLSRegression(n_components=10)),
        ('rf', RandomForestRegressor(n_estimators=500))
    ],
    final_estimator=Ridge(alpha=1.0)
)
```

### Feature Selection with Models

Use model-based feature selection:

```python
from sklearn.feature_selection import SelectFromModel

# Use Random Forest importances
rf = RandomForestRegressor(n_estimators=500)
selector = SelectFromModel(rf, threshold='median')
selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train final model on selected features
final_model = PLSRegression(n_components=5)
final_model.fit(X_train_selected, y_train)
```

### Transfer Learning

For small datasets, use models pretrained on larger datasets:

```python
# Example: Use pretrained model as feature extractor
# (Common in hyperspectral imaging, similar spectra types)

pretrained_model.fit(X_large_dataset, y_large_dataset)
features_train = pretrained_model.predict(X_small_train)

# Train simple model on extracted features
final_model = Ridge(alpha=1.0)
final_model.fit(features_train, y_small_train)
```
