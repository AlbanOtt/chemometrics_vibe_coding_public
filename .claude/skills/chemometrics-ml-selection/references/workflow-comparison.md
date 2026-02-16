# Workflow and Common Pitfalls

Standard workflow recommendations and common pitfalls for ML in chemometrics.

## Standard Approach for New Chemometrics Problems

### 1. Start Simple

```python
# Baseline 1: PLS (industry standard)
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=10)

# Baseline 2: Linear regression with regularization
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
```

### 2. Try Non-linear (if baseline is insufficient)

```python
# For small data
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=10, gamma='scale')

# For larger data
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500)
```

### 3. Compare Systematically

```python
from sklearn.model_selection import cross_val_score

models = {
    'PLS': PLSRegression(n_components=10),
    'Ridge': Ridge(alpha=1.0),
    'SVR': SVR(kernel='rbf'),
    'Random Forest': RandomForestRegressor(n_estimators=500)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='r2')
    print(f"{name}: R2 = {scores.mean():.3f} +/- {scores.std():.3f}")
```

### 4. Tune Best Performer

```python
from sklearn.model_selection import GridSearchCV

# Tune hyperparameters of best model
param_grid = {...}  # Define based on chosen model
grid_search = GridSearchCV(best_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 5. Validate Rigorously

```python
# Final evaluation on held-out test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Report multiple metrics
from sklearn.metrics import r2_score, mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
rpd = y_test.std() / rmse
```

## Common Pitfalls

### 1. Using Complex Models on Small Data

Wrong:
```python
# Only 30 samples, using deep neural network
model = MLPRegressor(hidden_layer_sizes=(200, 100, 50))
model.fit(X_train, y_train)  # Will overfit!
```

Correct:
```python
# Use appropriate method for small data
model = PLSRegression(n_components=5)  # Or SVM, GP
model.fit(X_train, y_train)
```

### 2. Forgetting to Scale Features

Wrong:
```python
# SVM without scaling (different units, scales)
svm = SVR(kernel='rbf')
svm.fit(X_raw, y)  # Will perform poorly!
```

Correct:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
svm = SVR(kernel='rbf')
svm.fit(X_scaled, y)
```

### 3. Ignoring Cross-Validation

Wrong:
```python
# Evaluating on training data
model.fit(X, y)
y_pred = model.predict(X)
print(f"R2 = {r2_score(y, y_pred)}")  # Overly optimistic!
```

Correct:
```python
# Use cross-validation or test set
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"R2 = {scores.mean():.3f} +/- {scores.std():.3f}")
```

### 4. Using Default Hyperparameters

Wrong:
```python
# Defaults may not be optimal for your data
rf = RandomForestRegressor()  # Only 100 trees by default
rf.fit(X_train, y_train)
```

Correct:
```python
# Tune hyperparameters
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```
