# Hybrid Modeling Approaches

## 1. Residual Modeling (Serial Hybrid)

**Concept:** Use mechanistic model first, then ML corrects residuals

```
y_pred = y_mechanistic + ML(x, residual)
```

**When to use:**
- Have decent mechanistic model but it's not perfect
- Deviations from theory are systematic
- Want to understand where physics breaks down

**Example: Beer-Lambert Law Deviations**

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def beer_lambert_model(spectra, wavelengths, extinction_coeff):
    """Mechanistic model: A = e * c * l"""
    # Simplified: assuming path length = 1 cm
    # Sum absorbances at key wavelengths
    concentrations = spectra[:, wavelengths] @ extinction_coeff
    return concentrations

# Step 1: Apply mechanistic model
y_mechanistic = beer_lambert_model(X_spectra, key_wavelengths, epsilon)

# Step 2: Calculate residuals
residuals_train = y_train - y_mechanistic_train

# Step 3: Train ML model to predict residuals
residual_model = RandomForestRegressor(n_estimators=500, random_state=42)
residual_model.fit(X_train, residuals_train)

# Step 4: Final prediction = mechanistic + ML correction
y_pred_test = y_mechanistic_test + residual_model.predict(X_test)

# Interpretation: Where does physics fail?
important_features = residual_model.feature_importances_
# High importance -> mechanistic model inadequate at these wavelengths
```

**Advantages:**
- Mechanistic part is always physically sound
- ML only models unexplained variance
- Requires less training data than pure ML
- Easy to interpret (see where physics breaks down)

## 2. Parallel Hybrid (Ensemble)

**Concept:** Combine predictions from mechanistic and ML models

```
y_pred = w1 * y_mechanistic + w2 * y_ML
```

**When to use:**
- Both mechanistic and ML models have merits
- Want to balance physics constraints with data fit
- Uncertain about model form

**Implementation:**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Mechanistic model predictions
y_mech_train = mechanistic_model(X_train, params)
y_mech_test = mechanistic_model(X_test, params)

# Data-driven model
ml_model = RandomForestRegressor(n_estimators=500, random_state=42)
ml_model.fit(X_train, y_train)
y_ml_train = ml_model.predict(X_train)
y_ml_test = ml_model.predict(X_test)

# Learn optimal weights
from scipy.optimize import minimize

def ensemble_loss(weights):
    w1, w2 = weights
    y_ensemble = w1 * y_mech_train + w2 * y_ml_train
    return np.mean((y_ensemble - y_train) ** 2)

# Constrain: w1 + w2 = 1, both >= 0
result = minimize(
    ensemble_loss,
    x0=[0.5, 0.5],
    bounds=[(0, 1), (0, 1)],
    constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
)

w1_opt, w2_opt = result.x
print(f"Optimal weights: mechanistic={w1_opt:.2f}, ML={w2_opt:.2f}")

# Final predictions
y_pred_test = w1_opt * y_mech_test + w2_opt * y_ml_test
```

**Alternatively: Stack with meta-model**

```python
from sklearn.ensemble import StackingRegressor

# Create feature matrix from both predictions
X_meta_train = np.column_stack([y_mech_train, y_ml_train])
X_meta_test = np.column_stack([y_mech_test, y_ml_test])

# Meta-model learns to combine (can learn non-linear combinations)
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta_train, y_train)
y_pred_test = meta_model.predict(X_meta_test)

print(f"Meta-model weights: {meta_model.coef_}")
```

## 3. Physics-Informed Neural Networks (PINNs)

**Concept:** Incorporate physical laws as constraints in ML training

**When to use:**
- Have differential equations governing the system
- Want ML to respect physical laws exactly
- Dealing with PDE-based models (diffusion, flow, heat transfer)

**Example: Constrained Neural Network**

```python
import torch
import torch.nn as nn

class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, x, y_pred):
        """Custom physics-based loss term"""
        # Example: Concentration must be non-negative
        negative_penalty = torch.relu(-y_pred).sum()

        # Example: Mass balance constraint
        # sum of concentrations = total_concentration
        mass_balance_error = (y_pred.sum(dim=1) - total_concentration).pow(2).mean()

        return negative_penalty + mass_balance_error

def train_with_physics(model, X_train, y_train, epochs=1000, lambda_physics=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Data loss
        y_pred = model(X_train)
        data_loss = criterion(y_pred, y_train)

        # Physics loss
        physics_loss = model.physics_loss(X_train, y_pred)

        # Combined loss
        total_loss = data_loss + lambda_physics * physics_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Data Loss = {data_loss:.4f}, "
                  f"Physics Loss = {physics_loss:.4f}")

    return model
```

**Physics constraints to consider:**
- **Non-negativity:** Concentrations, absorbances >= 0
- **Mass balance:** sum(mass_in) = sum(mass_out)
- **Energy conservation:** sum(energy_in) = sum(energy_out)
- **Thermodynamic limits:** 0 <= conversion <= 100%
- **Monotonicity:** Some relationships must be monotonic
- **Symmetry:** Molecular properties may have symmetry

## 4. Mechanistic Features for ML

**Concept:** Engineer features from mechanistic models as inputs to ML

**When to use:**
- Have partial mechanistic understanding
- Want ML to build on physics-based features
- Pure spectral features don't capture physics

**Example: Chemical Reaction Engineering**

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def calculate_mechanistic_features(temperature, pressure, flow_rate,
                                     feed_composition):
    """Calculate physics-based features"""
    # Arrhenius-based features
    activation_energy = 50000  # J/mol (example)
    R = 8.314  # Gas constant
    k_rate = np.exp(-activation_energy / (R * temperature))

    # Dimensionless numbers
    reynolds = (flow_rate * pipe_diameter) / viscosity
    peclet = reynolds * schmidt_number

    # Residence time
    tau = reactor_volume / flow_rate

    # Damkohler number (reaction vs flow)
    damkohler = k_rate * tau

    return np.column_stack([
        temperature,
        pressure,
        k_rate,
        reynolds,
        peclet,
        damkohler,
        feed_composition
    ])

# Create features
X_mech_train = calculate_mechanistic_features(T_train, P_train, Q_train, C_train)
X_mech_test = calculate_mechanistic_features(T_test, P_test, Q_test, C_test)

# ML learns non-linear combinations of physics-based features
model = GradientBoostingRegressor(n_estimators=500, max_depth=5)
model.fit(X_mech_train, y_train)
y_pred_test = model.predict(X_mech_test)

# Interpret: which dimensionless numbers matter most?
feature_names = ['T', 'P', 'k_rate', 'Re', 'Pe', 'Da', 'C_feed']
importances = model.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.3f}")
```

## 5. Constrained Optimization with ML

**Concept:** Use ML predictions subject to physical constraints

**When to use:**
- ML predictions violate physics
- Need to ensure predictions are physically feasible
- Have inequality constraints

**Example: Spectral Unmixing with Non-Negativity**

```python
from sklearn.decomposition import NMF  # Non-negative Matrix Factorization
from scipy.optimize import nnls  # Non-negative Least Squares

# Standard unconstrained approach (may give negative concentrations)
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=5)
pls.fit(X_spectra_train, y_conc_train)
y_pred_unconstrained = pls.predict(X_spectra_test)

# Some predictions may be negative! (physically impossible)
print(f"Negative predictions: {(y_pred_unconstrained < 0).sum()}")

# Constrained approach using NMF
nmf = NMF(n_components=5, init='random', random_state=42)
W_train = nmf.fit_transform(X_spectra_train)
H = nmf.components_

# Learn mapping from W to concentrations (non-negative)
from sklearn.linear_model import Ridge
regressor = Ridge(alpha=1.0, positive=True)  # Force positive coefficients
regressor.fit(W_train, y_conc_train)

# Predict (guaranteed non-negative)
W_test = nmf.transform(X_spectra_test)
y_pred_constrained = regressor.predict(W_test)

print(f"All predictions >= 0: {(y_pred_constrained >= 0).all()}")
```

**Post-processing constraints:**

```python
from scipy.optimize import minimize

def constrained_prediction(ml_model, X_test, constraints):
    """Post-process ML predictions to satisfy constraints"""
    y_pred_initial = ml_model.predict(X_test)

    def objective(y):
        # Minimize deviation from ML prediction
        return np.sum((y - y_pred_initial) ** 2)

    # Constraints example:
    # 1. All values >= 0
    # 2. Sum = 1 (for fractions)
    bounds = [(0, None) for _ in range(len(y_pred_initial))]
    cons = {'type': 'eq', 'fun': lambda y: y.sum() - 1.0}

    result = minimize(objective, y_pred_initial,
                     bounds=bounds, constraints=cons)

    return result.x

# Apply to each test sample
y_pred_constrained = np.array([
    constrained_prediction(model, x.reshape(1, -1), constraints)
    for x in X_test
])
```

---

## Best Practices

### 1. Start Simple

```python
# Always compare:
# 1. Pure mechanistic
# 2. Pure ML
# 3. Hybrid

# Choose hybrid only if it improves over both
```

### 2. Validate Physics Constraints

```python
def check_physical_constraints(y_pred, y_true):
    """Verify predictions satisfy physics"""
    print("Constraint Checks:")

    # Non-negativity
    n_negative = (y_pred < 0).sum()
    print(f"  Negative values: {n_negative} / {len(y_pred)}")

    # Range constraints
    if hasattr(y_true, 'min') and hasattr(y_true, 'max'):
        y_min, y_max = y_true.min(), y_true.max()
        n_out_of_range = ((y_pred < y_min) | (y_pred > y_max)).sum()
        print(f"  Out of training range: {n_out_of_range} / {len(y_pred)}")

    # Mass balance (if applicable)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        mass_balance_error = np.abs(y_pred.sum(axis=1) - 100)
        print(f"  Mass balance error: {mass_balance_error.mean():.2f}%")

    return {
        'n_negative': n_negative,
        'n_out_of_range': n_out_of_range if 'n_out_of_range' in locals() else None
    }

check_physical_constraints(y_pred_hybrid, y_test)
```

### 3. Interpret Hybrid Models

```python
# For residual models: where does physics fail?
if isinstance(model, HybridResidualModel):
    residual_model = model.ml_component
    importances = residual_model.feature_importances_

    print("Features where mechanistic model is inadequate:")
    for i, imp in enumerate(importances):
        if imp > np.percentile(importances, 90):
            print(f"  Feature {i}: importance = {imp:.3f}")

# For parallel models: how much to trust physics vs data?
if isinstance(model, HybridParallelModel):
    print(f"Mechanistic weight: {model.w_mechanistic:.2f}")
    print(f"ML weight: {model.w_ml:.2f}")
```

### 4. Extrapolation Testing

Hybrid models should extrapolate better than pure ML:

```python
# Create extrapolation test set (outside training range)
X_extrap = create_extrapolation_data()  # Beyond training ranges

y_pred_ml = pure_ml_model.predict(X_extrap)
y_pred_mech = mechanistic_model(X_extrap)
y_pred_hybrid = hybrid_model.predict(X_extrap)

# Hybrid should be closer to mechanistic when extrapolating
# (mechanistic is more trustworthy outside data range)
```

---

## Common Pitfalls

### 1. Ignoring Model Mismatch

**Wrong:**
```python
# Using mechanistic model that's completely wrong
y_pred = wrong_mechanism(X) + ml_model.predict(X)
# Garbage in, garbage out!
```

**Correct:**
```python
# Validate mechanistic component first
r2_mech = score(y_true, mechanistic_model(X))
if r2_mech < 0:  # Worse than mean!
    print("WARNING: Mechanistic model is very poor, reconsider hybrid approach")
```

### 2. Over-constraining

**Wrong:**
```python
# Too many physics constraints may prevent learning
loss = data_loss + 1000 * physics_loss  # Overwhelms data
# Model will satisfy physics perfectly but ignore data!
```

**Correct:**
```python
# Balance data and physics losses
lambda_physics = 0.1  # Tune this hyperparameter
loss = data_loss + lambda_physics * physics_loss
```

### 3. Not Validating Constraints

**Wrong:**
```python
# Assume constraints are satisfied
y_pred_hybrid = model.predict(X_test)
# May still violate physics!
```

**Correct:**
```python
# Always verify
y_pred_hybrid = model.predict(X_test)
assert (y_pred_hybrid >= 0).all(), "Negative predictions!"
assert np.allclose(y_pred_hybrid.sum(axis=1), 100), "Mass balance violated!"
```

---

## Advanced Topics

### Transfer Learning with Mechanistic Models

Use mechanistic simulations to pre-train, then fine-tune on real data:

```python
# Step 1: Generate synthetic data from mechanistic model
X_synthetic, y_synthetic = mechanistic_simulator(n_samples=10000)

# Step 2: Pre-train ML on synthetic data
model.fit(X_synthetic, y_synthetic)

# Step 3: Fine-tune on small real dataset
model.fit(X_real_train, y_real_train)  # Warm start from synthetic
```

### Multi-Fidelity Modeling

Combine high-fidelity (expensive) and low-fidelity (cheap) models:

```python
# Low-fidelity: fast approximate mechanistic model
y_low = fast_approximation(X)

# High-fidelity: expensive accurate model (few samples)
y_high = expensive_model(X_subset)

# ML learns correction: y_high ~ y_low + ML(X, y_low)
correction_model.fit(X_subset, y_high - y_low[subset_indices])

# Predict on new data
y_pred = fast_approximation(X_new) + correction_model.predict(X_new)
```
