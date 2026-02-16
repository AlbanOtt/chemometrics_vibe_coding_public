# Hybrid Modeling Application Examples

## Example 1: NIR Spectroscopy with Beer-Lambert Deviations

```python
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
X_spectra = load_spectra()  # n_samples x n_wavelengths
y_concentration = load_reference()

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_spectra, y_concentration, test_size=0.2, random_state=42
)

### Approach 1: Pure ML (PLS)
pls = PLSRegression(n_components=10)
pls.fit(X_train, y_train)
y_pred_pls = pls.predict(X_test).ravel()

print("Pure ML (PLS):")
print(f"  R2 = {r2_score(y_test, y_pred_pls):.3f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_pls)):.3f}")

### Approach 2: Mechanistic (Beer-Lambert at key wavelengths)
key_wavelengths = [1450, 1940, 2100]  # Water absorption bands
extinction_coeffs = np.array([0.5, 0.8, 0.3])  # Example values

def beer_lambert(spectra, wavelengths, epsilon):
    # A = e * c * l (assuming l = 1 cm)
    absorbances = spectra[:, wavelengths]
    concentration = absorbances @ epsilon
    return concentration

y_pred_mech = beer_lambert(X_test, key_wavelengths, extinction_coeffs)

print("\nPure Mechanistic:")
print(f"  R2 = {r2_score(y_test, y_pred_mech):.3f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_mech)):.3f}")

### Approach 3: Hybrid (Residual Modeling)
# Apply mechanistic model
y_mech_train = beer_lambert(X_train, key_wavelengths, extinction_coeffs)
y_mech_test = beer_lambert(X_test, key_wavelengths, extinction_coeffs)

# Calculate residuals
residuals_train = y_train - y_mech_train

# Train ML on residuals
residual_model = RandomForestRegressor(n_estimators=500, random_state=42)
residual_model.fit(X_train, residuals_train)

# Hybrid prediction
residuals_pred = residual_model.predict(X_test)
y_pred_hybrid = y_mech_test + residuals_pred

print("\nHybrid (Mechanistic + ML Residual):")
print(f"  R2 = {r2_score(y_test, y_pred_hybrid):.3f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_hybrid)):.3f}")

# Analyze where mechanistic model fails
residual_importance = residual_model.feature_importances_
top_wavelengths = np.argsort(residual_importance)[-10:]
print(f"\nTop wavelengths for residual correction: {top_wavelengths}")
```

## Example 2: Chemical Reactor with Kinetic Model

```python
import numpy as np
from sklearn.neural_network import MLPRegressor

# Mechanistic model: Arrhenius kinetics
def arrhenius_model(T, C_A, k0=1e10, Ea=50000):
    """First-order reaction: r = k * C_A"""
    R = 8.314  # J/(mol*K)
    k = k0 * np.exp(-Ea / (R * T))
    conversion = 1 - np.exp(-k * residence_time)
    return conversion * C_A

# Generate training data (real experiments)
T_train, C_A_train, conversion_train = load_experimental_data()

# Mechanistic predictions
conversion_mech_train = arrhenius_model(T_train, C_A_train)
conversion_mech_test = arrhenius_model(T_test, C_A_test)

# Hybrid approach: NN corrects mechanistic predictions
# Input: [T, C_A, conversion_mech]
X_hybrid_train = np.column_stack([T_train, C_A_train, conversion_mech_train])
X_hybrid_test = np.column_stack([T_test, C_A_test, conversion_mech_test])

# Neural network learns corrections
nn = MLPRegressor(
    hidden_layer_sizes=(50, 25),
    activation='relu',
    alpha=0.01,  # Regularization
    early_stopping=True,
    random_state=42
)

nn.fit(X_hybrid_train, conversion_train)
conversion_pred_hybrid = nn.predict(X_hybrid_test)

# Ensure physical constraints (0 <= conversion <= 1)
conversion_pred_hybrid = np.clip(conversion_pred_hybrid, 0, 1)

print(f"Hybrid model R2 = {r2_score(conversion_test, conversion_pred_hybrid):.3f}")
```

## Example 3: Spectral Unmixing with Mass Balance

```python
from sklearn.decomposition import NMF
import numpy as np

# Problem: Predict concentration of 3 components from mixture spectra
# Constraint: c1 + c2 + c3 = 100% (mass balance)

# Standard unconstrained PLS
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=5)
pls.fit(X_train, y_train)  # y_train is (n_samples, 3) concentrations
y_pred_pls = pls.predict(X_test)

# Check constraint violation
mass_balance_error_pls = np.abs(y_pred_pls.sum(axis=1) - 100)
print(f"PLS mass balance error: {mass_balance_error_pls.mean():.2f}%")

# Hybrid approach: Enforce mass balance
# Step 1: Predict with PLS
y_pred_unconstrained = pls.predict(X_test)

# Step 2: Normalize to satisfy constraint
y_pred_hybrid = y_pred_unconstrained / y_pred_unconstrained.sum(axis=1, keepdims=True) * 100

# Also enforce non-negativity
y_pred_hybrid = np.maximum(y_pred_hybrid, 0)
y_pred_hybrid = y_pred_hybrid / y_pred_hybrid.sum(axis=1, keepdims=True) * 100

# Check constraint satisfaction
mass_balance_error_hybrid = np.abs(y_pred_hybrid.sum(axis=1) - 100)
print(f"Hybrid mass balance error: {mass_balance_error_hybrid.mean():.2f}%")
# Should be ~0

# Compare predictive performance
from sklearn.metrics import mean_squared_error
rmse_pls = np.sqrt(mean_squared_error(y_test, y_pred_pls))
rmse_hybrid = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))
print(f"\nRMSE - PLS: {rmse_pls:.3f}, Hybrid: {rmse_hybrid:.3f}")
```
