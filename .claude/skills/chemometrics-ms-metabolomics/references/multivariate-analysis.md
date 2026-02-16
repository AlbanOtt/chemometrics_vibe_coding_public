# Multivariate Analysis for MS Metabolomics

## Principal Component Analysis (PCA)

Unsupervised method for exploratory analysis and outlier detection.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def perform_pca_analysis(
    X: np.ndarray,
    sample_labels: np.ndarray,
    n_components: int = 10
) -> dict:
    """Perform PCA with scores, loadings, and diagnostics."""
    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Loadings
    loadings = pca.components_.T

    # Hotelling's T2 for outlier detection
    t2_limit = n_components * (len(X) - 1) / (len(X) - n_components)
    t2_scores = np.sum((scores / np.std(scores, axis=0)) ** 2, axis=1)
    outliers = t2_scores > t2_limit * 2  # 2x limit as threshold

    return {
        'scores': scores,
        'loadings': loadings,
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        't2_scores': t2_scores,
        't2_limit': t2_limit,
        'outliers': outliers,
        'model': pca
    }

def plot_pca_scores(
    pca_results: dict,
    groups: np.ndarray,
    pc_x: int = 0,
    pc_y: int = 1
) -> None:
    """Plot PCA scores with group coloring."""
    scores = pca_results['scores']
    exp_var = pca_results['explained_variance']

    plt.figure(figsize=(10, 8))

    for group in np.unique(groups):
        mask = groups == group
        plt.scatter(
            scores[mask, pc_x],
            scores[mask, pc_y],
            label=group,
            alpha=0.7,
            s=80
        )

    plt.xlabel(f"PC{pc_x + 1} ({exp_var[pc_x]:.1%})")
    plt.ylabel(f"PC{pc_y + 1} ({exp_var[pc_y]:.1%})")
    plt.title("PCA Score Plot")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
```

## PLS-DA and OPLS-DA

Supervised methods for class discrimination and biomarker discovery.

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, permutation_test_score
import numpy as np

def perform_plsda(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    cv: int = 7
) -> dict:
    """Perform PLS-DA with validation metrics."""
    # Encode class labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # For binary classification, use single dummy variable
    if len(le.classes_) == 2:
        y_dummy = y_encoded
    else:
        # Multi-class: use one-hot encoding
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(sparse_output=False)
        y_dummy = ohe.fit_transform(y_encoded.reshape(-1, 1))

    # Fit PLS-DA
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y_dummy)

    # Cross-validation predictions
    y_pred_cv = cross_val_predict(pls, X, y_dummy, cv=cv)

    # VIP scores
    vip = calculate_vip(pls)

    # Permutation test for significance
    score, perm_scores, pvalue = permutation_test_score(
        pls, X, y_dummy, cv=cv, n_permutations=100,
        scoring='r2', random_state=42
    )

    return {
        'model': pls,
        'scores': pls.x_scores_,
        'loadings': pls.x_loadings_,
        'vip': vip,
        'r2': score,
        'q2': 1 - np.sum((y_dummy - y_pred_cv) ** 2) / np.sum((y_dummy - y_dummy.mean()) ** 2),
        'permutation_pvalue': pvalue,
        'label_encoder': le
    }

def calculate_vip(pls_model: PLSRegression) -> np.ndarray:
    """Calculate Variable Importance in Projection (VIP) scores."""
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_

    p, h = w.shape

    # Explained variance per component
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    # VIP calculation
    vip = np.sqrt(p * np.sum(s * (w ** 2), axis=1) / total_s)

    return vip
```

## S-Plot for Biomarker Selection

Combines correlation and covariance for feature selection.

```python
def create_splot(
    X: np.ndarray,
    y_binary: np.ndarray,
    pls_model: PLSRegression
) -> tuple[np.ndarray, np.ndarray]:
    """Create S-plot coordinates for biomarker selection."""
    # Scores for first component
    t1 = pls_model.x_scores_[:, 0]

    # Covariance (loading)
    p1 = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        p1[j] = np.cov(X[:, j], t1)[0, 1] / np.var(t1)

    # Correlation
    p_corr = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        p_corr[j] = np.corrcoef(X[:, j], t1)[0, 1]

    return p1, p_corr

def plot_splot(
    p_cov: np.ndarray,
    p_corr: np.ndarray,
    feature_names: list[str],
    threshold_cov: float = 0.05,
    threshold_corr: float = 0.8
) -> None:
    """Plot S-plot with significant features highlighted."""
    plt.figure(figsize=(10, 8))

    # Identify significant features
    significant = (np.abs(p_cov) > threshold_cov) & (np.abs(p_corr) > threshold_corr)

    plt.scatter(p_cov[~significant], p_corr[~significant], c='gray', alpha=0.3, s=30)
    plt.scatter(p_cov[significant], p_corr[significant], c='red', alpha=0.8, s=60)

    # Label significant features
    for i in np.where(significant)[0]:
        plt.annotate(feature_names[i], (p_cov[i], p_corr[i]), fontsize=8)

    plt.xlabel("p(cov) - Covariance")
    plt.ylabel("p(corr) - Correlation")
    plt.title("S-Plot for Biomarker Selection")
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
```

## Model Validation

```python
def validate_plsda(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    n_permutations: int = 1000,
    cv: int = 7
) -> dict:
    """Comprehensive PLS-DA validation with permutation testing."""
    from sklearn.model_selection import cross_val_score

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Original model performance
    pls = PLSRegression(n_components=n_components)

    # Q2 from cross-validation
    cv_scores = cross_val_score(pls, X, y_encoded, cv=cv, scoring='r2')
    q2 = cv_scores.mean()

    # Permutation test
    perm_q2 = []
    for i in range(n_permutations):
        y_perm = np.random.permutation(y_encoded)
        perm_scores = cross_val_score(pls, X, y_perm, cv=cv, scoring='r2')
        perm_q2.append(perm_scores.mean())

    # p-value: proportion of permuted Q2 >= observed Q2
    pvalue = (np.sum(np.array(perm_q2) >= q2) + 1) / (n_permutations + 1)

    # R2Y from training
    pls.fit(X, y_encoded)
    r2y = pls.score(X, y_encoded)

    return {
        'r2y': r2y,
        'q2': q2,
        'q2_std': cv_scores.std(),
        'permutation_pvalue': pvalue,
        'permutation_q2_null': np.array(perm_q2)
    }
```

## Overfitting in Multivariate Models

**Problem:** PLS-DA models perform well on training data but fail validation.

**Solutions:**
- Use appropriate cross-validation (7-fold CV for small datasets)
- Perform permutation testing (p < 0.05)
- Limit number of components to prevent overfitting
- Report both R2Y and Q2 (Q2 should be positive and close to R2Y)

```python
def check_overfitting(r2y: float, q2: float) -> str:
    """Check for overfitting in PLS-DA model."""
    diff = r2y - q2

    if q2 < 0:
        return "SEVERE OVERFITTING: Q2 < 0, model has no predictive ability"
    elif diff > 0.3:
        return f"WARNING: Large R2Y-Q2 difference ({diff:.2f}), possible overfitting"
    elif q2 < 0.5:
        return f"CAUTION: Low Q2 ({q2:.2f}), limited predictive ability"
    else:
        return f"OK: R2Y={r2y:.2f}, Q2={q2:.2f}, model appears valid"
```

## False Discovery in Biomarker Selection

**Problem:** Too many false positive biomarkers from multiple testing.

**Solutions:**
- Apply Benjamini-Hochberg FDR correction
- Use VIP > 1 as initial filter
- Validate with external cohort or targeted analysis
- Combine statistical (p-value) and biological (fold change) significance

```python
from scipy.stats import false_discovery_control

def select_biomarkers(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    vip_scores: np.ndarray,
    vip_threshold: float = 1.0,
    fdr_threshold: float = 0.05
) -> pd.DataFrame:
    """Select biomarkers using VIP and FDR-corrected p-values."""
    from scipy.stats import mannwhitneyu

    # VIP filter
    vip_mask = vip_scores > vip_threshold

    # Calculate p-values for VIP-selected features
    groups = np.unique(y)
    pvalues = []
    fold_changes = []

    for j in np.where(vip_mask)[0]:
        group0 = X[y == groups[0], j]
        group1 = X[y == groups[1], j]

        stat, pval = mannwhitneyu(group0, group1, alternative='two-sided')
        fc = np.median(group1) / np.median(group0)

        pvalues.append(pval)
        fold_changes.append(fc)

    # FDR correction
    pvalues = np.array(pvalues)
    fdr_adjusted = false_discovery_control(pvalues, method='bh')

    # Create results table
    results = pd.DataFrame({
        'feature': np.array(feature_names)[vip_mask],
        'vip': vip_scores[vip_mask],
        'fold_change': fold_changes,
        'pvalue': pvalues,
        'fdr': fdr_adjusted,
        'significant': fdr_adjusted < fdr_threshold
    })

    return results.sort_values('vip', ascending=False)
```
