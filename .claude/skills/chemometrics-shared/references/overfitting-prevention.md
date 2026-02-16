# Overfitting Prevention

## 1. Detecting Overfitting

Compare training and validation performance:

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y, cv=10, scoring='r2', return_train_score=True
)

train_r2 = results['train_score'].mean()
test_r2 = results['test_score'].mean()

print(f"Training R²:   {train_r2:.3f}")
print(f"Validation R²: {test_r2:.3f}")
print(f"Difference:    {train_r2 - test_r2:.3f}")

if train_r2 - test_r2 > 0.1:
    print("Warning: Possible overfitting!")
```

**Red flags:**
- Train-test gap > 0.1 (R²)
- Perfect training score with poor validation
- Performance degrades with more model complexity

## 2. Learning Curves

Visualize performance vs. training set size:

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2', n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training', marker='o')
plt.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation', marker='s')
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('R²')
plt.title('Learning Curve')
plt.legend()
plt.grid(alpha=0.3)
```

**Diagnosis:**
- High variance (overfitting): Large train-test gap, more data helps
- High bias (underfitting): Both scores low, need more complex model

## 3. Validation Curves

Visualize performance vs. hyperparameter:

```python
from sklearn.model_selection import validation_curve
from sklearn.cross_decomposition import PLSRegression
import numpy as np

param_range = [1, 2, 3, 5, 7, 10, 15, 20]
train_scores, val_scores = validation_curve(
    PLSRegression(), X, y,
    param_name='n_components', param_range=param_range,
    cv=5, scoring='r2', n_jobs=-1
)

optimal_n = param_range[np.argmax(val_scores.mean(axis=1))]
print(f"Optimal n_components: {optimal_n}")
```

## 4. Prevention Strategies

| Strategy | When to Use |
|----------|-------------|
| **Regularization** (alpha, L1/L2) | All models, especially linear |
| **Reduce complexity** (fewer components/depth) | When train >> test performance |
| **Early stopping** | Neural networks, gradient boosting |
| **Cross-validation** for model selection | Always |
| **Permutation testing** | PLS-DA, classification models |
| **Feature selection** | When p >> n |
| **More data** | When learning curve shows high variance |
