# Sample Size Guidance

## Decision Tree by Sample Size

```
n < 30:   PLS, GP, k-NN, SVM (careful of overfitting!)
          CV: LOOCV or repeated random splits
          Avoid: Neural networks, deep learning

30-100:   PLS, SVM, Random Forest (limit depth), GP
          CV: 5-Fold or 10-Fold, repeat 3-10x
          Consider: Nested CV for hyperparameter tuning

100-500:  PLS, SVM, Random Forest, Gradient Boosting
          CV: 10-Fold or hold-out (80/20)
          Can afford: Full hyperparameter grid search

n > 500:  Neural Networks, Gradient Boosting, Random Forest
          CV: 10-Fold or hold-out (70/30)
          Can afford: Train/validation/test three-way split
```

## Rules of Thumb

- **Minimum n per class (classification):** 10-20 samples per class for simple models
- **PLS components:** Never exceed n/3 components (risk of overfitting)
- **Features vs samples:** When p >> n, use PLS, PCR, or regularized methods
- **Cross-validation:** n < 50 favors LOOCV; n >= 50 favors k-fold
- **Test set:** Only meaningful if n > 100 (otherwise all data needed for CV)

## When Data Is Insufficient

1. **Repeated CV** — increases estimate stability without needing more data
2. **Bootstrap validation** — resampling-based alternative
3. **Transfer learning** — leverage models from larger related datasets
4. **Data augmentation** — spectral noise injection, mixtures of existing samples
5. **Collect more data** — often the best solution
