---
name: chemometrics-shared
description: >-
  Shared chemometrics foundations: cross-validation strategies, performance metrics,
  overfitting prevention, sample-size guidance, and reporting standards. Used by all
  chemometrics skills. Load only the reference file you need.
license: MIT
metadata:
  skill-author: Alban Ott
---

# Chemometrics Shared Foundations

## When to Use What

Task: Choose cross-validation strategy
Use: [references/validation-strategies.md](references/validation-strategies.md)

Task: Evaluate regression or classification model
Use: [references/performance-metrics.md](references/performance-metrics.md)

Task: Determine if sample size is sufficient
Use: [references/sample-size-guidance.md](references/sample-size-guidance.md)

Task: Detect or prevent overfitting
Use: [references/overfitting-prevention.md](references/overfitting-prevention.md)

Task: Write methods section or prepare for publication
Use: [references/reporting-standards.md](references/reporting-standards.md)

Task: Follow chemometrics project workflow
Use: [references/workflow.md](references/workflow.md)

## Quick Reference: CV Decision Tree

```
What is your sample size?

+-- n < 20: LOOCV (high variance — consider repeated random splits)
+-- 20 <= n < 50: LOOCV or 5-Fold CV (repeat 3-10x)
+-- 50 <= n < 200: 5-Fold or 10-Fold CV (repeat 3-10x)
+-- n >= 200: 10-Fold CV or Hold-Out (70/30 or 80/20)

Special cases:
  Time series      -> TimeSeriesSplit (no future leakage)
  Batches/groups   -> GroupKFold (keep groups together)
  Imbalanced       -> StratifiedKFold (preserve class ratios)
  Spatial data     -> Spatial CV (geographic splits)
```

## Quick Reference: Metrics

**Regression:** RMSEP (primary), R-squared, RPD, Bias, SEP
**Classification:** Sensitivity, Specificity, F1-score (primary), Accuracy, ROC AUC

### RPD Interpretation (Saeys et al. 2005)

| RPD | Quality |
|-----|---------|
| > 2.5 | Excellent quantitative |
| 2.0-2.5 | Good quantitative |
| 1.8-2.0 | Fair (screening) |
| 1.4-1.8 | Very rough screening |
| < 1.4 | Unreliable |

### R-squared Interpretation

| R-squared | Quality |
|-----------|---------|
| > 0.9 | Excellent |
| > 0.8 | Good |
| > 0.7 | Acceptable |
| < 0.7 | Poor (most applications) |

## See Also

- ML method selection: `../chemometrics-ml-selection/SKILL.md`
- MS metabolomics: `../chemometrics-ms-metabolomics/SKILL.md`
- Hybrid modeling: `../chemometrics-hybrid-modeling/SKILL.md`
- Model validation: `../chemometrics-validation/SKILL.md`
