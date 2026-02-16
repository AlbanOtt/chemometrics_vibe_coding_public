---
name: chemometrics-ml-selection
description: Expert guidance for selecting appropriate machine learning methods for chemometrics applications including spectroscopy, chromatography, and process analytics. Provides decision frameworks based on data characteristics, problem type, and domain best practices.
license: MIT
metadata:
  skill-author: Alban Ott
  based-on: Trinh et al. 2021 - Machine Learning in Chemical Product Engineering
---

# Chemometrics ML Method Selection

## Overview

Expert guidance for selecting ML methods for chemometrics applications. Covers algorithm choice based on data characteristics, problem type, and domain requirements. Chemometrics data is typically high-dimensional, small-sample, multicollinear, and interpretability-sensitive.

## When to Use This Skill

Use when starting a new chemometrics analysis, working with spectroscopic or chromatographic data, building calibration/classification models, dealing with small datasets (n < 100), or comparing ML approaches systematically.

## Decision Tree for Method Selection

```
START: What is your problem type?
|
+- REGRESSION (predict continuous values: concentration, properties)
|  |
|  +- Linear relationship expected?
|  |  +- YES -> Start with PLS or PCR
|  |  |         * PLS: When X and y should both be modeled
|  |  |         * PCR: When only X structure matters
|  |  |         * Consider: Ridge/Lasso for feature selection
|  |  |
|  |  +- NO -> Try non-linear methods
|  |            * Small data (n<100): Gaussian Processes (GP), SVM with RBF
|  |            * Large data (n>1000): Neural Networks, Random Forest
|  |            * Need interpretability: Tree-based (RF, Gradient Boosting)
|  |
|  +- How many samples do you have?
|     +- n < 50: GP, SVM, k-NN, or stay with PLS
|     +- 50 < n < 500: SVM, RF, GP, Neural Networks (small)
|     +- n > 500: Neural Networks, Deep Learning, Gradient Boosting
|
+- CLASSIFICATION (predict categories: pass/fail, species, authenticity)
|  |
|  +- How many samples do you have?
|  |  +- n < 50: PLS-DA, SVM, k-NN
|  |  +- 50 < n < 500: SVM, Random Forest, PLS-DA
|  |  +- n > 500: Neural Networks, Gradient Boosting
|  |
|  +- Need probabilistic outputs?
|     +- YES -> Logistic Regression, SVM with probability, RF
|     +- NO -> SVM, k-NN, PLS-DA
|
+- CLUSTERING (find natural groups: sample similarity, outlier detection)
|  |
|  +- Know number of clusters?
|  |  +- YES -> K-Means, GMM
|  |  +- NO -> Hierarchical Clustering, DBSCAN
|  |
|  +- Need soft assignments (probabilities)?
|     +- YES -> Gaussian Mixture Models (GMM)
|     +- NO -> K-Means, Hierarchical
|
+- DIMENSIONALITY REDUCTION (visualization, exploratory analysis)
   |
   +- Preserve variance -> PCA
   +- Preserve distances -> MDS, Isomap
   +- Visualization (2D/3D) -> t-SNE, UMAP
   +- With class information -> LDA, PLS-DA scores
```

## Core Methods (Summary)

**PLS**: Linear, high-dim, n=20-200. Handles multicollinearity. VIP for interpretation.
Details: [references/method-details.md](references/method-details.md)

**SVM/SVR**: Non-linear via kernels, n<100. Robust to outliers. Requires scaling.
Details: [references/method-details.md](references/method-details.md)

**Random Forest**: Non-linear, n>50. Feature importance. No scaling needed.
Details: [references/method-details.md](references/method-details.md)

**Gaussian Processes**: Very small data (n<50). Uncertainty quantification. O(n^3).
Details: [references/method-details.md](references/method-details.md)

**Neural Networks**: Large data (n>500). Complex patterns. Needs regularization.
Details: [references/method-details.md](references/method-details.md)

**k-NN**: Baseline model. Local patterns. Lazy learning. Requires scaling.
Details: [references/method-details.md](references/method-details.md)

## Method Comparison Table

| Method | Best for | Data Size | Interpretability | Handles Multicollinearity | Requires Scaling |
|--------|----------|-----------|------------------|---------------------------|------------------|
| **PLS** | Linear, high-dim | Small-Med (20-200) | High | Yes | Optional |
| **PCR** | Linear, dim. reduction | Small-Med | High | Yes | Optional |
| **SVM** | Non-linear, small data | Small (<100) | Low | Moderate | Required |
| **Random Forest** | Complex patterns | Med-Large (>50) | Medium | Yes | No |
| **Gradient Boosting** | Maximum performance | Med-Large (>100) | Medium | Yes | No |
| **GP** | Very small, uncertainty | Very Small (<50) | Medium | Moderate | Required |
| **Neural Networks** | Complex, large data | Large (>500) | Low | Moderate | Required |
| **k-NN** | Baseline, local | Small-Med | High | No | Required |

## Decision Heuristics

### By Data Characteristics

- **High-dimensional (p >> n):** PLS, PCR (dimensionality reduction first)
- **Multicollinear features:** PLS, Random Forest, Ridge Regression
- **Non-linear relationships:** SVM with RBF kernel, Random Forest, Neural Networks
- **Mixed data types:** Random Forest, Gradient Boosting
- **Need uncertainty estimates:** Gaussian Processes, Bayesian approaches

### By Problem Requirements

- **Maximum interpretability:** PLS, Linear Regression, Decision Trees
- **Maximum performance:** Gradient Boosting, Neural Networks (if enough data)
- **Robust to outliers:** SVM (soft margin), Random Forest
- **Fast prediction:** k-NN, Linear models (avoid GP for large test sets)
- **Probabilistic outputs:** Logistic Regression, SVM with probability, Gaussian Processes

### By Sample Size

See [../chemometrics-shared/references/sample-size-guidance.md](../chemometrics-shared/references/sample-size-guidance.md) for detailed guidance.

## Workflow and Common Pitfalls

Standard workflow (Start Simple -> Non-linear -> Compare -> Tune -> Validate) and common pitfalls (complex models on small data, forgetting scaling, ignoring CV, default hyperparameters).
Details: [references/workflow-comparison.md](references/workflow-comparison.md)

## Advanced Topics

Ensemble methods, feature selection with models, and transfer learning for small datasets.
Details: [references/method-details.md](references/method-details.md)

## See Also

- Validation strategies: [../chemometrics-shared/references/validation-strategies.md](../chemometrics-shared/references/validation-strategies.md)
- Performance metrics: [../chemometrics-shared/references/performance-metrics.md](../chemometrics-shared/references/performance-metrics.md)
- Sample size guidance: [../chemometrics-shared/references/sample-size-guidance.md](../chemometrics-shared/references/sample-size-guidance.md)
- MS metabolomics: [../chemometrics-ms-metabolomics/SKILL.md](../chemometrics-ms-metabolomics/SKILL.md)

## References

This skill is based on:

- **Trinh et al. (2021).** "Machine Learning in Chemical Product Engineering: The State of the Art and a Guide for Newcomers." *Processes*, 9(8), 1456. [doi:10.3390/pr9081456](https://doi.org/10.3390/pr9081456)

Additional recommended reading:

- Brereton, R. G. (2015). *Chemometrics for Pattern Recognition.* Wiley.
- Wold, S., Sjostrom, M., & Eriksson, L. (2001). "PLS-regression: a basic tool of chemometrics." *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109-130.
- Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt Publishing.
