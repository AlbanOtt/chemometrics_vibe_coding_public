---
name: chemometrics-ms-metabolomics
description: >-
  Expert guidance for processing and analyzing mass spectrometry (MS) based metabolomics data.
  Covers the complete workflow from sample normalization through data processing, multivariate
  analysis, and metabolite identification. Use when processing LC-MS or GC-MS metabolomics raw
  data, performing peak detection, alignment, and normalization, applying multivariate analysis
  (PCA, PLS-DA, OPLS-DA), identifying metabolites from MS/MS spectra, or designing clinical
  metabolomics studies.
license: MIT
author: Alban Ott
based-on: Boccard & Rudaz 2018 - Extracting Knowledge from MS Clinical Metabolomic Data
---

# MS Metabolomics Data Processing

Complete workflow from raw MS data to metabolite identification. Load only the reference file relevant to your current task.

## Workflow Decision Tree

```
START: What stage are you at?

├─ PRE-ACQUISITION: Sample normalization needed?
│  ├─ Cell cultures → Normalize by cell count or protein content
│  ├─ Urine → Dilute to constant osmolality or creatinine
│  ├─ Blood/Plasma → Use fixed volume (no normalization needed)
│  └─ Tissue → Normalize by wet weight or protein content

├─ DATA ACQUISITION: QC strategy?
│  ├─ Insert pooled QC samples every 5-10 injections
│  ├─ Randomize sample order within batches
│  └─ Include blanks and internal standards

├─ SIGNAL PROCESSING: Raw data to feature table?
│  ├─ Peak detection → XCMS, MZmine, MS-DIAL
│  ├─ Alignment → RT warping + m/z matching
│  └─ Gap filling → Re-integration at missing features

├─ NORMALIZATION: Which method?
│  ├─ Simple → TSN, MSTUS
│  ├─ Reference-based → PQN
│  └─ QC-based → LOESS, QC-RSC, QC-SVRC

├─ SCALING: How to transform data?
│  ├─ Equal importance → Unit variance (UV) scaling
│  ├─ Reduce impact of large peaks → Pareto scaling
│  └─ Reduce heteroscedasticity → Log transformation

├─ MULTIVARIATE ANALYSIS: Which method?
│  ├─ Exploratory → PCA
│  ├─ Discrimination → PLS-DA, OPLS-DA
│  └─ Biomarker selection → S-plot, VIP scores

└─ IDENTIFICATION: What level?
   ├─ Level 1 → Authentic standard (same RT, MS, MS/MS)
   ├─ Level 2 → Library match (MS/MS spectral match)
   ├─ Level 3 → Putative class (characteristic fragments)
   └─ Level 4 → Unknown (unidentified)
```

## When to Use What

### Signal Processing (pre-acquisition + raw data to feature table)

**Pre-acquisition normalization**: Cell count, creatinine, osmolality, fixed volume -- depends on sample type.
**Peak detection**: `pyopenms` FeatureFindingMetabo pipeline. Mass accuracy 5-20 ppm, S/N > 3-10.
**RT alignment**: LOESS warping or Obiwarp DTW. Tolerance 0.1-0.5 min.
**Gap filling**: Targeted re-integration at expected m/z + RT for missing features.
Details: [references/signal-processing.md](references/signal-processing.md)

### Post-Acquisition Normalization

**TSN**: Total signal normalization. Simple but affected by dominant peaks.
**MSTUS**: Median ratio to geometric mean reference. More robust than TSN.
**PQN**: Probabilistic quotient normalization. Best for biological variation.
**QC-LOESS**: Drift correction using pooled QC samples with LOESS smoothing.
**Missing values**: Filter >30% missing, then min/2 or kNN imputation.
**Batch effects**: QC-based correction, randomization, ComBat.
Details: [references/normalization.md](references/normalization.md)

### Data Scaling

| Method | Formula | Effect | Best For |
|--------|---------|--------|----------|
| **Centering** | x - mean | Centers data | All methods |
| **UV (Autoscaling)** | (x - mean) / std | Equal importance | When all features matter equally |
| **Pareto** | (x - mean) / sqrt(std) | Reduce dominant peaks | Balanced importance |
| **Log transform** | log(x + 1) | Reduce heteroscedasticity | Skewed distributions |
| **Range scaling** | (x - min) / (max - min) | 0-1 range | Neural networks |

### Multivariate Analysis

**PCA**: `sklearn.decomposition.PCA`. Hotelling's T2 for outlier detection. Unsupervised exploration.
**PLS-DA**: `PLSRegression` + `LabelEncoder`. 7-fold CV + permutation (n>=100). Q2>0.5 required.
**VIP scores**: Variable Importance in Projection. VIP > 1 as initial biomarker filter.
**S-plot**: Covariance vs. correlation plot. High |p(cov)| + high |p(corr)| = reliable biomarkers.
**Validation**: Permutation testing (p < 0.05), R2Y-Q2 gap < 0.3, FDR correction for biomarker selection.
Details: [references/multivariate-analysis.md](references/multivariate-analysis.md)

Cross-reference: [../chemometrics-shared/references/validation-strategies.md](../chemometrics-shared/references/validation-strategies.md)

### Metabolite Identification

**MSI levels**: Level 1 (authentic standard) through Level 4 (unknown).
**Spectral matching**: Cosine similarity on normalized MS/MS peaks, mz_tolerance=0.02, min 3 matched peaks.
**Databases**: HMDB, METLIN, MassBank, LipidMaps, KEGG, MoNA.
Details: [references/metabolite-identification.md](references/metabolite-identification.md)

## Software Tools

| Task | Open Source | Commercial |
|------|-------------|------------|
| **Peak detection** | XCMS, MZmine, MS-DIAL | Compound Discoverer |
| **Alignment** | XCMS, OpenMS | Progenesis QI |
| **Statistics** | MetaboAnalyst, scikit-learn | SIMCA, Progenesis |
| **Identification** | MS-FINDER, SIRIUS | mzCloud, Lipid Search |
| **Pathway analysis** | MetaboAnalyst, KEGG | IPA, MetaCore |

## See Also

- `chemometrics-shared`: Cross-validation strategies, performance metrics, overfitting prevention
- `chemometrics-validation`: Model validation best practices for analytical chemistry
- `chemometrics-ml-selection`: Machine learning method selection for chemometrics
- `chemometrics-hybrid-modeling`: Hybrid mechanistic-ML models

## References

- **Boccard, J., & Rudaz, S. (2018).** Extracting Knowledge from MS Clinical Metabolomic Data: Processing and Analysis Strategies. *Chimia*, 72(3), 160-167. doi:10.2533/chimia.2018.160
- **Dunn, W. B., et al. (2011).** Procedures for large-scale metabolic profiling. *Nature Protocols*, 6(7), 1060-1083.
- **Sumner, L. W., et al. (2007).** Proposed minimum reporting standards for chemical analysis. *Metabolomics*, 3(3), 211-221.
- **Trygg, J., & Wold, S. (2002).** Orthogonal projections to latent structures (O-PLS). *Journal of Chemometrics*, 16(3), 119-128.
