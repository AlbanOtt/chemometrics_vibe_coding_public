# Metabolomics Preprocessing Analysis: Complete Reproduction Guide

**Analysis**: LC-MS Metabolomics Data Preprocessing Pipeline
**Dataset**: Diphenhydramine Pharmacokinetics Study (GNPS Quantification Tables)
**Created**: 2026-02-09
**Purpose**: Step-by-step instructions to reproduce the preprocessing analysis from scratch

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Requirements](#data-requirements)
4. [Environment Setup](#environment-setup)
5. [Project Structure](#project-structure)
6. [Running the Analysis](#running-the-analysis)
7. [Expected Outputs](#expected-outputs)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This analysis implements a complete preprocessing pipeline for LC-MS untargeted metabolomics data following best practices from Boccard & Rudaz (2018). The pipeline processes GNPS (Global Natural Products Social Molecular Networking) quantification tables through four main steps:

1. **Blank Filtering**: Remove contaminant features using sample/blank ratio threshold
2. **Zero Handling**: Filter unreliable features and impute missing values
3. **PQN Normalization**: Correct for dilution effects using Probabilistic Quotient Normalization
4. **Pareto Scaling**: Balance feature influence for multivariate analysis

The analysis processes two datasets (forearm and forehead skin samples) from a pharmacokinetics study monitoring topical drug absorption.

---

## Prerequisites

### Required Software

- **Python**: ≥ 3.9
- **uv**: Modern Python package manager (recommended) or pip
- **Git**: For version control (optional but recommended)
- **Quarto**: For rendering the tutorial notebook (optional, for HTML output)

### Install uv (Recommended)

**Windows (PowerShell)**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Quarto (Optional)

Download from: https://quarto.org/docs/get-started/

---

## Data Requirements

### Required Data Files

All data files should be placed in: `data/CD-MRG-metabolomics_summerschool_2023/`

| File | Description | Size | Format |
|------|-------------|------|--------|
| `forearm_iimn_gnps_quant.csv` | GNPS quantification table for forearm samples | ~4.2 MB | CSV |
| `forehead_iimn_gnps_quant.csv` | GNPS quantification table for forehead samples | ~3.9 MB | CSV |
| `metadata.txt` | Sample metadata (subjects, timepoints, sample types) | ~10 KB | Tab-separated |

### Data Structure

**GNPS Quantification Tables**:
- Rows = features (defined by m/z, retention time, ion mobility)
- First 13 columns = feature metadata (row ID, row m/z, row retention time, etc.)
- Remaining columns = peak areas for each sample (format: `<filename> Peak area`)

**Metadata File**:
- Tab-separated format
- Required columns:
  - `filename`: Sample filename matching column names in quant tables
  - `ATTRIBUTE_Sample_Type`: Sample type (sample/blank)
  - `ATTRIBUTE_Subject`: Subject ID (or "not applicable" for blanks)
  - `ATTRIBUTE_Timepoint_min`: Timepoint in minutes
  - `ATTRIBUTE_Sampling_Location`: forearm/forehead
  - `ATTRIBUTE_Analysis_order`: Injection order

### Data Source

The dataset is from the CD-MRG Metabolomics Summer School 2023. If you don't have the data files, contact the data provider or use equivalent GNPS output from your own LC-MS experiments.

---

## Environment Setup

### 1. Clone or Create Project Directory

```bash
git clone <repository-url> chemometrics_vibe_coding
cd chemometrics_vibe_coding
```

Or create from scratch:
```bash
mkdir chemometrics_vibe_coding
cd chemometrics_vibe_coding
```

### 2. Install Dependencies

Using **uv** (recommended):
```bash
uv sync
```

This reads `pyproject.toml` and installs all required packages:
- numpy ≥ 2.0.2
- pandas ≥ 2.3.3
- matplotlib ≥ 3.9.4
- seaborn ≥ 0.13.2
- scikit-learn ≥ 1.6.1
- scipy ≥ 1.13.1
- statsmodels ≥ 0.14.6
- jupyter ≥ 1.1.1

Using **pip** (alternative):
```bash
pip install numpy>=2.0.2 pandas>=2.3.3 matplotlib>=3.9.4 seaborn>=0.13.2 \
            scikit-learn>=1.6.1 scipy>=1.13.1 statsmodels>=0.14.6 jupyter>=1.1.1
```

### 3. Verify Installation

```bash
uv run python -c "import numpy, pandas, sklearn; print('✓ All dependencies installed')"
```

---

## Project Structure

```
chemometrics_vibe_coding/
├── data/
│   └── CD-MRG-metabolomics_summerschool_2023/
│       ├── forearm_iimn_gnps_quant.csv      # Raw GNPS data
│       ├── forehead_iimn_gnps_quant.csv     # Raw GNPS data
│       └── metadata.txt                      # Sample metadata
├── src/
│   └── preprocessing.py                      # Core preprocessing functions
├── reports/
│   └── preprocessing_tutorial.qmd            # Quarto tutorial notebook
├── pyproject.toml                            # Python dependencies
└── REPRODUCTION_GUIDE.md                     # This file
```

### Key Files

- **`src/preprocessing.py`**: Implementation of all preprocessing functions
  - `load_gnps_quant_table()`: Load and parse GNPS CSV files
  - `load_metadata()`: Load sample metadata
  - `filter_blanks()`: Blank filtering with ratio threshold
  - `handle_zeros()`: Zero filtering and min/2 imputation
  - `pqn_normalize()`: Probabilistic Quotient Normalization
  - `pareto_scale()`: Pareto scaling
  - `build_preprocessing_pipeline()`: Complete automated pipeline

- **`reports/preprocessing_tutorial.qmd`**: Educational Quarto notebook
  - Step-by-step tutorial with explanations
  - Diagnostic visualizations for each step
  - Runs full pipeline on both datasets

---

## Running the Analysis

### Method 1: Execute Quarto Notebook (Recommended)

This method runs the complete tutorial with all visualizations and explanations:

```bash
cd reports
quarto render preprocessing_tutorial.qmd
```

**Output**: `preprocessing_tutorial.html` (opens in browser automatically)

### Method 2: Run Pipeline via Python Script

For batch processing without the tutorial:

```python
from pathlib import Path
from src.preprocessing import build_preprocessing_pipeline
import pandas as pd

# Set paths
project_root = Path.cwd()
data_dir = project_root / "data" / "CD-MRG-metabolomics_summerschool_2023"

forearm_path = data_dir / "forearm_iimn_gnps_quant.csv"
forehead_path = data_dir / "forehead_iimn_gnps_quant.csv"
metadata_path = data_dir / "metadata.txt"

# Run pipeline
result = build_preprocessing_pipeline(
    forearm_path=forearm_path,
    forehead_path=forehead_path,
    metadata_path=metadata_path,
    blank_ratio=10.0,       # Sample/blank ratio threshold
    max_zero_ratio=0.5      # Max fraction of zeros per feature
)

# Access results
forearm_processed = result["forearm"]["X_processed"]  # samples x features array
forehead_processed = result["forehead"]["X_processed"]

# Save preprocessed data
forearm_df = pd.DataFrame(
    forearm_processed,
    index=result["forearm"]["sample_names"],
    columns=result["forearm"]["feature_ids"]
)
forearm_df.to_csv(data_dir.parent / "forearm_preprocessed.csv")

forehead_df = pd.DataFrame(
    forehead_processed,
    index=result["forehead"]["sample_names"],
    columns=result["forehead"]["feature_ids"]
)
forehead_df.to_csv(data_dir.parent / "forehead_preprocessed.csv")
```

Save as `run_pipeline.py` and execute:
```bash
uv run python run_pipeline.py
```

### Method 3: Interactive Jupyter Notebook

For exploratory analysis:

```bash
uv run jupyter notebook
```

Then create a new notebook and copy code from `preprocessing_tutorial.qmd`.

---

## Expected Outputs

### Generated Files

| File | Location | Description | Format |
|------|----------|-------------|--------|
| `preprocessing_tutorial.html` | `reports/` | Rendered tutorial with all plots | HTML |
| `forearm_preprocessed.csv` | `data/` | Preprocessed forearm data | CSV (samples × features) |
| `forehead_preprocessed.csv` | `data/` | Preprocessed forehead data | CSV (samples × features) |

### Expected Results

**Forearm Dataset**:
- Raw features: ~2,500-3,000
- After blank filtering (10× ratio): ~1,200-1,500
- After zero handling (50% threshold): ~800-1,000
- Final: **~800-1,000 features × ~130 samples**

**Forehead Dataset**:
- Raw features: ~2,300-2,700
- After blank filtering: ~1,100-1,400
- After zero handling: ~700-900
- Final: **~700-900 features × ~130 samples**

### Diagnostic Visualizations

The tutorial generates the following plots:
1. Sample-to-blank intensity ratio distribution
2. Zero count distribution per feature
3. Total intensity before/after PQN normalization
4. Feature variance before/after Pareto scaling
5. Feature retention bar chart through pipeline steps

---

## Verification

### Check Preprocessing Success

After running, verify the outputs:

```python
import pandas as pd

# Load preprocessed data
forearm = pd.read_csv("data/forearm_preprocessed.csv", index_col=0)
forehead = pd.read_csv("data/forehead_preprocessed.csv", index_col=0)

print(f"Forearm shape: {forearm.shape}")   # Should be ~(130 samples, 800-1000 features)
print(f"Forehead shape: {forehead.shape}") # Should be ~(130 samples, 700-900 features)

# Check for NaN or inf values
assert not forearm.isnull().any().any(), "Forearm contains NaN values"
assert not forehead.isnull().any().any(), "Forehead contains NaN values"
assert not (forearm == float('inf')).any().any(), "Forearm contains inf values"
assert not (forehead == float('inf')).any().any(), "Forehead contains inf values"

print("✓ Preprocessing completed successfully")
```

### Expected Processing Time

- Full Quarto render: **2-5 minutes**
- Python script only: **10-30 seconds**

---

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'src'

**Problem**: Python can't find the `src` module.

**Solution**: Ensure you're running from the project root, or add to Python path:
```python
import sys
from pathlib import Path
project_root = Path.cwd().parent  # Adjust as needed
sys.path.insert(0, str(project_root))
```

#### 2. FileNotFoundError: Data files not found

**Problem**: Data files are not in the expected location.

**Solution**: Verify paths:
```bash
ls data/CD-MRG-metabolomics_summerschool_2023/
```
Should show: `forearm_iimn_gnps_quant.csv`, `forehead_iimn_gnps_quant.csv`, `metadata.txt`

#### 3. No blank samples found warning

**Problem**: Metadata may not correctly flag blanks.

**Solution**: Check metadata file:
```python
import pandas as pd
meta = pd.read_csv("data/CD-MRG-metabolomics_summerschool_2023/metadata.txt", sep="\t")
print(meta["ATTRIBUTE_Sample_Type"].value_counts())
```
Ensure some samples have "blank" in sample type.

#### 4. Quarto command not found

**Problem**: Quarto is not installed or not in PATH.

**Solution**:
- Install Quarto from https://quarto.org
- Or use Python script method instead

#### 5. Different number of features than expected

**Problem**: Results differ from expected ranges.

**Cause**: This is normal variation depending on:
- `blank_ratio` parameter (default: 10)
- `max_zero_ratio` parameter (default: 0.5)

**Solution**: These are user-adjustable parameters. The tutorial uses `blank_ratio=10` (conservative), but you can adjust:
```python
result = build_preprocessing_pipeline(
    ...,
    blank_ratio=3.0,   # More permissive (keeps more features)
    max_zero_ratio=0.3  # More strict (removes more features)
)
```

---

## References

**Key Citations**:

1. **Boccard, J., & Rudaz, S. (2018)**. "Exploring Omics data from designed experiments using analysis of variance multiblock Orthogonal Partial Least Squares." *Analytica Chimica Acta*, 920, 18-28.
   - Source for PQN normalization and preprocessing best practices

2. **Dieterle, F., Ross, A., Schlotterbeck, G., & Senn, H. (2006)**. "Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures. Application in 1H NMR metabonomics." *Analytical Chemistry*, 78(13), 4281-4290.
   - Original PQN method publication

3. **van den Berg, R. A., Hoefsloot, H. C., Westerhuis, J. A., Smilde, A. K., & van der Werf, M. J. (2006)**. "Centering, scaling, and transformations: improving the biological information content of metabolomics data." *BMC Genomics*, 7(1), 142.
   - Comprehensive review of scaling methods including Pareto

---

## Notes

### Important Considerations

1. **Parameter Choices**:
   - `blank_ratio=10`: Conservative threshold (original study used 3, tutorial uses 10)
   - `max_zero_ratio=0.5`: Standard threshold for metabolomics
   - These can be adjusted based on data quality and analysis goals

2. **Imputation Method**:
   - The tutorial uses **min/2 imputation** for educational purposes
   - This is a common but suboptimal method (reduces variance artificially)
   - Consider random imputation between 0 and min for production analyses

3. **Normalization Choice**:
   - Tutorial uses **PQN** (recommended by Boccard & Rudaz)
   - Alternatives: TSN (Total Sum Normalization), MSTUS
   - PQN is more robust to large biological changes

4. **Scaling Choice**:
   - Tutorial uses **Pareto scaling** (balanced approach)
   - Alternatives: Unit variance (UV/autoscaling), no scaling
   - Pareto is standard for metabolomics multivariate analysis

---

## Next Steps

After preprocessing, typical next steps include:

1. **Exploratory Data Analysis (EDA)**: PCA for outlier detection and data structure
2. **Multivariate Modeling**: PLS-DA, OPLS-DA for classification/discrimination
3. **Biomarker Discovery**: Feature selection, variable importance
4. **Statistical Testing**: Univariate tests, multiple testing correction
5. **Metabolite Identification**: Match features to metabolite databases

See additional analysis scripts in `src/` directory for these workflows.

---

## Contact & Support

For questions about:
- **Data**: Contact CD-MRG Metabolomics Summer School organizers
- **Methods**: See references above or Boccard & Rudaz (2018)
- **Code**: Check project repository issues or documentation

---

**Last Updated**: 2026-02-09
**Analysis Version**: 0.1.0
**Python Version**: ≥ 3.9
