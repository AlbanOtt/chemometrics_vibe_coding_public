# MS Signal Processing

## Pre-Acquisition Normalization

Sample-type specific strategies for normalizing biological variability before MS analysis.

### Cell Cultures

```python
# Normalize cell extract metabolites by cell count
def normalize_by_cell_count(peak_areas: np.ndarray, cell_counts: np.ndarray) -> np.ndarray:
    """Normalize peak areas by cell count."""
    return peak_areas / cell_counts[:, np.newaxis]

# Alternative: normalize by protein content
def normalize_by_protein(peak_areas: np.ndarray, protein_ug: np.ndarray) -> np.ndarray:
    """Normalize peak areas by protein content (ug)."""
    return peak_areas / protein_ug[:, np.newaxis]
```

### Urine Samples

Urine concentration varies significantly. Common approaches:

1. **Creatinine normalization:** Divide by creatinine concentration
2. **Osmolality normalization:** Dilute to constant osmolality (300-500 mOsm/kg)
3. **Specific gravity normalization:** Adjust for dilution using refractometer

```python
def normalize_urine_creatinine(
    peak_areas: np.ndarray, creatinine_mm: np.ndarray
) -> np.ndarray:
    """Normalize urine metabolites by creatinine concentration."""
    # Creatinine in mM (typical range: 2-20 mM)
    return peak_areas / creatinine_mm[:, np.newaxis]
```

### Blood/Plasma/Serum

Fixed volume approach is standard. No pre-acquisition normalization needed, but ensure:
- Consistent collection tubes (EDTA, heparin, or serum)
- Standardized processing time (< 30 min to centrifugation)
- Controlled storage conditions (-80 C)

## Peak Detection

Convert raw MS data to a feature table (samples x features).

**Key parameters:**
- **Mass accuracy:** +/-5-20 ppm for high-resolution MS
- **RT tolerance:** +/-0.1-0.5 min depending on chromatography
- **Intensity threshold:** Signal-to-noise ratio > 3-10

```python
# Using pyopenms for peak detection
from pyopenms import (
    MSExperiment, MzMLFile, MassTraceDetection, ElutionPeakDetection,
    FeatureFindingMetabo, FeatureMap
)

def detect_features_pyopenms(mzml_path: str) -> FeatureMap:
    """Detect features from mzML file using pyOpenMS."""
    # Load raw data
    exp = MSExperiment()
    MzMLFile().load(mzml_path, exp)

    # Mass trace detection
    mass_traces = []
    mtd = MassTraceDetection()
    mtd_params = mtd.getParameters()
    mtd_params.setValue("mass_error_ppm", 10.0)
    mtd_params.setValue("noise_threshold_int", 1000.0)
    mtd.setParameters(mtd_params)
    mtd.run(exp, mass_traces, 0)

    # Elution peak detection
    epd = ElutionPeakDetection()
    epd_params = epd.getParameters()
    epd_params.setValue("width_filtering", "auto")
    epd.setParameters(epd_params)

    mass_traces_final = []
    epd.detectPeaks(mass_traces, mass_traces_final)

    # Feature finding
    ffm = FeatureFindingMetabo()
    feature_map = FeatureMap()
    ffm.run(mass_traces_final, feature_map, [])

    return feature_map
```

## Retention Time Alignment

Correct for RT drift between samples using warping algorithms.

**Common methods:**
- **Obiwarp (DTW):** Dynamic time warping, robust to large shifts
- **Loess:** Locally weighted regression, smooth corrections
- **Peak groups:** Align based on reference peak positions

```python
def align_retention_times(
    feature_rts: np.ndarray,  # shape: (n_samples, n_features)
    reference_rts: np.ndarray,  # shape: (n_features,)
    method: str = "loess"
) -> np.ndarray:
    """Align retention times to reference sample."""
    from scipy.interpolate import UnivariateSpline
    import numpy as np

    aligned_rts = np.zeros_like(feature_rts)

    for i in range(feature_rts.shape[0]):
        sample_rts = feature_rts[i, :]

        # Remove NaN values for fitting
        valid_mask = ~np.isnan(sample_rts) & ~np.isnan(reference_rts)

        if valid_mask.sum() < 10:
            aligned_rts[i, :] = sample_rts
            continue

        # Fit warping function
        if method == "loess":
            from statsmodels.nonparametric.smoothers_lowess import lowess
            warped = lowess(
                reference_rts[valid_mask],
                sample_rts[valid_mask],
                frac=0.3
            )
            # Interpolate to all features
            spline = UnivariateSpline(
                warped[:, 0], warped[:, 1], s=0, ext=3
            )
            aligned_rts[i, :] = spline(sample_rts)
        else:
            aligned_rts[i, :] = sample_rts

    return aligned_rts
```

## Gap Filling

Re-integrate peaks for features not detected in all samples.

```python
def fill_gaps(
    feature_table: pd.DataFrame,
    raw_data_paths: list[str],
    mz_tolerance_ppm: float = 10.0,
    rt_tolerance_min: float = 0.2
) -> pd.DataFrame:
    """Fill missing values by targeted re-integration."""
    filled_table = feature_table.copy()

    for idx, row in feature_table.iterrows():
        if row.isna().any():
            target_mz = row.name[0]  # Assuming multi-index (mz, rt)
            target_rt = row.name[1]

            for col in row.index[row.isna()]:
                # Re-integrate from raw data at expected position
                intensity = reintegrate_peak(
                    raw_data_paths[col],
                    target_mz,
                    target_rt,
                    mz_tolerance_ppm,
                    rt_tolerance_min
                )
                filled_table.loc[idx, col] = intensity

    return filled_table
```

## Python Ecosystem

```python
# Core packages for MS metabolomics
# pip install pyopenms matchms metaspace2020

# Data processing
import pyopenms  # OpenMS Python bindings
import matchms   # MS/MS spectral matching

# Statistics
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data handling
import pandas as pd
import numpy as np
```
