"""Data loading and preprocessing pipeline for LC-MS metabolomics data.

Handles GNPS quantification tables from the diphenhydramine pharmacokinetics study.
Implements: data loading, blank filtering, zero handling, PQN normalization, and Pareto scaling.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature metadata columns in GNPS quant tables
FEATURE_META_COLS = [
    "row ID",
    "row m/z",
    "row retention time",
    "row ion mobility",
    "row ion mobility unit",
    "row CCS",
    "correlation group ID",
    "annotation network number",
    "best ion",
    "auto MS2 verify",
    "identified by n=",
    "partners",
    "neutral M mass",
]

PEAK_AREA_SUFFIX = " Peak area"


def load_gnps_quant_table(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a GNPS quantification CSV and separate feature metadata from peak areas.

    Parameters
    ----------
    path : Path
        Path to the GNPS quant CSV file.

    Returns
    -------
    feature_metadata : pd.DataFrame
        Feature metadata (m/z, RT, neutral mass, etc.) indexed by row ID.
    peak_areas : pd.DataFrame
        Peak area matrix with features as rows and samples as columns.
        Column names are the sample filenames (without ' Peak area' suffix).
    """
    df = pd.read_csv(path)
    logger.info(
        "Loaded %d features x %d columns from %s", len(df), len(df.columns), path.name
    )

    # Separate metadata and peak area columns
    meta_cols = [c for c in df.columns if c in FEATURE_META_COLS]
    peak_cols = [c for c in df.columns if c.endswith(PEAK_AREA_SUFFIX)]

    feature_metadata = df[meta_cols].copy()
    feature_metadata = feature_metadata.set_index("row ID")

    peak_areas = df[peak_cols].copy()
    peak_areas.index = df["row ID"]
    # Clean column names: remove ' Peak area' suffix
    peak_areas.columns = [c.replace(PEAK_AREA_SUFFIX, "") for c in peak_areas.columns]

    logger.info(
        "Found %d features, %d samples", len(feature_metadata), len(peak_areas.columns)
    )
    return feature_metadata, peak_areas


def load_metadata(path: Path) -> pd.DataFrame:
    """Load and parse sample metadata from tab-separated file.

    Parameters
    ----------
    path : Path
        Path to metadata.txt file.

    Returns
    -------
    metadata : pd.DataFrame
        Parsed metadata with columns: filename, sample_type, subject, timepoint, location.
        Rows with 'not applicable' subjects (blanks) are included but flagged.
    """
    meta = pd.read_csv(path, sep="\t")
    meta = meta.rename(
        columns={
            "ATTRIBUTE_Sample_Type": "sample_type",
            "ATTRIBUTE_Subject": "subject",
            "ATTRIBUTE_Timepoint_min": "timepoint",
            "ATTRIBUTE_Sampling_Location": "location",
            "ATTRIBUTE_Analysis_order": "analysis_order",
        }
    )

    # Convert timepoint to numeric where possible
    meta["timepoint"] = pd.to_numeric(meta["timepoint"], errors="coerce")

    # Flag blanks
    meta["is_blank"] = meta["sample_type"].str.contains("blank", case=False)

    logger.info(
        "Loaded metadata: %d samples (%d blanks)",
        len(meta),
        meta["is_blank"].sum(),
    )
    return meta


def filter_blanks(
    peak_areas: pd.DataFrame,
    metadata: pd.DataFrame,
    blank_ratio: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove blank samples and features dominated by blank signal.

    Parameters
    ----------
    peak_areas : pd.DataFrame
        Features (rows) x samples (columns). Column names are filenames.
    metadata : pd.DataFrame
        Sample metadata with 'filename' and 'is_blank' columns.
    blank_ratio : float
        Minimum ratio of sample mean / blank mean to keep a feature.

    Returns
    -------
    filtered_peak_areas : pd.DataFrame
        Peak areas with blank samples removed and blank-dominated features filtered.
    metadata_no_blanks : pd.DataFrame
        Metadata without blank samples.
    """
    # Identify blank and non-blank sample columns present in peak_areas
    blank_files = set(metadata.loc[metadata["is_blank"], "filename"])
    all_files = set(peak_areas.columns)

    blank_cols = sorted(blank_files & all_files)
    sample_cols = sorted(all_files - blank_files)

    if not blank_cols:
        logger.warning("No blank samples found in peak area table")
        return peak_areas, metadata[~metadata["is_blank"]]

    # Calculate mean intensity in blanks vs samples
    blank_mean = peak_areas[blank_cols].mean(axis=1)
    sample_mean = peak_areas[sample_cols].mean(axis=1)

    # Keep features where sample signal is at least blank_ratio times blank signal
    # Avoid division by zero: if blank_mean is 0, feature passes
    ratio = sample_mean / blank_mean.replace(0, np.nan)
    keep_mask = ratio.isna() | (ratio >= blank_ratio)

    n_removed = (~keep_mask).sum()
    logger.info(
        "Blank filtering: removed %d/%d features (ratio threshold: %.1f)",
        n_removed,
        len(peak_areas),
        blank_ratio,
    )

    # Remove blank columns and filtered features
    filtered = peak_areas.loc[keep_mask, sample_cols]
    metadata_no_blanks = metadata[~metadata["is_blank"]].copy()

    return filtered, metadata_no_blanks


def handle_zeros(
    peak_areas: pd.DataFrame,
    max_zero_ratio: float = 0.5,
) -> pd.DataFrame:
    """Filter features with too many zeros and impute remaining zeros with min/2.

    Parameters
    ----------
    peak_areas : pd.DataFrame
        Peak area matrix (features x samples).
    max_zero_ratio : float
        Maximum fraction of zeros allowed per feature.

    Returns
    -------
    imputed : pd.DataFrame
        Filtered and imputed peak area matrix.
    """
    n_samples = peak_areas.shape[1]
    zero_counts = (peak_areas == 0).sum(axis=1)
    zero_ratio = zero_counts / n_samples

    keep_mask = zero_ratio <= max_zero_ratio
    n_removed = (~keep_mask).sum()
    logger.info(
        "Zero filtering: removed %d/%d features (>%.0f%% zeros)",
        n_removed,
        len(peak_areas),
        max_zero_ratio * 100,
    )

    filtered = peak_areas.loc[keep_mask].copy()

    # Replace remaining zeros with min/2 per feature
    for feature_id in filtered.index:
        row = filtered.loc[feature_id]
        nonzero_vals = row[row > 0]
        if len(nonzero_vals) > 0:
            min_val = nonzero_vals.min()
            filtered.loc[feature_id] = row.replace(0, min_val / 2)

    n_zeros_remaining = (filtered == 0).sum().sum()
    if n_zeros_remaining > 0:
        logger.warning("%d zeros remain after imputation", n_zeros_remaining)

    return filtered


def pqn_normalize(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Probabilistic Quotient Normalization.

    Robust normalization method that accounts for dilution effects
    while being resistant to biological variation.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (samples x features).
    reference : np.ndarray, optional
        Reference spectrum. If None, the median spectrum is used.

    Returns
    -------
    X_normalized : np.ndarray
        PQN-normalized data matrix.
    """
    if reference is None:
        reference = np.median(X, axis=0)

    # Avoid division by zero in reference
    ref_nonzero = reference.copy()
    ref_nonzero[ref_nonzero == 0] = np.nan

    # Calculate quotients
    quotients = X / ref_nonzero

    # Normalization factors from median quotient per sample
    norm_factors = np.nanmedian(quotients, axis=1, keepdims=True)

    return X / norm_factors


def pareto_scale(X: np.ndarray) -> np.ndarray:
    """Pareto scaling: mean-center and divide by sqrt of standard deviation.

    Reduces the influence of large peaks while preserving data structure
    better than unit variance scaling.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (samples x features).

    Returns
    -------
    X_scaled : np.ndarray
        Pareto-scaled data matrix.
    """
    X_centered = X - X.mean(axis=0)
    std = X.std(axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0
    return X_centered / np.sqrt(std)


def _match_samples_to_metadata(
    peak_areas: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Match peak area sample columns to metadata rows.

    Returns metadata filtered to only samples present in peak_areas,
    with column order matching metadata row order.
    """
    available = set(peak_areas.columns)
    matched_meta = metadata[metadata["filename"].isin(available)].copy()

    if len(matched_meta) < len(metadata[~metadata["is_blank"]]):
        n_missing = len(metadata[~metadata["is_blank"]]) - len(matched_meta)
        logger.warning(
            "%d non-blank metadata samples not found in peak area table", n_missing
        )

    return matched_meta


def build_preprocessing_pipeline(
    forearm_path: Path,
    forehead_path: Path,
    metadata_path: Path,
    blank_ratio: float = 3.0,
    max_zero_ratio: float = 0.5,
) -> dict:
    """Run the full preprocessing pipeline on both forearm and forehead datasets.

    Pipeline steps:
    1. Load GNPS quant tables
    2. Load metadata
    3. Filter blank-dominated features
    4. Handle zeros (filter + min/2 imputation)
    5. PQN normalization
    6. Pareto scaling

    Parameters
    ----------
    forearm_path : Path
        Path to forearm GNPS quant CSV.
    forehead_path : Path
        Path to forehead GNPS quant CSV.
    metadata_path : Path
        Path to metadata.txt.
    blank_ratio : float
        Minimum sample/blank intensity ratio to keep a feature.
    max_zero_ratio : float
        Maximum fraction of zeros per feature.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'forearm': dict with 'feature_metadata', 'peak_areas_raw', 'X_processed',
          'sample_names', 'metadata'
        - 'forehead': same structure
        - 'metadata_full': full metadata DataFrame
    """
    metadata = load_metadata(metadata_path)

    result: dict = {"metadata_full": metadata}

    for name, path in [("forearm", forearm_path), ("forehead", forehead_path)]:
        logger.info("Processing %s dataset from %s", name, path.name)

        # Step 1: Load
        feature_meta, peak_areas = load_gnps_quant_table(path)

        # Step 2: Filter blanks
        peak_areas_filtered, meta_no_blanks = filter_blanks(
            peak_areas, metadata, blank_ratio
        )

        # Step 3: Handle zeros
        peak_areas_imputed = handle_zeros(peak_areas_filtered, max_zero_ratio)

        # Step 4: Match metadata to available samples
        matched_meta = _match_samples_to_metadata(peak_areas_imputed, meta_no_blanks)
        sample_order = matched_meta["filename"].tolist()

        # Transpose to samples x features for normalization/scaling
        X = peak_areas_imputed[sample_order].values.T

        # Step 5: PQN normalization
        X_norm = pqn_normalize(X)

        # Step 6: Pareto scaling
        X_scaled = pareto_scale(X_norm)

        # Filter feature metadata to match retained features
        retained_features = peak_areas_imputed.index
        feature_meta_filtered = feature_meta.loc[
            feature_meta.index.isin(retained_features)
        ]

        result[name] = {
            "feature_metadata": feature_meta_filtered,
            "peak_areas_raw": peak_areas_imputed,
            "X_processed": X_scaled,
            "sample_names": sample_order,
            "feature_ids": retained_features.tolist(),
            "metadata": matched_meta.reset_index(drop=True),
        }

        logger.info(
            "%s: %d samples x %d features after preprocessing",
            name,
            X_scaled.shape[0],
            X_scaled.shape[1],
        )

    return result
