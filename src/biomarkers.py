"""Proxy biomarker discovery: correlation analysis and candidate selection.

Implements skin-plasma pairing, univariate correlation screening with FDR correction,
PLS regression with LOSO cross-validation, and composite biomarker ranking for
identifying skin metabolite features that serve as non-invasive proxies for
plasma drug levels.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.stats.multitest import multipletests

from src.drug_detection import TARGET_COMPOUNDS, search_features_by_mz
from src.multivariate import VIP_THRESHOLD, calculate_vip

logger = logging.getLogger(__name__)

DEFAULT_FDR_THRESHOLD = 0.05
DEFAULT_CORRELATION_THRESHOLD = 0.5
DEFAULT_N_COMPONENTS_PLS = 2
DEFAULT_N_TOP_CANDIDATES = 20


def pair_skin_plasma_samples(
    peak_areas_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    target_mz: float,
    tolerance_ppm: float = 10.0,
) -> dict:
    """Pair skin and plasma samples to extract aligned subject-timepoint data.

    Finds the diphenhydramine feature in plasma, then pairs plasma DPH intensity
    with skin samples based on matching (subject, timepoint) combinations.

    Parameters
    ----------
    peak_areas_raw : pd.DataFrame
        Raw peak area matrix (features x samples).
    metadata : pd.DataFrame
        Sample metadata with columns: filename, subject, timepoint, sample_type.
    feature_metadata : pd.DataFrame
        Feature metadata with 'row m/z' column.
    target_mz : float
        Target m/z for diphenhydramine.
    tolerance_ppm : float
        Mass tolerance in ppm for feature search.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - status: str ("success" or "dph_not_found")
        - n_pairs: int (number of paired samples)
        - skin_indices: np.ndarray (row indices into metadata for paired skin samples)
        - y_plasma_dph: np.ndarray (plasma DPH raw intensities for paired samples)
        - subjects: np.ndarray (subject IDs for paired samples)
        - timepoints: np.ndarray (timepoints for paired samples)
        - dph_feature_id: int or None (feature ID for DPH)
        - pairing_df: pd.DataFrame (full pairing info with subject, timepoint, indices)
    """
    # Find DPH feature
    matches = search_features_by_mz(feature_metadata, target_mz, tolerance_ppm)
    if matches.empty:
        logger.warning(
            "DPH feature not found at m/z %.4f +/- %.1f ppm", target_mz, tolerance_ppm
        )
        return {
            "status": "dph_not_found",
            "n_pairs": 0,
            "skin_indices": np.array([], dtype=int),
            "y_plasma_dph": np.array([]),
            "subjects": np.array([]),
            "timepoints": np.array([]),
            "dph_feature_id": None,
            "pairing_df": pd.DataFrame(),
        }

    # Use the best match (smallest ppm error)
    dph_feature_id = int(matches.index[0])

    # Split metadata by sample type
    plasma_meta = metadata[metadata["sample_type"] == "plasma"].copy()
    skin_meta = metadata[metadata["sample_type"] == "skin"].copy()

    # Extract plasma DPH intensities
    plasma_meta["plasma_dph_intensity"] = plasma_meta["filename"].map(
        peak_areas_raw.loc[dph_feature_id]
    )

    # Inner join on (subject, timepoint) to pair samples
    pairing = skin_meta[["subject", "timepoint", "filename"]].merge(
        plasma_meta[["subject", "timepoint", "plasma_dph_intensity"]],
        on=["subject", "timepoint"],
        how="inner",
    )

    n_pairs = len(pairing)

    if n_pairs == 0:
        logger.warning("No paired skin-plasma samples found")
        return {
            "status": "success",
            "n_pairs": 0,
            "skin_indices": np.array([], dtype=int),
            "y_plasma_dph": np.array([]),
            "subjects": np.array([]),
            "timepoints": np.array([]),
            "dph_feature_id": dph_feature_id,
            "pairing_df": pairing,
        }

    # Get row indices for skin samples
    skin_indices = metadata.index[
        metadata["filename"].isin(pairing["filename"])
    ].to_numpy()

    # Extract arrays for modeling
    y_plasma_dph = pairing["plasma_dph_intensity"].to_numpy()
    subjects = pairing["subject"].to_numpy()
    timepoints = pairing["timepoint"].to_numpy()

    logger.info(
        "Paired %d skin-plasma samples (%d subjects, DPH feature ID %d)",
        n_pairs,
        pairing["subject"].nunique(),
        dph_feature_id,
    )

    return {
        "status": "success",
        "n_pairs": n_pairs,
        "skin_indices": skin_indices,
        "y_plasma_dph": y_plasma_dph,
        "subjects": subjects,
        "timepoints": timepoints,
        "dph_feature_id": dph_feature_id,
        "pairing_df": pairing,
    }


def correlate_skin_features_to_target(
    X_skin: np.ndarray,
    y_target: np.ndarray,
    feature_ids: list[int],
    fdr_threshold: float = DEFAULT_FDR_THRESHOLD,
) -> pd.DataFrame:
    """Compute Spearman correlations between skin features and target variable.

    Applies FDR correction using Benjamini-Hochberg method for multiple testing.

    Parameters
    ----------
    X_skin : np.ndarray
        Preprocessed skin feature matrix (samples x features).
    y_target : np.ndarray
        Target variable (e.g., plasma DPH intensities).
    feature_ids : list[int]
        Feature identifiers matching columns of X_skin.
    fdr_threshold : float
        FDR threshold for significance.

    Returns
    -------
    results : pd.DataFrame
        Correlation results with columns:
        - feature_id: Feature ID
        - rho: Spearman correlation coefficient
        - pvalue: Raw p-value
        - pvalue_fdr: FDR-corrected p-value
        - abs_rho: Absolute correlation coefficient
        - significant: Boolean (FDR < threshold)
        Sorted by abs_rho descending.
    """
    n_features = X_skin.shape[1]
    rhos = np.zeros(n_features)
    pvalues = np.zeros(n_features)

    for j in range(n_features):
        rho, pval = stats.spearmanr(X_skin[:, j], y_target)
        rhos[j] = rho if np.isfinite(rho) else 0.0
        pvalues[j] = pval if np.isfinite(pval) else 1.0

    # FDR correction
    _, pvalues_fdr, _, _ = multipletests(pvalues, alpha=fdr_threshold, method="fdr_bh")

    results = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "rho": rhos,
            "pvalue": pvalues,
            "pvalue_fdr": pvalues_fdr,
            "abs_rho": np.abs(rhos),
            "significant": pvalues_fdr < fdr_threshold,
        }
    )

    results = results.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    n_sig = int(results["significant"].sum())
    logger.info(
        "Correlation screening: %d/%d features significant (FDR < %.2f)",
        n_sig,
        n_features,
        fdr_threshold,
    )

    return results


def select_biomarker_candidates(
    correlation_results: pd.DataFrame,
    vip_scores: np.ndarray,
    feature_ids: list[int],
    fdr_threshold: float = DEFAULT_FDR_THRESHOLD,
    vip_threshold: float = VIP_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Select biomarker candidates using composite ranking.

    Combines FDR-corrected correlation significance, VIP scores from PLS regression,
    and correlation magnitude to rank features.

    Parameters
    ----------
    correlation_results : pd.DataFrame
        Output from correlate_skin_features_to_target.
    vip_scores : np.ndarray
        VIP scores from PLS regression (length = n_features).
    feature_ids : list[int]
        Feature identifiers.
    fdr_threshold : float
        FDR threshold for significance.
    vip_threshold : float
        VIP threshold for importance.
    correlation_threshold : float
        Minimum absolute correlation coefficient.

    Returns
    -------
    candidates : pd.DataFrame
        Biomarker candidates with columns:
        - feature_id: Feature ID
        - rho: Spearman correlation
        - pvalue_fdr: FDR-corrected p-value
        - abs_rho: Absolute correlation
        - vip: VIP score
        - candidate: Boolean (passes all thresholds)
        - rank_score: Composite score (abs_rho * vip)
        Sorted by rank_score descending.
    """
    # Merge correlation results with VIP scores
    vip_df = pd.DataFrame({"feature_id": feature_ids, "vip": vip_scores})

    candidates = correlation_results.merge(vip_df, on="feature_id", how="inner")

    # Apply filters
    candidates["candidate"] = (
        (candidates["pvalue_fdr"] < fdr_threshold)
        & (candidates["vip"] > vip_threshold)
        & (candidates["abs_rho"] > correlation_threshold)
    )

    # Composite ranking
    candidates["rank_score"] = candidates["abs_rho"] * candidates["vip"]

    candidates = candidates.sort_values("rank_score", ascending=False).reset_index(
        drop=True
    )

    n_candidates = int(candidates["candidate"].sum())
    logger.info(
        "Selected %d biomarker candidates (FDR<%.2f, VIP>%.1f, |rho|>%.2f)",
        n_candidates,
        fdr_threshold,
        vip_threshold,
        correlation_threshold,
    )

    return candidates


def pls_regression_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS_PLS,
) -> dict:
    """Perform PLS regression with leave-one-subject-out cross-validation.

    Fits a full PLS regression model and validates it using LOSO-CV to prevent
    within-subject data leakage.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed feature matrix (samples x features).
    y : np.ndarray
        Target variable (e.g., plasma DPH intensities).
    groups : np.ndarray
        Subject identifiers for LOSO-CV.
    n_components : int
        Number of PLS components.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - model: Fitted PLSRegression object
        - r2: Training R² score
        - q2: Cross-validated Q² score
        - rmsecv: Root mean squared error of cross-validation
        - bias: Mean prediction error from CV
        - vip: VIP scores (n_features,)
        - y_pred_cv: Cross-validated predictions
        - y_actual: Actual y values (CV order)
        - subjects_cv: Subject IDs for CV predictions
        - n_components: Number of components used
        - n_subjects: Number of subjects
    """
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])

    # Fit full model
    model = PLSRegression(n_components=n_components)
    model.fit(X, y)

    # Training R²
    r2 = float(model.score(X, y))

    # LOSO-CV
    logo = LeaveOneGroupOut()
    y_pred_cv = np.zeros(len(y))
    y_actual = np.zeros(len(y))
    subjects_cv = np.zeros(len(y), dtype=groups.dtype)

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        cv_model = PLSRegression(n_components=n_components)
        cv_model.fit(X_train, y_train)
        y_pred_cv[test_idx] = cv_model.predict(X_test).ravel()
        y_actual[test_idx] = y_test
        subjects_cv[test_idx] = groups[test_idx]

    # Compute Q², RMSECV, bias
    ss_res = np.sum((y_actual - y_pred_cv) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    q2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmsecv = float(np.sqrt(np.mean((y_actual - y_pred_cv) ** 2)))
    bias = float(np.mean(y_pred_cv - y_actual))

    # VIP from full model
    vip = calculate_vip(model)

    n_subjects = int(len(np.unique(groups)))

    logger.info(
        "PLS regression LOSO-CV: %d components, R²=%.3f, Q²=%.3f, RMSECV=%.1f, %d subjects",
        n_components,
        r2,
        q2,
        rmsecv,
        n_subjects,
    )

    return {
        "model": model,
        "r2": r2,
        "q2": q2,
        "rmsecv": rmsecv,
        "bias": bias,
        "vip": vip,
        "y_pred_cv": y_pred_cv,
        "y_actual": y_actual,
        "subjects_cv": subjects_cv,
        "n_components": n_components,
        "n_subjects": n_subjects,
    }


def run_biomarker_discovery(
    X_processed: np.ndarray,
    metadata: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    feature_ids: list[int],
    peak_areas_raw: pd.DataFrame,
    target_mz: float = TARGET_COMPOUNDS["Diphenhydramine"],
    tolerance_ppm: float = 10.0,
    n_components: int = DEFAULT_N_COMPONENTS_PLS,
    fdr_threshold: float = DEFAULT_FDR_THRESHOLD,
    vip_threshold: float = VIP_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> dict:
    """Orchestrate complete biomarker discovery workflow.

    Pairs skin-plasma samples, computes univariate correlations, fits PLS regression
    with LOSO-CV, and selects biomarker candidates.

    Parameters
    ----------
    X_processed : np.ndarray
        Preprocessed feature matrix (samples x features, including both plasma and skin).
    metadata : pd.DataFrame
        Sample metadata with columns: filename, subject, timepoint, sample_type.
    feature_metadata : pd.DataFrame
        Feature metadata with 'row m/z' column.
    feature_ids : list[int]
        Feature identifiers matching columns of X_processed.
    peak_areas_raw : pd.DataFrame
        Raw peak area matrix (features x samples).
    target_mz : float
        Target m/z for diphenhydramine.
    tolerance_ppm : float
        Mass tolerance in ppm.
    n_components : int
        Number of PLS components.
    fdr_threshold : float
        FDR threshold for correlation significance.
    vip_threshold : float
        VIP threshold for importance.
    correlation_threshold : float
        Minimum absolute correlation coefficient.

    Returns
    -------
    result : dict
        Nested dictionary with keys:
        - pairing: dict from pair_skin_plasma_samples
        - correlation: pd.DataFrame from correlate_skin_features_to_target
        - pls_result: dict from pls_regression_loso_cv
        - candidates: pd.DataFrame from select_biomarker_candidates
        - status: str ("success" or "dph_not_found")
    """
    # Step 1: Pair samples
    pairing = pair_skin_plasma_samples(
        peak_areas_raw,
        metadata,
        feature_metadata,
        target_mz,
        tolerance_ppm,
    )

    if pairing["status"] == "dph_not_found" or pairing["n_pairs"] == 0:
        return {
            "pairing": pairing,
            "correlation": pd.DataFrame(),
            "pls_result": {},
            "candidates": pd.DataFrame(),
            "status": pairing["status"],
        }

    # Extract paired skin data
    skin_indices = pairing["skin_indices"]
    X_skin_paired = X_processed[skin_indices]
    y_plasma_dph = pairing["y_plasma_dph"]
    subjects = pairing["subjects"]

    # Step 2: Univariate correlation screening
    correlation_results = correlate_skin_features_to_target(
        X_skin_paired,
        y_plasma_dph,
        feature_ids,
        fdr_threshold,
    )

    # Step 3: PLS regression with LOSO-CV
    pls_result = pls_regression_loso_cv(
        X_skin_paired,
        y_plasma_dph,
        subjects,
        n_components,
    )

    # Step 4: Candidate selection
    candidates = select_biomarker_candidates(
        correlation_results,
        pls_result["vip"],
        feature_ids,
        fdr_threshold,
        vip_threshold,
        correlation_threshold,
    )

    logger.info("Biomarker discovery complete")

    return {
        "pairing": pairing,
        "correlation": correlation_results,
        "pls_result": pls_result,
        "candidates": candidates,
        "status": "success",
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_correlation_volcano(
    correlation_results: pd.DataFrame,
    fdr_threshold: float = DEFAULT_FDR_THRESHOLD,
    n_top_labels: int = 10,
    ax: Axes | None = None,
) -> Axes:
    """Plot volcano-style correlation screening results.

    X-axis is Spearman rho, Y-axis is -log10(FDR p-value).

    Parameters
    ----------
    correlation_results : pd.DataFrame
        Output from correlate_skin_features_to_target.
    fdr_threshold : float
        FDR threshold for significance line.
    n_top_labels : int
        Number of top features to label.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    neg_log_p = -np.log10(
        np.clip(correlation_results["pvalue_fdr"].values, 1e-300, 1.0)
    )
    rho = correlation_results["rho"].values
    significant = correlation_results["significant"].values

    ax.scatter(
        rho[~significant],
        neg_log_p[~significant],
        c="grey",
        alpha=0.3,
        s=15,
        zorder=1,
    )
    ax.scatter(
        rho[significant],
        neg_log_p[significant],
        c="tab:red",
        alpha=0.7,
        s=30,
        zorder=2,
        label=f"Significant (n={int(significant.sum())})",
    )

    # Label top features
    top_indices = correlation_results.index[:n_top_labels]
    for idx in top_indices:
        feature_id = correlation_results.loc[idx, "feature_id"]
        ax.annotate(
            str(feature_id),
            (rho[idx], neg_log_p[idx]),
            fontsize=6,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.axhline(
        -np.log10(fdr_threshold),
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"FDR={fdr_threshold}",
    )
    ax.set_xlabel("Spearman ρ (skin feature vs plasma DPH)")
    ax.set_ylabel("-log₁₀(FDR p-value)")
    ax.set_title("Univariate Correlation Screening")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    return ax


def plot_predicted_vs_observed(
    pls_result: dict,
    location_name: str = "",
    ax: Axes | None = None,
) -> Axes:
    """Plot predicted vs observed values from PLS regression LOSO-CV.

    Colors points by subject to visualize subject-specific patterns.

    Parameters
    ----------
    pls_result : dict
        Output from pls_regression_loso_cv.
    location_name : str
        Location name for plot title (e.g., "forearm", "forehead").
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    y_actual = pls_result["y_actual"]
    y_pred_cv = pls_result["y_pred_cv"]
    subjects_cv = pls_result["subjects_cv"]
    q2 = pls_result["q2"]
    rmsecv = pls_result["rmsecv"]

    # Color by subject
    unique_subjects = sorted(np.unique(subjects_cv))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_subjects), 1)))

    for i, subject in enumerate(unique_subjects):
        mask = subjects_cv == subject
        color = colors[i % len(colors)]
        ax.scatter(
            y_actual[mask],
            y_pred_cv[mask],
            c=[color],
            label=str(subject),
            s=60,
            edgecolors="white",
            linewidth=0.5,
            alpha=0.8,
        )

    # Identity line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="grey", alpha=0.5, zorder=1)

    ax.set_xlabel("Observed (plasma DPH, raw peak area)")
    ax.set_ylabel("Predicted (PLS LOSO-CV)")
    title = f"PLS Regression: Q²={q2:.3f}, RMSECV={rmsecv:.0f}"
    if location_name:
        title = f"{location_name.capitalize()} — {title}"
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", title="Subject", ncol=2)
    ax.grid(alpha=0.3)

    return ax


def plot_candidate_heatmap(
    X_skin_paired: np.ndarray,
    candidates: pd.DataFrame,
    pairing: dict,
    feature_ids: list[int],
    n_top: int = DEFAULT_N_TOP_CANDIDATES,
    ax: Axes | None = None,
) -> Axes:
    """Plot heatmap of top biomarker candidates across samples.

    Rows are top candidates, columns are samples (grouped by subject and timepoint).

    Parameters
    ----------
    X_skin_paired : np.ndarray
        Paired skin feature matrix (samples x features, preprocessed).
    candidates : pd.DataFrame
        Output from select_biomarker_candidates.
    pairing : dict
        Pairing result from pair_skin_plasma_samples.
    feature_ids : list[int]
        Feature identifiers matching columns of X_skin_paired.
    n_top : int
        Number of top candidates to display.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, max(6, n_top * 0.3)))

    # Get top candidates
    top_candidates = candidates.head(n_top)

    if top_candidates.empty:
        ax.text(
            0.5,
            0.5,
            "No candidates to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return ax

    # Extract feature indices
    feature_id_to_idx = {fid: i for i, fid in enumerate(feature_ids)}
    candidate_indices = [feature_id_to_idx[fid] for fid in top_candidates["feature_id"]]

    # Extract data for top candidates
    heatmap_data = X_skin_paired[:, candidate_indices].T

    # Create column labels (subject_timepoint)
    pairing_df = pairing["pairing_df"]
    col_labels = [
        f"{row['subject']}_{int(row['timepoint'])}" for _, row in pairing_df.iterrows()
    ]

    # Row labels (feature IDs)
    row_labels = [str(fid) for fid in top_candidates["feature_id"]]

    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, ha="right", fontsize=6)
    ax.set_xlabel("Sample (subject_timepoint)")
    ax.set_ylabel("Feature ID")
    ax.set_title(f"Top {n_top} Biomarker Candidates (Pareto-scaled)")
    plt.colorbar(im, ax=ax, label="Scaled intensity")

    return ax


def plot_biomarker_summary(
    discovery_result: dict,
    location_name: str = "",
) -> Figure:
    """Plot 2x2 multi-panel summary of biomarker discovery results.

    Panels:
    1. Volcano plot (correlation screening)
    2. Predicted vs observed (PLS regression)
    3. Top VIP scores
    4. Candidate heatmap

    Parameters
    ----------
    discovery_result : dict
        Output from run_biomarker_discovery.
    location_name : str
        Location name for plot titles (e.g., "forearm", "forehead").

    Returns
    -------
    fig : Figure
        Matplotlib figure with 4 subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Volcano plot
    ax = axes[0, 0]
    plot_correlation_volcano(
        discovery_result["correlation"],
        ax=ax,
    )

    # Panel 2: Predicted vs observed
    ax = axes[0, 1]
    plot_predicted_vs_observed(
        discovery_result["pls_result"],
        location_name=location_name,
        ax=ax,
    )

    # Panel 3: Top VIP scores (horizontal bar chart)
    ax = axes[1, 0]
    from src.multivariate import plot_vip_scores

    vip = discovery_result["pls_result"]["vip"]
    feature_ids = discovery_result["candidates"]["feature_id"].tolist()
    if len(feature_ids) == 0:
        # Fallback: use all features from correlation results
        feature_ids = discovery_result["correlation"]["feature_id"].tolist()

    plot_vip_scores(vip, feature_ids, n_top=20, ax=ax)
    ax.set_title("VIP Scores (PLS Regression)")

    # Panel 4: Candidate heatmap (placeholder if no paired data)
    ax = axes[1, 1]
    if "X_skin_paired" in discovery_result:
        plot_candidate_heatmap(
            discovery_result["X_skin_paired"],
            discovery_result["candidates"],
            discovery_result["pairing"],
            feature_ids,
            n_top=15,
            ax=ax,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Candidate heatmap requires X_skin_paired in result",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")

    suptitle = "Biomarker Discovery Summary"
    if location_name:
        suptitle = f"{location_name.capitalize()} — {suptitle}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig
