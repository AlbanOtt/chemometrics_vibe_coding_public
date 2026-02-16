"""Targeted compound detection and pharmacokinetic analysis for LC-MS metabolomics.

Implements m/z-based feature search, pharmacokinetic curve extraction,
PK parameter calculation, and plasma-skin comparison for diphenhydramine
and its metabolites.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_PPM_TOLERANCE = 10.0

TARGET_COMPOUNDS: dict[str, float] = {
    "Diphenhydramine": 256.1696,
    "N-desmethyl-DPH": 242.1539,
    "DPH N-oxide": 272.1645,
}


def search_features_by_mz(
    feature_metadata: pd.DataFrame,
    target_mz: float,
    tolerance_ppm: float = DEFAULT_PPM_TOLERANCE,
) -> pd.DataFrame:
    """Search for features matching a target m/z within a ppm tolerance window.

    Parameters
    ----------
    feature_metadata : pd.DataFrame
        Feature metadata with a ``row m/z`` column. Index is row ID.
    target_mz : float
        Target m/z value to search for ([M+H]+ ion).
    tolerance_ppm : float
        Maximum allowed mass error in parts per million.

    Returns
    -------
    matches : pd.DataFrame
        Subset of feature_metadata for matching features, with an added
        ``ppm_error`` column. Empty DataFrame if no matches found.
    """
    mz_values = feature_metadata["row m/z"]
    ppm_error = np.abs(mz_values - target_mz) / target_mz * 1e6

    mask = ppm_error <= tolerance_ppm
    matches = feature_metadata.loc[mask].copy()
    matches["ppm_error"] = ppm_error[mask]

    logger.info(
        "m/z search: %.4f +/- %.1f ppm -> %d match(es)",
        target_mz,
        tolerance_ppm,
        len(matches),
    )
    return matches


def extract_pk_curves(
    peak_areas_raw: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_id: int,
) -> pd.DataFrame:
    """Extract pharmacokinetic curves for a single feature across all subjects.

    Uses raw (non-scaled) peak areas to preserve absolute intensity relationships
    needed for pharmacokinetic parameter calculation.

    Parameters
    ----------
    peak_areas_raw : pd.DataFrame
        Raw peak area matrix (features x samples). Index is row ID,
        columns are sample filenames.
    metadata : pd.DataFrame
        Sample metadata with columns: filename, subject, timepoint, sample_type.
    feature_id : int
        Row ID of the feature to extract.

    Returns
    -------
    pk_curves : pd.DataFrame
        DataFrame with columns: subject, timepoint, sample_type, intensity.
        Sorted by subject, sample_type, and timepoint. Empty if feature not found.
    """
    if feature_id not in peak_areas_raw.index:
        logger.warning("Feature %d not found in peak area table", feature_id)
        return pd.DataFrame(
            columns=["subject", "timepoint", "sample_type", "intensity"]
        )

    intensities = peak_areas_raw.loc[feature_id]
    intensity_df = pd.DataFrame(
        {"filename": intensities.index, "intensity": intensities.values}
    )

    pk_curves = metadata[["filename", "subject", "timepoint", "sample_type"]].merge(
        intensity_df, on="filename", how="inner"
    )
    pk_curves = pk_curves[["subject", "timepoint", "sample_type", "intensity"]]
    pk_curves = pk_curves.sort_values(
        ["subject", "sample_type", "timepoint"]
    ).reset_index(drop=True)

    logger.info(
        "Extracted PK curves for feature %d: %d data points, %d subjects",
        feature_id,
        len(pk_curves),
        pk_curves["subject"].nunique(),
    )
    return pk_curves


def calculate_pk_parameters(
    timepoints: np.ndarray,
    intensities: np.ndarray,
) -> dict[str, float]:
    """Calculate pharmacokinetic parameters from a single concentration-time curve.

    Parameters
    ----------
    timepoints : np.ndarray
        Time values (e.g., minutes post-dose).
    intensities : np.ndarray
        Peak area intensities at each timepoint.

    Returns
    -------
    params : dict[str, float]
        Dictionary with keys: Cmax, Tmax, AUC.
        Returns zeros for all parameters if input is empty or all-zero.
    """
    if len(timepoints) == 0 or len(intensities) == 0 or np.all(intensities == 0):
        return {"Cmax": 0.0, "Tmax": 0.0, "AUC": 0.0}

    sort_idx = np.argsort(timepoints)
    t_sorted = timepoints[sort_idx]
    i_sorted = intensities[sort_idx]

    cmax = float(np.max(i_sorted))
    tmax = float(t_sorted[np.argmax(i_sorted)])
    auc = float(np.trapezoid(i_sorted, t_sorted))

    return {"Cmax": cmax, "Tmax": tmax, "AUC": auc}


def summarize_pk_all_subjects(pk_curves: pd.DataFrame) -> pd.DataFrame:
    """Compute PK parameters (Cmax, Tmax, AUC) for each subject and sample type.

    Parameters
    ----------
    pk_curves : pd.DataFrame
        Output from ``extract_pk_curves`` with columns:
        subject, timepoint, sample_type, intensity.

    Returns
    -------
    summary : pd.DataFrame
        One row per subject-sample_type combination with columns:
        subject, sample_type, Cmax, Tmax, AUC.
    """
    rows: list[dict] = []
    for (subject, sample_type), group in pk_curves.groupby(["subject", "sample_type"]):
        params = calculate_pk_parameters(
            group["timepoint"].values,
            group["intensity"].values,
        )
        rows.append(
            {
                "subject": subject,
                "sample_type": sample_type,
                "Cmax": params["Cmax"],
                "Tmax": params["Tmax"],
                "AUC": params["AUC"],
            }
        )

    summary = pd.DataFrame(rows)
    logger.info("PK summary: %d subject-type combinations", len(summary))
    return summary


def compare_plasma_skin(pk_summary: pd.DataFrame) -> dict:
    """Compare pharmacokinetic parameters between plasma and skin samples.

    Computes Spearman correlations for Cmax and AUC between plasma and skin,
    and analyzes the time lag (skin Tmax - plasma Tmax) per subject.

    Parameters
    ----------
    pk_summary : pd.DataFrame
        Output from ``summarize_pk_all_subjects`` with columns:
        subject, sample_type, Cmax, Tmax, AUC.

    Returns
    -------
    comparison : dict
        Dictionary with keys:
        - cmax_corr: dict with r, pvalue, n
        - auc_corr: dict with r, pvalue, n
        - tmax_lag: dict with mean_lag, std_lag, per_subject (dict subject->lag)
        - paired_data: pd.DataFrame with merged plasma/skin values per subject
    """
    plasma = pk_summary[pk_summary["sample_type"] == "plasma"].set_index("subject")
    skin = pk_summary[pk_summary["sample_type"] == "skin"].set_index("subject")

    paired = plasma[["Cmax", "Tmax", "AUC"]].join(
        skin[["Cmax", "Tmax", "AUC"]],
        lsuffix="_plasma",
        rsuffix="_skin",
        how="inner",
    )

    n_paired = len(paired)

    if n_paired >= 3:
        cmax_r, cmax_p = stats.spearmanr(paired["Cmax_plasma"], paired["Cmax_skin"])
        auc_r, auc_p = stats.spearmanr(paired["AUC_plasma"], paired["AUC_skin"])
    else:
        cmax_r, cmax_p = float("nan"), float("nan")
        auc_r, auc_p = float("nan"), float("nan")
        logger.warning("Too few paired subjects (%d) for correlation", n_paired)

    tmax_lag = paired["Tmax_skin"] - paired["Tmax_plasma"]

    comparison = {
        "cmax_corr": {"r": float(cmax_r), "pvalue": float(cmax_p), "n": n_paired},
        "auc_corr": {"r": float(auc_r), "pvalue": float(auc_p), "n": n_paired},
        "tmax_lag": {
            "mean_lag": float(tmax_lag.mean()),
            "std_lag": float(tmax_lag.std()),
            "per_subject": tmax_lag.to_dict(),
        },
        "paired_data": paired.reset_index(),
    }

    logger.info(
        "Plasma-skin comparison: n=%d, Cmax rho=%.3f (p=%.3f), AUC rho=%.3f (p=%.3f)",
        n_paired,
        cmax_r,
        cmax_p,
        auc_r,
        auc_p,
    )
    return comparison


def plot_pk_curves(
    pk_curves: pd.DataFrame,
    compound_name: str = "",
    ax: Axes | None = None,
) -> Axes:
    """Plot pharmacokinetic curves: intensity vs timepoint per subject and sample type.

    Parameters
    ----------
    pk_curves : pd.DataFrame
        Output from ``extract_pk_curves`` with columns:
        subject, timepoint, sample_type, intensity.
    compound_name : str
        Compound name for the plot title.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    sample_type_colors = {"plasma": "tab:blue", "skin": "tab:orange"}
    sample_type_linestyles = {"plasma": "-", "skin": "--"}

    subjects = sorted(pk_curves["subject"].unique())
    sample_types = sorted(pk_curves["sample_type"].unique())

    for i, subject in enumerate(subjects):
        marker = markers[i % len(markers)]
        for sample_type in sample_types:
            mask = (pk_curves["subject"] == subject) & (
                pk_curves["sample_type"] == sample_type
            )
            subset = pk_curves.loc[mask].sort_values("timepoint")
            if subset.empty:
                continue
            color = sample_type_colors.get(sample_type, "tab:grey")
            linestyle = sample_type_linestyles.get(sample_type, "-")
            ax.plot(
                subset["timepoint"],
                subset["intensity"],
                marker=marker,
                linestyle=linestyle,
                color=color,
                alpha=0.7,
                label=f"{subject} ({sample_type})",
                markersize=5,
            )

    ax.set_xlabel("Timepoint (min)")
    ax.set_ylabel("Peak area (raw)")
    title = "Pharmacokinetic Curves"
    if compound_name:
        title = f"{compound_name} — {title}"
    ax.set_title(title)
    ax.legend(fontsize=6, loc="best", ncol=2)
    ax.grid(alpha=0.3)

    return ax


def plot_pk_comparison(
    pk_summary: pd.DataFrame,
    comparison: dict,
    compound_name: str = "",
) -> Figure:
    """Plot plasma vs skin PK parameter comparison as a 3-panel figure.

    Parameters
    ----------
    pk_summary : pd.DataFrame
        Output from ``summarize_pk_all_subjects``.
    comparison : dict
        Output from ``compare_plasma_skin``.
    compound_name : str
        Compound name for the plot title.

    Returns
    -------
    fig : Figure
        Matplotlib figure with 3 subplots: Cmax scatter, AUC scatter, Tmax lag bar.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    paired = comparison["paired_data"]

    # Panel 1: Cmax plasma vs skin
    ax = axes[0]
    ax.scatter(
        paired["Cmax_plasma"], paired["Cmax_skin"], s=60, edgecolors="white", zorder=3
    )
    for _, row in paired.iterrows():
        ax.annotate(
            row["subject"],
            (row["Cmax_plasma"], row["Cmax_skin"]),
            fontsize=6,
            alpha=0.7,
        )
    r = comparison["cmax_corr"]["r"]
    p = comparison["cmax_corr"]["pvalue"]
    ax.set_xlabel("Cmax (plasma)")
    ax.set_ylabel("Cmax (skin)")
    ax.set_title(f"Cmax: Spearman r={r:.3f}, p={p:.3f}")
    ax.grid(alpha=0.3)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="grey", alpha=0.5, zorder=1)

    # Panel 2: AUC plasma vs skin
    ax = axes[1]
    ax.scatter(
        paired["AUC_plasma"], paired["AUC_skin"], s=60, edgecolors="white", zorder=3
    )
    for _, row in paired.iterrows():
        ax.annotate(
            row["subject"], (row["AUC_plasma"], row["AUC_skin"]), fontsize=6, alpha=0.7
        )
    r = comparison["auc_corr"]["r"]
    p = comparison["auc_corr"]["pvalue"]
    ax.set_xlabel("AUC (plasma)")
    ax.set_ylabel("AUC (skin)")
    ax.set_title(f"AUC: Spearman r={r:.3f}, p={p:.3f}")
    ax.grid(alpha=0.3)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="grey", alpha=0.5, zorder=1)

    # Panel 3: Tmax lag per subject
    ax = axes[2]
    lag_data = comparison["tmax_lag"]["per_subject"]
    subjects = sorted(lag_data.keys())
    lags = [lag_data[s] for s in subjects]
    ax.bar(range(len(subjects)), lags, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tmax lag (skin - plasma, min)")
    mean_lag = comparison["tmax_lag"]["mean_lag"]
    std_lag = comparison["tmax_lag"]["std_lag"]
    ax.set_title(f"Tmax lag: {mean_lag:.0f} +/- {std_lag:.0f} min")
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    suptitle = "Plasma vs Skin Comparison"
    if compound_name:
        suptitle = f"{compound_name} — {suptitle}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_compound_search_results(
    feature_metadata: pd.DataFrame,
    matches: pd.DataFrame,
    target_mz: float,
    compound_name: str = "",
    ax: Axes | None = None,
) -> Axes:
    """Plot m/z vs retention time with matched features highlighted.

    Parameters
    ----------
    feature_metadata : pd.DataFrame
        Full feature metadata with ``row m/z`` and ``row retention time`` columns.
    matches : pd.DataFrame
        Matching features from ``search_features_by_mz`` (subset of feature_metadata).
    target_mz : float
        Target m/z value.
    compound_name : str
        Compound name for the plot title.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        feature_metadata["row retention time"],
        feature_metadata["row m/z"],
        c="lightgrey",
        s=5,
        alpha=0.3,
        label="All features",
        zorder=1,
    )

    delta_mz = target_mz * DEFAULT_PPM_TOLERANCE / 1e6
    ax.axhspan(
        target_mz - delta_mz,
        target_mz + delta_mz,
        color="red",
        alpha=0.08,
        label=f"+/- {DEFAULT_PPM_TOLERANCE:.0f} ppm",
        zorder=0,
    )
    ax.axhline(target_mz, color="red", linestyle="--", alpha=0.4, linewidth=0.8)

    if not matches.empty:
        ax.scatter(
            matches["row retention time"],
            matches["row m/z"],
            c="red",
            s=80,
            marker="*",
            edgecolors="darkred",
            linewidth=0.5,
            label=f"Matches (n={len(matches)})",
            zorder=3,
        )
        for row_id, row in matches.iterrows():
            ax.annotate(
                f"ID {row_id}\n{row['ppm_error']:.1f} ppm",
                (row["row retention time"], row["row m/z"]),
                fontsize=7,
                fontweight="bold",
                color="darkred",
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_xlabel("Retention time (min)")
    ax.set_ylabel("m/z")
    title = f"Feature Search: m/z {target_mz:.4f}"
    if compound_name:
        title = f"{compound_name} — {title}"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    return ax
