"""Location comparison: forearm vs forehead for drug monitoring.

Compares two sampling locations by matching features across separate GNPS datasets
using m/z + RT tolerance windows, then evaluates drug detection, PK parameters,
and PLS-DA classification performance to recommend the optimal location.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.drug_detection import (
    compare_plasma_skin,
    extract_pk_curves,
    plot_pk_curves,
    search_features_by_mz,
    summarize_pk_all_subjects,
)
from src.multivariate import assign_time_class, perform_plsda, plot_plsda_scores

logger = logging.getLogger(__name__)

DEFAULT_MZ_TOLERANCE_PPM = 10.0
DEFAULT_RT_TOLERANCE_MIN = 0.5
LOCATION_COLORS: dict[str, str] = {"forearm": "tab:blue", "forehead": "tab:orange"}


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def match_features_across_locations(
    feature_meta_a: pd.DataFrame,
    feature_meta_b: pd.DataFrame,
    mz_tolerance_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
    rt_tolerance_min: float = DEFAULT_RT_TOLERANCE_MIN,
) -> pd.DataFrame:
    """Match features between two datasets by m/z (ppm) and RT (absolute minutes).

    Uses numpy broadcasting for vectorized pairwise comparison. For each feature
    in dataset A, finds the closest match in dataset B within the tolerance window.
    Only the best match (smallest combined normalised error) is kept per feature.

    Parameters
    ----------
    feature_meta_a : pd.DataFrame
        Feature metadata for dataset A with ``row m/z`` and ``row retention time``.
    feature_meta_b : pd.DataFrame
        Feature metadata for dataset B with ``row m/z`` and ``row retention time``.
    mz_tolerance_ppm : float
        Maximum allowed mass error in parts per million.
    rt_tolerance_min : float
        Maximum allowed retention-time difference in minutes.

    Returns
    -------
    matches : pd.DataFrame
        Columns: feature_id_a, feature_id_b, mz_a, mz_b, rt_a, rt_b,
        mz_ppm_error, rt_diff_min.  Empty DataFrame if no matches found.
    """
    mz_a = feature_meta_a["row m/z"].values
    mz_b = feature_meta_b["row m/z"].values
    rt_a = feature_meta_a["row retention time"].values
    rt_b = feature_meta_b["row retention time"].values
    ids_a = feature_meta_a.index.values
    ids_b = feature_meta_b.index.values

    # Vectorized pairwise ppm error: |mz_a_i - mz_b_j| / mz_a_i * 1e6
    ppm_error = np.abs(mz_a[:, None] - mz_b[None, :]) / mz_a[:, None] * 1e6
    rt_diff = np.abs(rt_a[:, None] - rt_b[None, :])

    within_tol = (ppm_error <= mz_tolerance_ppm) & (rt_diff <= rt_tolerance_min)

    rows: list[dict] = []
    for i in range(len(mz_a)):
        candidates = np.where(within_tol[i])[0]
        if len(candidates) == 0:
            continue
        # Pick best match: smallest combined normalised error
        combined = (
            ppm_error[i, candidates] / mz_tolerance_ppm
            + rt_diff[i, candidates] / rt_tolerance_min
        )
        best_j = candidates[np.argmin(combined)]
        rows.append(
            {
                "feature_id_a": ids_a[i],
                "feature_id_b": ids_b[best_j],
                "mz_a": mz_a[i],
                "mz_b": mz_b[best_j],
                "rt_a": rt_a[i],
                "rt_b": rt_b[best_j],
                "mz_ppm_error": ppm_error[i, best_j],
                "rt_diff_min": rt_diff[i, best_j],
            }
        )

    matches = pd.DataFrame(rows)
    logger.info(
        "Feature matching: %d A x %d B -> %d matches (ppm<=%.1f, RT<=%.2f min)",
        len(mz_a),
        len(mz_b),
        len(matches),
        mz_tolerance_ppm,
        rt_tolerance_min,
    )
    return matches


def compute_feature_overlap(
    feature_meta_a: pd.DataFrame,
    feature_meta_b: pd.DataFrame,
    mz_tolerance_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
    rt_tolerance_min: float = DEFAULT_RT_TOLERANCE_MIN,
) -> dict:
    """Compute feature overlap statistics between two datasets.

    Parameters
    ----------
    feature_meta_a : pd.DataFrame
        Feature metadata for dataset A.
    feature_meta_b : pd.DataFrame
        Feature metadata for dataset B.
    mz_tolerance_ppm : float
        Maximum allowed mass error in parts per million.
    rt_tolerance_min : float
        Maximum allowed retention-time difference in minutes.

    Returns
    -------
    overlap : dict
        Keys: n_a, n_b, n_matched_a, n_matched_b, n_unique_a, n_unique_b,
        overlap_fraction_a, overlap_fraction_b, matches (DataFrame).
    """
    matches = match_features_across_locations(
        feature_meta_a, feature_meta_b, mz_tolerance_ppm, rt_tolerance_min
    )

    n_a = len(feature_meta_a)
    n_b = len(feature_meta_b)
    n_matched_a = matches["feature_id_a"].nunique() if not matches.empty else 0
    n_matched_b = matches["feature_id_b"].nunique() if not matches.empty else 0

    overlap = {
        "n_a": n_a,
        "n_b": n_b,
        "n_matched_a": n_matched_a,
        "n_matched_b": n_matched_b,
        "n_unique_a": n_a - n_matched_a,
        "n_unique_b": n_b - n_matched_b,
        "overlap_fraction_a": n_matched_a / n_a if n_a > 0 else 0.0,
        "overlap_fraction_b": n_matched_b / n_b if n_b > 0 else 0.0,
        "matches": matches,
    }

    logger.info(
        "Feature overlap: A %d/%d (%.0f%%), B %d/%d (%.0f%%)",
        n_matched_a,
        n_a,
        overlap["overlap_fraction_a"] * 100,
        n_matched_b,
        n_b,
        overlap["overlap_fraction_b"] * 100,
    )
    return overlap


def compare_drug_detection(
    feat_meta_a: pd.DataFrame,
    feat_meta_b: pd.DataFrame,
    peak_areas_a: pd.DataFrame,
    peak_areas_b: pd.DataFrame,
    metadata_a: pd.DataFrame,
    metadata_b: pd.DataFrame,
    target_compounds: dict[str, float],
    tolerance_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
) -> pd.DataFrame:
    """Compare drug detection across two sampling locations.

    Searches for each target compound in both datasets and reports detection
    status and mean skin intensity.

    Parameters
    ----------
    feat_meta_a : pd.DataFrame
        Feature metadata for location A.
    feat_meta_b : pd.DataFrame
        Feature metadata for location B.
    peak_areas_a : pd.DataFrame
        Raw peak area matrix for location A (features x samples).
    peak_areas_b : pd.DataFrame
        Raw peak area matrix for location B (features x samples).
    metadata_a : pd.DataFrame
        Sample metadata for location A.
    metadata_b : pd.DataFrame
        Sample metadata for location B.
    target_compounds : dict[str, float]
        Compound name -> target m/z mapping.
    tolerance_ppm : float
        Mass tolerance in ppm for feature search.

    Returns
    -------
    comparison : pd.DataFrame
        One row per compound with columns: compound, location,
        n_matches, best_ppm_error, mean_skin_intensity, detected.
    """
    rows: list[dict] = []

    for compound, target_mz in target_compounds.items():
        for label, feat_meta, peak_areas, metadata in [
            ("forearm", feat_meta_a, peak_areas_a, metadata_a),
            ("forehead", feat_meta_b, peak_areas_b, metadata_b),
        ]:
            matches = search_features_by_mz(feat_meta, target_mz, tolerance_ppm)
            n_matches = len(matches)
            best_ppm = (
                float(matches["ppm_error"].min()) if n_matches > 0 else float("nan")
            )

            # Mean skin intensity for the best-matching feature
            mean_intensity = 0.0
            if n_matches > 0:
                best_id = matches["ppm_error"].idxmin()
                skin_mask = metadata["sample_type"].values == "skin"
                skin_filenames = metadata.loc[skin_mask, "filename"].values
                skin_cols = [c for c in skin_filenames if c in peak_areas.columns]
                if best_id in peak_areas.index and len(skin_cols) > 0:
                    mean_intensity = float(peak_areas.loc[best_id, skin_cols].mean())

            rows.append(
                {
                    "compound": compound,
                    "location": label,
                    "n_matches": n_matches,
                    "best_ppm_error": best_ppm,
                    "mean_skin_intensity": mean_intensity,
                    "detected": n_matches > 0,
                }
            )

    comparison = pd.DataFrame(rows)
    logger.info(
        "Drug detection comparison: %d compounds x 2 locations",
        len(target_compounds),
    )
    return comparison


def compare_pk_parameters(
    peak_areas_a: pd.DataFrame,
    peak_areas_b: pd.DataFrame,
    metadata_a: pd.DataFrame,
    metadata_b: pd.DataFrame,
    feat_meta_a: pd.DataFrame,
    feat_meta_b: pd.DataFrame,
    target_mz: float,
    tolerance_ppm: float = DEFAULT_MZ_TOLERANCE_PPM,
) -> dict:
    """Compare pharmacokinetic parameters for a compound between two locations.

    Chains search -> extract PK -> summarize -> compare for each location.

    Parameters
    ----------
    peak_areas_a : pd.DataFrame
        Raw peak area matrix for location A.
    peak_areas_b : pd.DataFrame
        Raw peak area matrix for location B.
    metadata_a : pd.DataFrame
        Sample metadata for location A.
    metadata_b : pd.DataFrame
        Sample metadata for location B.
    feat_meta_a : pd.DataFrame
        Feature metadata for location A.
    feat_meta_b : pd.DataFrame
        Feature metadata for location B.
    target_mz : float
        Target m/z value for the compound.
    tolerance_ppm : float
        Mass tolerance in ppm.

    Returns
    -------
    result : dict
        Keys: location_a (dict with pk_curves, pk_summary, comparison),
        location_b (same), summary (DataFrame comparing key PK metrics).
    """
    location_results: dict[str, dict] = {}

    for label, feat_meta, peak_areas, metadata in [
        ("forearm", feat_meta_a, peak_areas_a, metadata_a),
        ("forehead", feat_meta_b, peak_areas_b, metadata_b),
    ]:
        matches = search_features_by_mz(feat_meta, target_mz, tolerance_ppm)
        loc_result: dict = {
            "pk_curves": pd.DataFrame(),
            "pk_summary": pd.DataFrame(),
            "comparison": {},
        }

        if not matches.empty:
            best_id = int(matches["ppm_error"].idxmin())
            pk_curves = extract_pk_curves(peak_areas, metadata, best_id)
            if not pk_curves.empty:
                pk_summary = summarize_pk_all_subjects(pk_curves)
                comparison = compare_plasma_skin(pk_summary)
                loc_result = {
                    "pk_curves": pk_curves,
                    "pk_summary": pk_summary,
                    "comparison": comparison,
                }

        location_results[label] = loc_result

    # Build summary DataFrame
    summary_rows: list[dict] = []
    for label, loc_data in location_results.items():
        comp = loc_data.get("comparison", {})
        cmax_r = comp.get("cmax_corr", {}).get("r", float("nan"))
        auc_r = comp.get("auc_corr", {}).get("r", float("nan"))
        mean_lag = comp.get("tmax_lag", {}).get("mean_lag", float("nan"))
        summary_rows.append(
            {
                "location": label,
                "cmax_corr_r": cmax_r,
                "auc_corr_r": auc_r,
                "mean_tmax_lag": mean_lag,
            }
        )

    return {
        "location_a": location_results["forearm"],
        "location_b": location_results["forehead"],
        "summary": pd.DataFrame(summary_rows),
    }


def compare_plsda_performance(
    X_skin_a: np.ndarray,
    X_skin_b: np.ndarray,
    metadata_skin_a: pd.DataFrame,
    metadata_skin_b: pd.DataFrame,
    time_threshold: float = 60.0,
    n_components: int = 2,
    cv: int = 7,
) -> dict:
    """Compare PLS-DA classification performance between two sampling locations.

    Fits PLS-DA on skin-only samples at each location using early/late time class
    labels, and returns performance metrics for comparison.

    Parameters
    ----------
    X_skin_a : np.ndarray
        Preprocessed skin-only data for location A (samples x features).
    X_skin_b : np.ndarray
        Preprocessed skin-only data for location B (samples x features).
    metadata_skin_a : pd.DataFrame
        Skin-only metadata for location A with ``timepoint`` column.
    metadata_skin_b : pd.DataFrame
        Skin-only metadata for location B with ``timepoint`` column.
    time_threshold : float
        Timepoint threshold for early/late classification.
    n_components : int
        Number of PLS components.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    result : dict
        Keys: plsda_a, plsda_b (perform_plsda outputs), labels_a, labels_b,
        summary (DataFrame with location, r2y, q2, q2_std).
    """
    labels_a = assign_time_class(metadata_skin_a, time_threshold)
    labels_b = assign_time_class(metadata_skin_b, time_threshold)

    plsda_a = perform_plsda(X_skin_a, labels_a, n_components=n_components, cv=cv)
    plsda_b = perform_plsda(X_skin_b, labels_b, n_components=n_components, cv=cv)

    summary = pd.DataFrame(
        [
            {
                "location": "forearm",
                "r2y": plsda_a["r2y"],
                "q2": plsda_a["q2"],
                "q2_std": plsda_a["q2_std"],
            },
            {
                "location": "forehead",
                "r2y": plsda_b["r2y"],
                "q2": plsda_b["q2"],
                "q2_std": plsda_b["q2_std"],
            },
        ]
    )

    logger.info(
        "PLS-DA comparison: forearm Q2=%.3f, forehead Q2=%.3f",
        plsda_a["q2"],
        plsda_b["q2"],
    )

    return {
        "plsda_a": plsda_a,
        "plsda_b": plsda_b,
        "labels_a": labels_a,
        "labels_b": labels_b,
        "summary": summary,
    }


def generate_recommendation(
    feature_overlap: dict,
    detection_comparison: pd.DataFrame,
    pk_comparison: dict,
    plsda_comparison: dict,
) -> dict:
    """Score sampling locations and generate a recommendation.

    Evaluates four criteria (0-1 each): feature coverage, detection quality,
    PK plasma-skin correlation, and PLS-DA cross-validated Q2.

    Parameters
    ----------
    feature_overlap : dict
        Output from ``compute_feature_overlap``.
    detection_comparison : pd.DataFrame
        Output from ``compare_drug_detection``.
    pk_comparison : dict
        Output from ``compare_pk_parameters``.
    plsda_comparison : dict
        Output from ``compare_plsda_performance``.

    Returns
    -------
    recommendation : dict
        Keys: scores (DataFrame), recommended (str), rationale (str),
        raw_metrics (dict).
    """
    scores: dict[str, dict[str, float]] = {"forearm": {}, "forehead": {}}

    # 1. Feature coverage: fraction of features that overlap with the other dataset
    scores["forearm"]["feature_coverage"] = feature_overlap["overlap_fraction_a"]
    scores["forehead"]["feature_coverage"] = feature_overlap["overlap_fraction_b"]

    # 2. Detection quality: fraction of compounds detected + mean ppm quality
    for loc_label in ["forearm", "forehead"]:
        loc_df = detection_comparison[detection_comparison["location"] == loc_label]
        n_detected = int(loc_df["detected"].sum())
        n_total = len(loc_df)
        scores[loc_label]["detection_quality"] = (
            n_detected / n_total if n_total > 0 else 0.0
        )

    # 3. PK correlation: absolute Cmax Spearman r (NaN -> 0)
    pk_summary = pk_comparison["summary"]
    for _, row in pk_summary.iterrows():
        loc = row["location"]
        r_val = row["cmax_corr_r"]
        scores[loc]["pk_correlation"] = abs(r_val) if np.isfinite(r_val) else 0.0

    # 4. Model Q2: clamp to [0, 1]
    plsda_summary = plsda_comparison["summary"]
    for _, row in plsda_summary.iterrows():
        loc = row["location"]
        q2 = row["q2"]
        scores[loc]["model_q2"] = max(0.0, min(1.0, q2)) if np.isfinite(q2) else 0.0

    # Build summary DataFrame
    criteria = ["feature_coverage", "detection_quality", "pk_correlation", "model_q2"]
    score_rows = []
    for loc in ["forearm", "forehead"]:
        row_data = {"location": loc}
        row_data.update(scores[loc])
        row_data["total"] = sum(scores[loc][c] for c in criteria)
        score_rows.append(row_data)
    scores_df = pd.DataFrame(score_rows)

    # Determine recommendation
    forearm_total = scores_df.loc[scores_df["location"] == "forearm", "total"].values[0]
    forehead_total = scores_df.loc[scores_df["location"] == "forehead", "total"].values[
        0
    ]

    if forearm_total > forehead_total:
        recommended = "forearm"
    elif forehead_total > forearm_total:
        recommended = "forehead"
    else:
        recommended = "forearm"  # tie-break: forearm (more established in literature)

    # Generate rationale
    other = "forehead" if recommended == "forearm" else "forearm"
    advantages = [c for c in criteria if scores[recommended][c] >= scores[other][c]]
    rationale = (
        f"The {recommended} is recommended based on superior performance in "
        f"{len(advantages)}/4 criteria: {', '.join(advantages)}. "
        f"Total score: {max(forearm_total, forehead_total):.2f} vs "
        f"{min(forearm_total, forehead_total):.2f}."
    )

    logger.info(
        "Recommendation: %s (%.2f vs %.2f)", recommended, forearm_total, forehead_total
    )

    return {
        "scores": scores_df,
        "recommended": recommended,
        "rationale": rationale,
        "raw_metrics": scores,
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_feature_overlap(
    overlap: dict,
    label_a: str = "Forearm",
    label_b: str = "Forehead",
    ax: Axes | None = None,
) -> Axes:
    """Plot feature overlap as a stacked bar: unique-A / shared / unique-B.

    Parameters
    ----------
    overlap : dict
        Output from ``compute_feature_overlap``.
    label_a : str
        Display label for dataset A.
    label_b : str
        Display label for dataset B.
    ax : Axes or None
        Matplotlib axes. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    unique_a = overlap["n_unique_a"]
    shared = overlap["n_matched_a"]  # matched features from A's perspective
    unique_b = overlap["n_unique_b"]

    categories = [f"Unique {label_a}", "Shared", f"Unique {label_b}"]
    values = [unique_a, shared, unique_b]
    colors = [LOCATION_COLORS["forearm"], "tab:green", LOCATION_COLORS["forehead"]]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            str(val),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("Number of features")
    ax.set_title("Feature Overlap Between Sampling Locations")
    ax.grid(axis="y", alpha=0.3)

    return ax


def plot_detection_comparison(
    detection_df: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    """Plot grouped bars comparing skin intensity per compound per location.

    Parameters
    ----------
    detection_df : pd.DataFrame
        Output from ``compare_drug_detection``.
    ax : Axes or None
        Matplotlib axes. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    compounds = detection_df["compound"].unique()
    locations = detection_df["location"].unique()
    x = np.arange(len(compounds))
    width = 0.35

    for i, loc in enumerate(locations):
        loc_data = detection_df[detection_df["location"] == loc]
        intensities = [
            float(
                loc_data.loc[loc_data["compound"] == c, "mean_skin_intensity"].values[0]
            )
            if len(loc_data.loc[loc_data["compound"] == c]) > 0
            else 0.0
            for c in compounds
        ]
        color = LOCATION_COLORS.get(loc, "tab:grey")
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            intensities,
            width,
            label=loc.capitalize(),
            color=color,
            alpha=0.8,
        )

        # Mark non-detected as 'ND'
        for j, c in enumerate(compounds):
            detected = detection_df.loc[
                (detection_df["compound"] == c) & (detection_df["location"] == loc),
                "detected",
            ].values
            if len(detected) > 0 and not detected[0]:
                ax.text(
                    x[j] + offset,
                    max(intensities) * 0.01 if max(intensities) > 0 else 0.1,
                    "ND",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(compounds, rotation=15, ha="right")
    ax.set_ylabel("Mean skin intensity (raw peak area)")
    ax.set_title("Drug Detection: Forearm vs Forehead")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    return ax


def plot_pk_comparison_locations(
    pk_comparison: dict,
    compound_name: str = "Diphenhydramine",
) -> Figure:
    """Plot PK curves side by side for two locations.

    Parameters
    ----------
    pk_comparison : dict
        Output from ``compare_pk_parameters``.
    compound_name : str
        Compound name for plot title.

    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (label, loc_key) in zip(
        axes, [("Forearm", "location_a"), ("Forehead", "location_b")]
    ):
        pk_curves = pk_comparison[loc_key]["pk_curves"]
        if pk_curves.empty:
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=14
            )
            ax.set_title(f"{label} — {compound_name}")
        else:
            plot_pk_curves(pk_curves, compound_name=f"{label} — {compound_name}", ax=ax)

    fig.suptitle(
        f"{compound_name} — PK Curves by Sampling Location",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_model_comparison(
    plsda_comparison: dict,
    label_a: str = "Forearm",
    label_b: str = "Forehead",
) -> Figure:
    """Plot 3-panel model comparison: R2Y/Q2 bars + score plots per location.

    Parameters
    ----------
    plsda_comparison : dict
        Output from ``compare_plsda_performance``.
    label_a : str
        Display label for location A.
    label_b : str
        Display label for location B.

    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    summary = plsda_comparison["summary"]

    # Panel 1: R2Y and Q2 grouped bars
    ax = axes[0]
    x = np.arange(2)
    width = 0.3
    r2y_vals = summary["r2y"].values
    q2_vals = summary["q2"].values
    q2_std_vals = summary["q2_std"].values

    ax.bar(
        x - width / 2, r2y_vals, width, label="R\u00b2Y", color="steelblue", alpha=0.8
    )
    ax.bar(
        x + width / 2,
        q2_vals,
        width,
        yerr=q2_std_vals,
        label="Q\u00b2",
        color="darkorange",
        alpha=0.8,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Score")
    ax.set_title("PLS-DA Performance")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Forearm score plot
    plot_plsda_scores(
        plsda_comparison["plsda_a"], plsda_comparison["labels_a"], ax=axes[1]
    )
    axes[1].set_title(f"{label_a} PLS-DA Scores")

    # Panel 3: Forehead score plot
    plot_plsda_scores(
        plsda_comparison["plsda_b"], plsda_comparison["labels_b"], ax=axes[2]
    )
    axes[2].set_title(f"{label_b} PLS-DA Scores")

    fig.suptitle("PLS-DA Model Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_recommendation_summary(
    recommendation: dict,
    ax: Axes | None = None,
) -> Axes:
    """Plot a color-coded scorecard table summarizing the recommendation.

    Parameters
    ----------
    recommendation : dict
        Output from ``generate_recommendation``.
    ax : Axes or None
        Matplotlib axes. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    scores_df = recommendation["scores"]
    criteria = [
        "feature_coverage",
        "detection_quality",
        "pk_correlation",
        "model_q2",
        "total",
    ]
    display_labels = [
        "Feature Coverage",
        "Detection Quality",
        "PK Correlation",
        "Model Q\u00b2",
        "Total",
    ]

    cell_text = []
    cell_colors = []
    for _, row in scores_df.iterrows():
        row_text = []
        row_colors = []
        for c in criteria:
            val = row[c]
            row_text.append(f"{val:.3f}")
            row_colors.append("white")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    # Highlight winner per criterion
    for j, c in enumerate(criteria):
        vals = [scores_df.iloc[i][c] for i in range(len(scores_df))]
        winner_idx = int(np.argmax(vals))
        cell_colors[winner_idx][j] = "#c6efce"  # light green

    table = ax.table(
        cellText=cell_text,
        rowLabels=[r.capitalize() for r in scores_df["location"]],
        colLabels=display_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    ax.set_title(
        f"Recommendation: {recommendation['recommended'].capitalize()}",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    return ax
