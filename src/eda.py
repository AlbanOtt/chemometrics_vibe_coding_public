"""Exploratory data analysis for metabolomics data.

Implements dataset summarization, intensity distributions, PCA quality control,
and Hotelling's T2 outlier detection for preprocessed LC-MS metabolomics data.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

DEFAULT_N_COMPONENTS = 10
T2_THRESHOLD_MULTIPLIER = 2


def dataset_summary(
    X: np.ndarray,
    metadata: pd.DataFrame,
    feature_metadata: pd.DataFrame,
) -> dict:
    """Compute a comprehensive summary of a preprocessed metabolomics dataset.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed data matrix (samples x features).
    metadata : pd.DataFrame
        Sample metadata with columns: subject, timepoint, sample_type.
    feature_metadata : pd.DataFrame
        Feature metadata with columns: 'row m/z', 'row retention time'.

    Returns
    -------
    summary : dict
        Dictionary with keys: n_samples, n_features, n_subjects, n_timepoints,
        sample_types, zero_count, zero_fraction, intensity_min, intensity_max,
        intensity_median, dynamic_range_log10, mz_range, rt_range.
    """
    n_samples, n_features = X.shape

    zero_count = int(np.sum(X == 0))
    zero_fraction = zero_count / X.size if X.size > 0 else 0.0

    abs_X = np.abs(X)
    intensity_min = float(np.min(abs_X))
    intensity_max = float(np.max(abs_X))
    intensity_median = float(np.median(abs_X))

    # Dynamic range from non-zero absolute values
    nonzero_abs = abs_X[abs_X > 0]
    if len(nonzero_abs) > 0:
        dynamic_range_log10 = float(np.log10(np.max(nonzero_abs) / np.min(nonzero_abs)))
    else:
        dynamic_range_log10 = 0.0

    # Feature metadata ranges
    mz_range = (
        float(feature_metadata["row m/z"].min()),
        float(feature_metadata["row m/z"].max()),
    )
    rt_range = (
        float(feature_metadata["row retention time"].min()),
        float(feature_metadata["row retention time"].max()),
    )

    summary = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_subjects": int(metadata["subject"].nunique()),
        "n_timepoints": int(metadata["timepoint"].dropna().nunique()),
        "sample_types": metadata["sample_type"].value_counts().to_dict(),
        "zero_count": zero_count,
        "zero_fraction": zero_fraction,
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "intensity_median": intensity_median,
        "dynamic_range_log10": dynamic_range_log10,
        "mz_range": mz_range,
        "rt_range": rt_range,
    }

    logger.info(
        "Dataset: %d samples x %d features, %d subjects, %d timepoints",
        n_samples,
        n_features,
        summary["n_subjects"],
        summary["n_timepoints"],
    )
    return summary


def plot_intensity_distribution(
    X: np.ndarray,
    metadata: pd.DataFrame,
    group_col: str = "sample_type",
    ax: Axes | None = None,
) -> Axes:
    """Plot total intensity per sample as box plots grouped by a metadata column.

    Uses absolute values because Pareto-scaled data contains negatives.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed data matrix (samples x features).
    metadata : pd.DataFrame
        Sample metadata. Must contain ``group_col``.
    group_col : str
        Column in metadata to group samples by.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    total_intensity = np.sum(np.abs(X), axis=1)
    plot_df = pd.DataFrame(
        {"total_intensity": total_intensity, group_col: metadata[group_col].values}
    )

    groups = sorted(plot_df[group_col].unique())
    data_by_group = [
        plot_df.loc[plot_df[group_col] == g, "total_intensity"].values for g in groups
    ]

    bp = ax.boxplot(data_by_group, labels=groups, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Total absolute intensity")
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_title("Intensity Distribution by Group")
    ax.grid(axis="y", alpha=0.3)

    return ax


def perform_pca(
    X: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> dict:
    """Perform PCA on the preprocessed data matrix.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed data matrix (samples x features).
    n_components : int
        Number of principal components to compute. Clamped to
        min(n_components, n_samples, n_features).

    Returns
    -------
    result : dict
        Dictionary with keys:
        - scores: np.ndarray (n_samples x n_components)
        - loadings: np.ndarray (n_features x n_components)
        - explained_variance: np.ndarray (n_components,) as fractions
        - cumulative_variance: np.ndarray (n_components,)
        - model: fitted sklearn PCA object
    """
    n_components = min(n_components, *X.shape)

    model = PCA(n_components=n_components)
    scores = model.fit_transform(X)
    loadings = model.components_.T  # n_features x n_components

    explained_variance = model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    logger.info(
        "PCA: %d components, %.1f%% cumulative variance explained",
        n_components,
        cumulative_variance[-1] * 100,
    )

    return {
        "scores": scores,
        "loadings": loadings,
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance,
        "model": model,
    }


def compute_hotelling_t2(
    scores: np.ndarray,
    n_components: int | None = None,
) -> dict:
    """Compute Hotelling's T2 statistic for outlier detection.

    Parameters
    ----------
    scores : np.ndarray
        PCA score matrix (n_samples x n_components_available).
    n_components : int or None
        Number of components to use. If None, all columns of scores are used.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - t2_scores: np.ndarray (n_samples,) T2 values per sample
        - t2_limit: float, F-distribution-based limit
        - outlier_mask: np.ndarray (n_samples,) boolean, True for outliers
        - n_outliers: int
    """
    n_samples = scores.shape[0]
    if n_components is None:
        n_components = scores.shape[1]
    else:
        n_components = min(n_components, scores.shape[1])

    scores_used = scores[:, :n_components]

    # T2 = sum((score_j / std_j)^2) for each sample
    stds = scores_used.std(axis=0, ddof=1)
    # Guard against zero std (constant score column)
    stds[stds == 0] = 1.0

    t2_scores = np.sum((scores_used / stds) ** 2, axis=1)

    # Simplified T2 limit: n_comp * (n - 1) / (n - n_comp)
    if n_samples <= n_components:
        logger.warning(
            "n_samples (%d) <= n_components (%d): T2 limit unreliable",
            n_samples,
            n_components,
        )
        t2_limit = float(np.max(t2_scores) * 10)
    else:
        t2_limit = float(n_components * (n_samples - 1) / (n_samples - n_components))

    threshold = T2_THRESHOLD_MULTIPLIER * t2_limit
    outlier_mask = t2_scores > threshold
    n_outliers = int(np.sum(outlier_mask))

    logger.info(
        "Hotelling T2: %d outliers detected (threshold=%.2f, limit=%.2f)",
        n_outliers,
        threshold,
        t2_limit,
    )

    return {
        "t2_scores": t2_scores,
        "t2_limit": t2_limit,
        "outlier_mask": outlier_mask,
        "n_outliers": n_outliers,
    }


def plot_pca_scores(
    pca_result: dict,
    labels: pd.Series | np.ndarray,
    pc_x: int = 0,
    pc_y: int = 1,
    ax: Axes | None = None,
) -> Axes:
    """Plot PCA score scatter plot colored by group labels.

    Parameters
    ----------
    pca_result : dict
        Output from ``perform_pca``.
    labels : pd.Series or np.ndarray
        Group labels for coloring, one per sample.
    pc_x : int
        Principal component index for x-axis (0-based).
    pc_y : int
        Principal component index for y-axis (0-based).
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    scores = pca_result["scores"]
    explained = pca_result["explained_variance"]

    unique_labels = sorted(pd.Series(labels).unique(), key=str)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(
            scores[mask, pc_x],
            scores[mask, pc_y],
            c=[colors[i % len(colors)]],
            label=str(label),
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            s=50,
        )

    ax.set_xlabel(f"PC{pc_x + 1} ({explained[pc_x]:.1%})")
    ax.set_ylabel(f"PC{pc_y + 1} ({explained[pc_y]:.1%})")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    return ax


def plot_pca_scores_interactive(
    pca_result: dict,
    metadata: pd.DataFrame,
    color_by: str,
    pc_x: int = 0,
    pc_y: int = 1,
) -> go.Figure:
    """Plot interactive PCA score scatter with hover tooltips.

    Parameters
    ----------
    pca_result : dict
        Output from ``perform_pca`` with keys "scores" (n_samples x n_components)
        and "explained_variance" (n_components array).
    metadata : pd.DataFrame
        Sample metadata with columns: subject, sample_type, timepoint.
        Must have same number of rows as pca_result["scores"].
    color_by : str
        Column name in metadata to color points by (e.g., "sample_type",
        "subject", "timepoint").
    pc_x : int, default=0
        Principal component index for x-axis (0-based).
    pc_y : int, default=1
        Principal component index for y-axis (0-based).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure with hover tooltips.

    Raises
    ------
    ValueError
        If metadata row count doesn't match PCA scores, or if color_by column
        is missing, or if PC indices are out of range.

    Examples
    --------
    >>> fig = plot_pca_scores_interactive(
    ...     pca_result, metadata, color_by="sample_type"
    ... )
    >>> fig.show()
    """
    scores = pca_result["scores"]
    explained = pca_result["explained_variance"]

    # Input validation
    if len(metadata) != scores.shape[0]:
        msg = (
            f"Metadata row count ({len(metadata)}) does not match "
            f"PCA scores ({scores.shape[0]})"
        )
        raise ValueError(msg)

    if color_by not in metadata.columns:
        msg = f"Column '{color_by}' not found in metadata"
        raise ValueError(msg)

    if pc_x >= scores.shape[1] or pc_y >= scores.shape[1]:
        msg = (
            f"PC indices ({pc_x}, {pc_y}) out of range for {scores.shape[1]} components"
        )
        raise ValueError(msg)

    # Prepare data
    unique_groups = sorted(metadata[color_by].unique(), key=str)
    colors = [
        f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
        for r, g, b, _ in [plt.cm.tab10(i / 10) for i in range(len(unique_groups))]
    ]

    # Build customdata array with sample metadata
    customdata = metadata[["sample_type", "subject", "timepoint"]].values

    # Create traces for each group
    traces = []
    for i, group in enumerate(unique_groups):
        mask = metadata[color_by] == group
        group_scores = scores[mask]
        group_customdata = customdata[mask]

        trace = go.Scatter(
            x=group_scores[:, pc_x],
            y=group_scores[:, pc_y],
            mode="markers",
            name=str(group),
            marker={
                "size": 8,
                "color": colors[i % len(colors)],
                "line": {"width": 1, "color": "white"},
            },
            customdata=group_customdata,
            hovertemplate=(
                "<b>Sample Type:</b> %{customdata[0]}<br>"
                "<b>Subject:</b> %{customdata[1]}<br>"
                "<b>Timepoint:</b> %{customdata[2]}<br>"
                f"<b>PC{pc_x + 1}:</b> %{{x:.3f}} ({explained[pc_x]:.1%})<br>"
                f"<b>PC{pc_y + 1}:</b> %{{y:.3f}} ({explained[pc_y]:.1%})<br>"
                "<extra></extra>"
            ),
        )
        traces.append(trace)

    # Create figure with all traces
    fig = go.Figure(data=traces)

    # Update layout with styling
    fig.update_layout(
        xaxis={
            "title": f"PC{pc_x + 1} ({explained[pc_x]:.1%})",
            "zeroline": True,
            "zerolinewidth": 1,
            "zerolinecolor": "lightgray",
            "gridcolor": "lightgray",
        },
        yaxis={
            "title": f"PC{pc_y + 1} ({explained[pc_y]:.1%})",
            "zeroline": True,
            "zerolinewidth": 1,
            "zerolinecolor": "lightgray",
            "gridcolor": "lightgray",
        },
        plot_bgcolor="white",
        width=700,
        height=500,
        legend={"x": 1.05, "y": 1, "xanchor": "left"},
        hovermode="closest",
    )

    logger.info(
        "Interactive PCA plot: %d groups, %d samples, PC%d vs PC%d",
        len(unique_groups),
        len(metadata),
        pc_x + 1,
        pc_y + 1,
    )

    return fig


def plot_explained_variance(
    pca_result: dict,
    ax: Axes | None = None,
) -> Axes:
    """Plot explained variance (scree plot) with individual and cumulative values.

    Parameters
    ----------
    pca_result : dict
        Output from ``perform_pca``.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    explained = pca_result["explained_variance"]
    cumulative = pca_result["cumulative_variance"]
    n_comp = len(explained)
    x = np.arange(1, n_comp + 1)

    ax.bar(x, explained * 100, alpha=0.7, color="steelblue", label="Individual")
    ax2 = ax.twinx()
    ax2.plot(
        x, cumulative * 100, "o-", color="darkorange", linewidth=2, label="Cumulative"
    )
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 105)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_xticks(x)
    ax.set_title("Scree Plot")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    return ax


def plot_hotelling_t2(
    t2_scores: np.ndarray,
    t2_limit: float,
    sample_names: list[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot Hotelling's T2 bar chart with outlier threshold.

    Parameters
    ----------
    t2_scores : np.ndarray
        T2 values per sample.
    t2_limit : float
        T2 limit value. Threshold is drawn at ``T2_THRESHOLD_MULTIPLIER * t2_limit``.
    sample_names : list of str or None
        Sample names for x-axis labels. If None, integer indices are used.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    threshold = T2_THRESHOLD_MULTIPLIER * t2_limit
    n_samples = len(t2_scores)
    x = np.arange(n_samples)

    colors = ["red" if t > threshold else "steelblue" for t in t2_scores]
    ax.bar(x, t2_scores, color=colors, alpha=0.8, width=0.8)
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({T2_THRESHOLD_MULTIPLIER}x limit)",
    )
    ax.axhline(t2_limit, color="orange", linestyle=":", linewidth=1, label="T2 limit")

    ax.set_ylabel("Hotelling's T$^2$")
    ax.set_xlabel("Sample")
    ax.set_title("Hotelling's T$^2$ Outlier Detection")
    ax.legend(fontsize=8)

    if sample_names is not None and n_samples <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels(sample_names, rotation=90, fontsize=6)
    elif n_samples > 60:
        ax.set_xticks(x[::5])
        ax.set_xticklabels([str(i) for i in x[::5]], fontsize=7)

    ax.grid(axis="y", alpha=0.3)

    return ax
