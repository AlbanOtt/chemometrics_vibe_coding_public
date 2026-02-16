"""Multivariate analysis: PCA trajectories, PLS-DA, VIP scores, S-plot, and time-trending features.

Implements supervised and unsupervised multivariate methods for discriminating
drug effect timepoints and identifying biomarker candidates in LC-MS metabolomics data.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

DEFAULT_N_COMPONENTS_PLSDA = 2
DEFAULT_CV_FOLDS = 7
DEFAULT_N_PERMUTATIONS = 100
VIP_THRESHOLD = 1.0
FDR_THRESHOLD = 0.05
DEFAULT_TIME_THRESHOLD = 60.0  # minutes, separates early vs late


def assign_time_class(
    metadata: pd.DataFrame,
    time_threshold: float = DEFAULT_TIME_THRESHOLD,
) -> np.ndarray:
    """Assign binary class labels based on timepoint relative to a threshold.

    Parameters
    ----------
    metadata : pd.DataFrame
        Sample metadata with a ``timepoint`` column (numeric, in minutes).
    time_threshold : float
        Timepoint threshold in minutes. Samples with timepoint <= threshold
        are labelled "early", samples with timepoint > threshold are "late".

    Returns
    -------
    labels : np.ndarray
        Array of string labels ("early" or "late"), one per sample.
    """
    labels = np.where(metadata["timepoint"].values <= time_threshold, "early", "late")
    n_early = int(np.sum(labels == "early"))
    n_late = int(np.sum(labels == "late"))
    logger.info(
        "Time class assignment (threshold=%.0f min): %d early, %d late",
        time_threshold,
        n_early,
        n_late,
    )
    return labels


def perform_plsda(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS_PLSDA,
    cv: int = DEFAULT_CV_FOLDS,
) -> dict:
    """Perform PLS-DA with cross-validated Q2 and VIP scores.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed data matrix (samples x features).
    y : np.ndarray
        Binary class labels (e.g., "early" / "late").
    n_components : int
        Number of PLS components to fit.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - model: fitted PLSRegression object
        - scores: np.ndarray (n_samples x n_components) X scores
        - x_loadings: np.ndarray (n_features x n_components)
        - vip: np.ndarray (n_features,) VIP scores
        - y_encoded: np.ndarray numeric class labels (0/1)
        - classes: list[str] class names in sorted order
        - r2y: float training R2
        - q2: float cross-validated Q2
        - q2_std: float standard deviation of CV fold R2 scores
    """
    classes = sorted(np.unique(y))
    y_encoded = (y == classes[1]).astype(float)

    n_components = min(n_components, X.shape[0] - 1, X.shape[1])

    model = PLSRegression(n_components=n_components)
    model.fit(X, y_encoded)

    # Cross-validated Q2
    cv_folds = min(cv, X.shape[0])
    cv_scores = cross_val_score(model, X, y_encoded, cv=cv_folds, scoring="r2")
    q2 = float(cv_scores.mean())
    q2_std = float(cv_scores.std())

    # Training R2
    r2y = float(model.score(X, y_encoded))

    vip = calculate_vip(model)

    logger.info(
        "PLS-DA: %d components, R2Y=%.3f, Q2=%.3f (+/- %.3f)",
        n_components,
        r2y,
        q2,
        q2_std,
    )

    return {
        "model": model,
        "scores": model.x_scores_,
        "x_loadings": model.x_loadings_,
        "vip": vip,
        "y_encoded": y_encoded,
        "classes": list(classes),
        "r2y": r2y,
        "q2": q2,
        "q2_std": q2_std,
    }


def calculate_vip(pls_model: PLSRegression) -> np.ndarray:
    """Calculate Variable Importance in Projection (VIP) scores.

    Features with VIP > 1.0 are considered important for the model.

    Parameters
    ----------
    pls_model : PLSRegression
        Fitted PLS model.

    Returns
    -------
    vip : np.ndarray
        VIP scores, one per feature (shape: n_features).
    """
    t = pls_model.x_scores_  # (n_samples, h)
    w = pls_model.x_weights_  # (p, h)
    q = pls_model.y_loadings_  # (n_targets, h)

    p = w.shape[0]

    # SS per component: diag(T'T @ Q'Q) gives ||t_a||^2 * ||q_a||^2 for orthogonal scores
    ss = np.diag(t.T @ t @ q.T @ q)  # shape (h,)
    total_ss = np.sum(ss)

    if total_ss == 0:
        logger.warning("Total explained SS is zero; returning uniform VIP scores")
        return np.ones(p)

    # VIP_j = sqrt(p * sum_a(SS_a * w_ja^2) / total_SS)
    vip = np.sqrt(p * (w**2 @ ss) / total_ss)

    return vip


def permutation_test_plsda(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS_PLSDA,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    cv: int = DEFAULT_CV_FOLDS,
    random_state: int = 42,
) -> dict:
    """Run permutation test to assess PLS-DA model significance.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed data matrix (samples x features).
    y : np.ndarray
        Binary class labels.
    n_components : int
        Number of PLS components.
    n_permutations : int
        Number of label permutations.
    cv : int
        Number of cross-validation folds.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - observed_q2: float
        - null_q2: np.ndarray (n_permutations,)
        - pvalue: float
    """
    classes = sorted(np.unique(y))
    y_encoded = (y == classes[1]).astype(float)

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    model = PLSRegression(n_components=n_comp)
    cv_folds = min(cv, X.shape[0])

    # Observed Q2
    cv_scores = cross_val_score(model, X, y_encoded, cv=cv_folds, scoring="r2")
    observed_q2 = float(cv_scores.mean())

    # Permutation null distribution
    rng = np.random.default_rng(random_state)
    null_q2 = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y_encoded)
        perm_scores = cross_val_score(model, X, y_perm, cv=cv_folds, scoring="r2")
        null_q2[i] = perm_scores.mean()

    pvalue = float((np.sum(null_q2 >= observed_q2) + 1) / (n_permutations + 1))

    logger.info(
        "Permutation test: observed Q2=%.3f, p=%.4f (%d permutations)",
        observed_q2,
        pvalue,
        n_permutations,
    )

    return {
        "observed_q2": observed_q2,
        "null_q2": null_q2,
        "pvalue": pvalue,
    }


def create_splot(
    X: np.ndarray,
    pls_model: PLSRegression,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute S-plot coordinates from a fitted PLS model.

    The S-plot combines covariance (magnitude of change) with correlation
    (reliability of change) for each feature against the first PLS component scores.
    Features in the upper-right or lower-left corners are the best biomarker candidates.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed data matrix (samples x features).
    pls_model : PLSRegression
        Fitted PLS model.

    Returns
    -------
    p_cov : np.ndarray
        Covariance of each feature with the first component scores (shape: n_features).
    p_corr : np.ndarray
        Correlation of each feature with the first component scores (shape: n_features).
    """
    t1 = pls_model.x_scores_[:, 0]
    t1_var = np.var(t1)

    n_features = X.shape[1]
    p_cov = np.zeros(n_features)
    p_corr = np.zeros(n_features)

    for j in range(n_features):
        xj = X[:, j]
        p_cov[j] = np.cov(xj, t1)[0, 1] / t1_var if t1_var > 0 else 0.0
        corr_val = np.corrcoef(xj, t1)[0, 1]
        p_corr[j] = corr_val if np.isfinite(corr_val) else 0.0

    return p_cov, p_corr


def extract_splot_biomarkers(
    p_cov: np.ndarray,
    p_corr: np.ndarray,
    feature_ids: list[int],
    threshold_cov: float | None = None,
    threshold_corr: float = 0.8,
) -> pd.DataFrame:
    """Extract biomarker candidates from S-plot.

    Identifies features with both high covariance AND high correlation
    with the first PLS-DA component.

    Parameters
    ----------
    p_cov : np.ndarray
        Covariance values (from create_splot).
    p_corr : np.ndarray
        Correlation values (from create_splot).
    feature_ids : list[int]
        Feature identifiers.
    threshold_cov : float | None
        Covariance threshold (if None, uses 95th percentile of |p_cov|).
    threshold_corr : float
        Correlation threshold (default 0.8).

    Returns
    -------
    pd.DataFrame
        Biomarker candidates with columns:
        - feature_id: Feature identifier
        - covariance: p[cov] value
        - correlation: p[corr] value
        - abs_covariance: |p[cov]|
        - abs_correlation: |p[corr]|

        Sorted by abs_covariance descending (most important first).
    """
    if threshold_cov is None:
        threshold_cov = float(np.percentile(np.abs(p_cov), 95))

    significant = (np.abs(p_cov) > threshold_cov) & (np.abs(p_corr) > threshold_corr)

    if not significant.any():
        logger.info("No biomarker candidates found with given thresholds")
        return pd.DataFrame(
            columns=[
                "feature_id",
                "covariance",
                "correlation",
                "abs_covariance",
                "abs_correlation",
            ]
        )

    sig_indices = np.where(significant)[0]

    biomarkers = pd.DataFrame(
        {
            "feature_id": [feature_ids[i] for i in sig_indices],
            "covariance": p_cov[sig_indices],
            "correlation": p_corr[sig_indices],
            "abs_covariance": np.abs(p_cov[sig_indices]),
            "abs_correlation": np.abs(p_corr[sig_indices]),
        }
    )

    biomarkers = biomarkers.sort_values("abs_covariance", ascending=False).reset_index(
        drop=True
    )

    logger.info(
        "Extracted %d biomarker candidates (cov>%.3f, corr>%.2f)",
        len(biomarkers),
        threshold_cov,
        threshold_corr,
    )

    return biomarkers


def find_time_trending_features(
    X: np.ndarray,
    timepoints: np.ndarray,
    feature_ids: list[int],
    fdr_threshold: float = FDR_THRESHOLD,
) -> pd.DataFrame:
    """Identify features with significant monotonic trends over time.

    Uses Spearman rank correlation between each feature and timepoint, with
    Benjamini-Hochberg FDR correction for multiple testing.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed data matrix (samples x features).
    timepoints : np.ndarray
        Timepoint values per sample (numeric, e.g. minutes).
    feature_ids : list[int]
        Feature identifiers matching columns of X.
    fdr_threshold : float
        FDR-corrected p-value threshold for significance.

    Returns
    -------
    results : pd.DataFrame
        One row per feature with columns: feature_id, rho, pvalue, pvalue_fdr,
        significant. Sorted by absolute rho descending.
    """
    n_features = X.shape[1]
    rhos = np.zeros(n_features)
    pvalues = np.zeros(n_features)

    for j in range(n_features):
        rho, pval = stats.spearmanr(timepoints, X[:, j])
        rhos[j] = rho if np.isfinite(rho) else 0.0
        pvalues[j] = pval if np.isfinite(pval) else 1.0

    # Benjamini-Hochberg FDR correction
    _, pvalues_fdr, _, _ = multipletests(pvalues, alpha=fdr_threshold, method="fdr_bh")

    results = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "rho": rhos,
            "pvalue": pvalues,
            "pvalue_fdr": pvalues_fdr,
            "significant": pvalues_fdr < fdr_threshold,
        }
    )

    results = results.sort_values("rho", key=np.abs, ascending=False).reset_index(
        drop=True
    )

    n_sig = int(results["significant"].sum())
    logger.info(
        "Time-trending features: %d/%d significant (FDR < %.2f)",
        n_sig,
        n_features,
        fdr_threshold,
    )

    return results


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_pca_trajectories(
    pca_result: dict,
    metadata: pd.DataFrame,
    pc_x: int = 0,
    pc_y: int = 1,
    ax: Axes | None = None,
) -> Axes:
    """Plot PCA score trajectories connecting timepoints per subject.

    Parameters
    ----------
    pca_result : dict
        Output from ``eda.perform_pca``.
    metadata : pd.DataFrame
        Sample metadata with columns: subject, timepoint.
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
        _, ax = plt.subplots(figsize=(10, 8))

    scores = pca_result["scores"]
    explained = pca_result["explained_variance"]

    subjects = sorted(metadata["subject"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 1)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

    for i, subject in enumerate(subjects):
        mask = metadata["subject"].values == subject
        sub_scores = scores[mask]
        sub_timepoints = metadata.loc[mask, "timepoint"].values

        # Sort by timepoint to draw trajectory in order
        sort_idx = np.argsort(sub_timepoints)
        sub_scores = sub_scores[sort_idx]
        sub_timepoints = sub_timepoints[sort_idx]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Trajectory line
        ax.plot(
            sub_scores[:, pc_x],
            sub_scores[:, pc_y],
            "-",
            color=color,
            alpha=0.4,
            linewidth=1.5,
            zorder=1,
        )

        # Points
        ax.scatter(
            sub_scores[:, pc_x],
            sub_scores[:, pc_y],
            c=[color],
            marker=marker,
            s=60,
            edgecolors="white",
            linewidth=0.5,
            label=str(subject),
            zorder=2,
        )

        # Annotate timepoints
        for k, tp in enumerate(sub_timepoints):
            ax.annotate(
                f"{tp:.0f}",
                (sub_scores[k, pc_x], sub_scores[k, pc_y]),
                fontsize=6,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel(f"PC{pc_x + 1} ({explained[pc_x]:.1%})")
    ax.set_ylabel(f"PC{pc_y + 1} ({explained[pc_y]:.1%})")
    ax.set_title("PCA Time Trajectories")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, loc="best", title="Subject")
    ax.grid(alpha=0.3)

    return ax


def plot_pca_trajectories_plotly(
    pca_result: dict,
    metadata: pd.DataFrame,
    pc_x: int = 0,
    pc_y: int = 1,
) -> go.Figure:
    """Plot PCA score trajectories with plotly (interactive).

    Parameters
    ----------
    pca_result : dict
        Output from ``eda.perform_pca``.
    metadata : pd.DataFrame
        Sample metadata with columns: subject, timepoint.
    pc_x : int
        Principal component index for x-axis (0-based).
    pc_y : int
        Principal component index for y-axis (0-based).

    Returns
    -------
    go.Figure
        Plotly figure with interactive trajectory plot.
    """
    scores = pca_result["scores"]
    explained = pca_result["explained_variance"]

    subjects = sorted(metadata["subject"].unique())
    colors = [
        f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
        for r, g, b, _ in plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 1)))
    ]

    fig = go.Figure()

    for i, subject in enumerate(subjects):
        mask = metadata["subject"].values == subject
        sub_scores = scores[mask]
        sub_timepoints = metadata.loc[mask, "timepoint"].values

        sort_idx = np.argsort(sub_timepoints)
        sub_scores = sub_scores[sort_idx]
        sub_timepoints = sub_timepoints[sort_idx]

        color = colors[i % len(colors)]

        # Trajectory line
        fig.add_trace(
            go.Scatter(
                x=sub_scores[:, pc_x],
                y=sub_scores[:, pc_y],
                mode="lines+markers+text",
                name=str(subject),
                line={"color": color, "width": 2},
                marker={"size": 10, "color": color},
                text=[f"{tp:.0f}" for tp in sub_timepoints],
                textposition="top right",
                textfont={"size": 8},
                hovertemplate=(
                    f"Subject: {subject}<br>"
                    + "PC%d: %%{x:.3f}<br>" % (pc_x + 1)
                    + "PC%d: %%{y:.3f}<br>" % (pc_y + 1)
                    + "Time: %{text} min<extra></extra>"
                ),
            )
        )

    # Add reference lines
    fig.add_hline(y=0, line={"color": "grey", "width": 0.5, "dash": "dash"})
    fig.add_vline(x=0, line={"color": "grey", "width": 0.5, "dash": "dash"})

    fig.update_layout(
        title="PCA Time Trajectories",
        xaxis_title=f"PC{pc_x + 1} ({explained[pc_x]:.1%})",
        yaxis_title=f"PC{pc_y + 1} ({explained[pc_y]:.1%})",
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
        legend={"title": {"text": "Subject"}},
    )

    return fig


def plot_plsda_scores(
    plsda_result: dict,
    labels: np.ndarray,
    comp_x: int = 0,
    comp_y: int = 1,
    ax: Axes | None = None,
) -> Axes:
    """Plot PLS-DA score scatter plot colored by class labels.

    Parameters
    ----------
    plsda_result : dict
        Output from ``perform_plsda``.
    labels : np.ndarray
        Class labels for coloring (e.g., "early" / "late").
    comp_x : int
        PLS component index for x-axis (0-based).
    comp_y : int
        PLS component index for y-axis (0-based).
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    scores = plsda_result["scores"]
    class_colors = {"early": "tab:blue", "late": "tab:red"}

    for label in sorted(np.unique(labels)):
        mask = labels == label
        color = class_colors.get(label, "tab:grey")
        ax.scatter(
            scores[mask, comp_x],
            scores[mask, comp_y],
            c=color,
            label=label,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            s=60,
        )

    r2y = plsda_result["r2y"]
    q2 = plsda_result["q2"]
    ax.set_xlabel(f"LV{comp_x + 1}")
    ax.set_ylabel(f"LV{comp_y + 1}")
    ax.set_title(f"PLS-DA Scores (R\u00b2Y={r2y:.3f}, Q\u00b2={q2:.3f})")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    return ax


def plot_plsda_scores_plotly(
    plsda_result: dict,
    labels: np.ndarray,
    comp_x: int = 0,
    comp_y: int = 1,
) -> go.Figure:
    """Plot PLS-DA score scatter with plotly (interactive).

    Parameters
    ----------
    plsda_result : dict
        Output from ``perform_plsda``.
    labels : np.ndarray
        Class labels for coloring (e.g., "early" / "late").
    comp_x : int
        PLS component index for x-axis (0-based).
    comp_y : int
        PLS component index for y-axis (0-based).

    Returns
    -------
    go.Figure
        Plotly figure with interactive score plot.
    """
    scores = plsda_result["scores"]
    r2y = plsda_result["r2y"]
    q2 = plsda_result["q2"]

    class_colors = {"early": "blue", "late": "red"}

    fig = go.Figure()

    for label in sorted(np.unique(labels)):
        mask = labels == label
        color = class_colors.get(label, "grey")

        fig.add_trace(
            go.Scatter(
                x=scores[mask, comp_x],
                y=scores[mask, comp_y],
                mode="markers",
                name=label,
                marker={
                    "size": 10,
                    "color": color,
                    "line": {"width": 0.5, "color": "white"},
                },
                hovertemplate=(
                    f"{label}<br>"
                    + "LV%d: %%{x:.3f}<br>" % (comp_x + 1)
                    + "LV%d: %%{y:.3f}<extra></extra>" % (comp_y + 1)
                ),
            )
        )

    # Add reference lines
    fig.add_hline(y=0, line={"color": "grey", "width": 0.5, "dash": "dash"})
    fig.add_vline(x=0, line={"color": "grey", "width": 0.5, "dash": "dash"})

    fig.update_layout(
        title=f"PLS-DA Scores (R²Y={r2y:.3f}, Q²={q2:.3f})",
        xaxis_title=f"LV{comp_x + 1}",
        yaxis_title=f"LV{comp_y + 1}",
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
    )

    return fig


def plot_vip_scores(
    vip: np.ndarray,
    feature_ids: list[int],
    n_top: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Plot top VIP scores as a horizontal bar chart.

    Parameters
    ----------
    vip : np.ndarray
        VIP scores, one per feature.
    feature_ids : list[int]
        Feature identifiers matching VIP array indices.
    n_top : int
        Number of top features to display.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(6, n_top * 0.25)))

    sorted_idx = np.argsort(vip)[::-1][:n_top]
    top_vip = vip[sorted_idx]
    top_ids = [feature_ids[i] for i in sorted_idx]

    # Reverse for bottom-to-top display (highest at top)
    y_pos = np.arange(len(top_vip))
    colors = ["tab:red" if v >= VIP_THRESHOLD else "steelblue" for v in top_vip[::-1]]

    ax.barh(y_pos, top_vip[::-1], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(fid) for fid in top_ids[::-1]], fontsize=7)
    ax.axvline(
        VIP_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"VIP={VIP_THRESHOLD}",
    )
    ax.set_xlabel("VIP Score")
    ax.set_ylabel("Feature ID")
    ax.set_title(f"Top {n_top} VIP Scores")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    return ax


def plot_vip_scores_plotly(
    vip: np.ndarray,
    feature_ids: list[int],
    n_top: int = 30,
) -> go.Figure:
    """Plot top VIP scores with plotly (interactive horizontal bar chart).

    Parameters
    ----------
    vip : np.ndarray
        VIP scores, one per feature.
    feature_ids : list[int]
        Feature identifiers matching VIP array indices.
    n_top : int
        Number of top features to display.

    Returns
    -------
    go.Figure
        Plotly figure with interactive VIP bar chart.
    """
    sorted_idx = np.argsort(vip)[::-1][:n_top]
    top_vip = vip[sorted_idx]
    top_ids = [feature_ids[i] for i in sorted_idx]

    # Reverse for bottom-to-top display (highest at top)
    top_vip_reversed = top_vip[::-1]
    top_ids_reversed = top_ids[::-1]

    colors = ["red" if v >= VIP_THRESHOLD else "steelblue" for v in top_vip_reversed]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_vip_reversed,
            y=[str(fid) for fid in top_ids_reversed],
            orientation="h",
            marker={"color": colors},
            hovertemplate="Feature: %{y}<br>VIP: %{x:.3f}<extra></extra>",
        )
    )

    # Add threshold line
    fig.add_vline(
        x=VIP_THRESHOLD,
        line={"color": "red", "width": 2, "dash": "dash"},
        annotation_text=f"VIP={VIP_THRESHOLD}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Top {n_top} VIP Scores",
        xaxis_title="VIP Score",
        yaxis_title="Feature ID",
        template="plotly_white",
        showlegend=False,
        height=max(400, n_top * 20),
    )

    return fig


def plot_splot(
    p_cov: np.ndarray,
    p_corr: np.ndarray,
    feature_ids: list[int],
    threshold_cov: float | None = None,
    threshold_corr: float = 0.8,
    ax: Axes | None = None,
) -> Axes:
    """Plot S-plot with significant biomarker candidates highlighted.

    Parameters
    ----------
    p_cov : np.ndarray
        Covariance values from ``create_splot``.
    p_corr : np.ndarray
        Correlation values from ``create_splot``.
    feature_ids : list[int]
        Feature identifiers.
    threshold_cov : float or None
        Covariance magnitude threshold. If None, uses the 95th percentile of |p_cov|.
    threshold_corr : float
        Correlation magnitude threshold for highlighting.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if threshold_cov is None:
        threshold_cov = float(np.percentile(np.abs(p_cov), 95))

    significant = (np.abs(p_cov) > threshold_cov) & (np.abs(p_corr) > threshold_corr)

    ax.scatter(
        p_cov[~significant],
        p_corr[~significant],
        c="grey",
        alpha=0.3,
        s=20,
        zorder=1,
    )
    ax.scatter(
        p_cov[significant],
        p_corr[significant],
        c="red",
        alpha=0.8,
        s=60,
        edgecolors="darkred",
        linewidth=0.5,
        zorder=2,
        label=f"Candidates (n={int(significant.sum())})",
    )

    for idx in np.where(significant)[0]:
        ax.annotate(
            str(feature_ids[idx]),
            (p_cov[idx], p_corr[idx]),
            fontsize=6,
            color="darkred",
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("p(cov) \u2014 Covariance")
    ax.set_ylabel("p(corr) \u2014 Correlation")
    ax.set_title("S-Plot for Biomarker Selection")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    return ax


def plot_splot_plotly(
    p_cov: np.ndarray,
    p_corr: np.ndarray,
    feature_ids: list[int],
    threshold_cov: float | None = None,
    threshold_corr: float = 0.8,
) -> go.Figure:
    """Plot S-plot with plotly (interactive biomarker selection).

    Parameters
    ----------
    p_cov : np.ndarray
        Covariance values from ``create_splot``.
    p_corr : np.ndarray
        Correlation values from ``create_splot``.
    feature_ids : list[int]
        Feature identifiers.
    threshold_cov : float or None
        Covariance magnitude threshold. If None, uses the 95th percentile of |p_cov|.
    threshold_corr : float
        Correlation magnitude threshold for highlighting.

    Returns
    -------
    go.Figure
        Plotly figure with interactive S-plot.
    """
    if threshold_cov is None:
        threshold_cov = float(np.percentile(np.abs(p_cov), 95))

    significant = (np.abs(p_cov) > threshold_cov) & (np.abs(p_corr) > threshold_corr)

    fig = go.Figure()

    # Non-significant features
    fig.add_trace(
        go.Scatter(
            x=p_cov[~significant],
            y=p_corr[~significant],
            mode="markers",
            name="Non-significant",
            marker={"size": 5, "color": "grey", "opacity": 0.3},
            hovertemplate="Feature: N/A<br>Cov: %{x:.3f}<br>Corr: %{y:.3f}<extra></extra>",
        )
    )

    # Significant features (biomarker candidates)
    if significant.any():
        sig_indices = np.where(significant)[0]
        fig.add_trace(
            go.Scatter(
                x=p_cov[significant],
                y=p_corr[significant],
                mode="markers+text",
                name=f"Candidates (n={int(significant.sum())})",
                marker={
                    "size": 10,
                    "color": "red",
                    "opacity": 0.8,
                    "line": {"width": 0.5, "color": "darkred"},
                },
                text=[str(feature_ids[i]) for i in sig_indices],
                textposition="top right",
                textfont={"size": 8, "color": "darkred"},
                hovertemplate=(
                    "Feature: %{text}<br>Cov: %{x:.3f}<br>Corr: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Add reference lines
    fig.add_hline(y=0, line={"color": "grey", "width": 0.5, "dash": "dash"})
    fig.add_vline(x=0, line={"color": "grey", "width": 0.5, "dash": "dash"})
    fig.add_hline(
        y=threshold_corr,
        line={"color": "red", "width": 1, "dash": "dot"},
        opacity=0.5,
    )
    fig.add_hline(
        y=-threshold_corr,
        line={"color": "red", "width": 1, "dash": "dot"},
        opacity=0.5,
    )

    fig.update_layout(
        title="S-Plot for Biomarker Selection",
        xaxis_title="p(cov) — Covariance",
        yaxis_title="p(corr) — Correlation",
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
    )

    return fig


def plot_permutation_test(
    permutation_result: dict,
    ax: Axes | None = None,
) -> Axes:
    """Plot permutation test histogram with observed Q2 line.

    Parameters
    ----------
    permutation_result : dict
        Output from ``permutation_test_plsda``.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    null_q2 = permutation_result["null_q2"]
    observed_q2 = permutation_result["observed_q2"]
    pvalue = permutation_result["pvalue"]

    ax.hist(
        null_q2,
        bins=30,
        color="steelblue",
        alpha=0.7,
        edgecolor="white",
        label="Null distribution",
    )
    ax.axvline(
        observed_q2,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Observed Q\u00b2={observed_q2:.3f}",
    )
    ax.set_xlabel("Q\u00b2")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Test (p={pvalue:.4f})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    return ax


def plot_permutation_test_plotly(
    permutation_result: dict,
) -> go.Figure:
    """Plot permutation test with plotly (interactive histogram).

    Parameters
    ----------
    permutation_result : dict
        Output from ``permutation_test_plsda``.

    Returns
    -------
    go.Figure
        Plotly figure with interactive permutation test histogram.
    """
    null_q2 = permutation_result["null_q2"]
    observed_q2 = permutation_result["observed_q2"]
    pvalue = permutation_result["pvalue"]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=null_q2,
            nbinsx=30,
            name="Null distribution",
            marker={"color": "steelblue", "line": {"width": 0.5, "color": "white"}},
            hovertemplate="Q²: %{x:.3f}<br>Count: %{y}<extra></extra>",
        )
    )

    # Add observed Q² line
    fig.add_vline(
        x=observed_q2,
        line={"color": "red", "width": 2, "dash": "dash"},
        annotation_text=f"Observed Q²={observed_q2:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Permutation Test (p={pvalue:.4f})",
        xaxis_title="Q²",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=True,
    )

    return fig


def plot_time_trending_volcano(
    trending_results: pd.DataFrame,
    fdr_threshold: float = FDR_THRESHOLD,
    ax: Axes | None = None,
) -> Axes:
    """Plot volcano-style scatter of time-trending features.

    X-axis is Spearman rho, Y-axis is -log10(FDR p-value).

    Parameters
    ----------
    trending_results : pd.DataFrame
        Output from ``find_time_trending_features``.
    fdr_threshold : float
        FDR threshold for significance line.
    ax : Axes or None
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    neg_log_p = -np.log10(np.clip(trending_results["pvalue_fdr"].values, 1e-300, 1.0))
    rho = trending_results["rho"].values
    significant = trending_results["significant"].values

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

    ax.axhline(
        -np.log10(fdr_threshold),
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"FDR={fdr_threshold}",
    )
    ax.set_xlabel("Spearman \u03c1 (correlation with timepoint)")
    ax.set_ylabel("-log\u2081\u2080(FDR p-value)")
    ax.set_title("Time-Trending Features")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    return ax


def plot_time_trending_volcano_plotly(
    trending_results: pd.DataFrame,
    fdr_threshold: float = FDR_THRESHOLD,
) -> go.Figure:
    """Plot volcano-style scatter with plotly (interactive time-trending features).

    X-axis is Spearman rho, Y-axis is -log10(FDR p-value).

    Parameters
    ----------
    trending_results : pd.DataFrame
        Output from ``find_time_trending_features``.
    fdr_threshold : float
        FDR threshold for significance line.

    Returns
    -------
    go.Figure
        Plotly figure with interactive volcano plot.
    """
    neg_log_p = -np.log10(np.clip(trending_results["pvalue_fdr"].values, 1e-300, 1.0))
    rho = trending_results["rho"].values
    significant = trending_results["significant"].values
    feature_ids = trending_results["feature_id"].values

    fig = go.Figure()

    # Non-significant features
    fig.add_trace(
        go.Scatter(
            x=rho[~significant],
            y=neg_log_p[~significant],
            mode="markers",
            name="Non-significant",
            marker={"size": 5, "color": "grey", "opacity": 0.3},
            hovertemplate=(
                "Feature: %{customdata}<br>"
                + "ρ: %{x:.3f}<br>"
                + "-log10(FDR p): %{y:.2f}<extra></extra>"
            ),
            customdata=feature_ids[~significant],
        )
    )

    # Significant features
    if significant.any():
        fig.add_trace(
            go.Scatter(
                x=rho[significant],
                y=neg_log_p[significant],
                mode="markers",
                name=f"Significant (n={int(significant.sum())})",
                marker={"size": 8, "color": "red", "opacity": 0.7},
                hovertemplate=(
                    "Feature: %{customdata}<br>"
                    + "ρ: %{x:.3f}<br>"
                    + "-log10(FDR p): %{y:.2f}<extra></extra>"
                ),
                customdata=feature_ids[significant],
            )
        )

    # Add significance threshold line
    fig.add_hline(
        y=-np.log10(fdr_threshold),
        line={"color": "red", "width": 1, "dash": "dash"},
        annotation_text=f"FDR={fdr_threshold}",
        annotation_position="right",
    )

    fig.update_layout(
        title="Time-Trending Features",
        xaxis_title="Spearman ρ (correlation with timepoint)",
        yaxis_title="-log₁₀(FDR p-value)",
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
    )

    return fig
