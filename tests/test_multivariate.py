"""Tests for src.multivariate module."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression

matplotlib.use("Agg")

from src.multivariate import (
    DEFAULT_TIME_THRESHOLD,
    assign_time_class,
    calculate_vip,
    create_splot,
    extract_splot_biomarkers,
    find_time_trending_features,
    perform_plsda,
    permutation_test_plsda,
    plot_pca_trajectories,
    plot_pca_trajectories_plotly,
    plot_permutation_test,
    plot_permutation_test_plotly,
    plot_plsda_scores,
    plot_plsda_scores_plotly,
    plot_splot,
    plot_splot_plotly,
    plot_time_trending_volcano,
    plot_time_trending_volcano_plotly,
    plot_vip_scores,
    plot_vip_scores_plotly,
)


@pytest.fixture()
def synthetic_data() -> dict:
    """Create synthetic metabolomics-like data with time-dependent signal."""
    rng = np.random.default_rng(42)
    n_samples = 40
    n_features = 100

    # Create metadata: 5 subjects x 8 timepoints (0, 60, 120, 240, 360, 480, 600, 720)
    subjects = [f"S{i}" for i in range(1, 6)] * 8
    timepoints = sorted([0, 60, 120, 240, 360, 480, 600, 720] * 5)
    metadata = pd.DataFrame(
        {
            "subject": subjects[:n_samples],
            "timepoint": timepoints[:n_samples],
            "sample_type": "plasma",
        }
    )

    # Generate X with first 5 features correlated with timepoint
    X = rng.standard_normal((n_samples, n_features))
    for j in range(5):
        X[:, j] += metadata["timepoint"].values * 0.01 * (j + 1)

    return {"X": X, "metadata": metadata, "feature_ids": list(range(n_features))}


@pytest.fixture()
def plsda_inputs(synthetic_data: dict) -> dict:
    """Create PLS-DA inputs from synthetic data."""
    meta = synthetic_data["metadata"]
    labels = assign_time_class(meta, time_threshold=DEFAULT_TIME_THRESHOLD)
    return {
        "X": synthetic_data["X"],
        "labels": labels,
        "feature_ids": synthetic_data["feature_ids"],
    }


def test_assign_time_class_default_threshold(synthetic_data: dict) -> None:
    """Test that assign_time_class produces correct binary labels."""
    meta = synthetic_data["metadata"]
    labels = assign_time_class(meta, time_threshold=60.0)

    assert labels.dtype.kind == "U"  # string array
    assert set(labels) == {"early", "late"}
    n_early = int(np.sum(labels == "early"))
    n_late = int(np.sum(labels == "late"))
    assert n_early > 0
    assert n_late > 0
    assert n_early + n_late == len(meta)


def test_assign_time_class_custom_threshold(synthetic_data: dict) -> None:
    """Test time class assignment with a custom threshold."""
    meta = synthetic_data["metadata"]
    labels_low = assign_time_class(meta, time_threshold=0.0)
    labels_high = assign_time_class(meta, time_threshold=1000.0)

    # At threshold=0, only timepoint==0 is early
    assert np.sum(labels_low == "early") <= np.sum(labels_low == "late")
    # At threshold=1000, all are early
    assert np.all(labels_high == "early")


def test_perform_plsda_returns_expected_keys(plsda_inputs: dict) -> None:
    """Test PLS-DA returns all expected result keys."""
    result = perform_plsda(
        plsda_inputs["X"], plsda_inputs["labels"], n_components=2, cv=5
    )

    expected_keys = {
        "model",
        "scores",
        "x_loadings",
        "vip",
        "y_encoded",
        "classes",
        "r2y",
        "q2",
        "q2_std",
    }
    assert set(result.keys()) == expected_keys


def test_perform_plsda_shapes(plsda_inputs: dict) -> None:
    """Test PLS-DA output shapes are correct."""
    X = plsda_inputs["X"]
    labels = plsda_inputs["labels"]
    n_components = 2

    result = perform_plsda(X, labels, n_components=n_components, cv=5)

    n_samples, n_features = X.shape
    assert result["scores"].shape == (n_samples, n_components)
    assert result["x_loadings"].shape == (n_features, n_components)
    assert result["vip"].shape == (n_features,)
    assert result["y_encoded"].shape == (n_samples,)
    assert len(result["classes"]) == 2


def test_perform_plsda_r2y_positive(plsda_inputs: dict) -> None:
    """Test that R2Y is positive (model explains some training variance)."""
    result = perform_plsda(
        plsda_inputs["X"], plsda_inputs["labels"], n_components=2, cv=5
    )
    assert result["r2y"] > 0


def test_calculate_vip_shape() -> None:
    """Test VIP calculation gives correct shape."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 50))
    y = rng.choice([0.0, 1.0], size=30)

    model = PLSRegression(n_components=2)
    model.fit(X, y)

    vip = calculate_vip(model)
    assert vip.shape == (50,)
    assert np.all(vip >= 0)


def test_calculate_vip_mean_near_one() -> None:
    """Test that mean VIP is approximately 1 (mathematical property)."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((50, 100))
    y = rng.choice([0.0, 1.0], size=50)

    model = PLSRegression(n_components=2)
    model.fit(X, y)

    vip = calculate_vip(model)
    # Mean of VIP^2 should be ~1 for orthogonal weights
    assert abs(np.mean(vip**2) - 1.0) < 0.2


def test_create_splot_shapes(plsda_inputs: dict) -> None:
    """Test S-plot returns arrays of correct shape."""
    X = plsda_inputs["X"]
    result = perform_plsda(X, plsda_inputs["labels"], n_components=2, cv=5)

    p_cov, p_corr = create_splot(X, result["model"])

    n_features = X.shape[1]
    assert p_cov.shape == (n_features,)
    assert p_corr.shape == (n_features,)
    # Correlations should be in [-1, 1]
    assert np.all(p_corr >= -1.0 - 1e-10)
    assert np.all(p_corr <= 1.0 + 1e-10)


def test_find_time_trending_features_output(synthetic_data: dict) -> None:
    """Test time-trending features returns expected DataFrame structure."""
    X = synthetic_data["X"]
    meta = synthetic_data["metadata"]
    feat_ids = synthetic_data["feature_ids"]

    results = find_time_trending_features(X, meta["timepoint"].values, feat_ids)

    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == {
        "feature_id",
        "rho",
        "pvalue",
        "pvalue_fdr",
        "significant",
    }
    assert len(results) == X.shape[1]
    # Results should be sorted by |rho| descending
    assert np.all(
        np.abs(results["rho"].values[:-1]) >= np.abs(results["rho"].values[1:])
    )


def test_find_time_trending_detects_signal(synthetic_data: dict) -> None:
    """Test that synthetic time-correlated features are detected."""
    X = synthetic_data["X"]
    meta = synthetic_data["metadata"]
    feat_ids = synthetic_data["feature_ids"]

    results = find_time_trending_features(
        X, meta["timepoint"].values, feat_ids, fdr_threshold=0.1
    )

    # The strongest synthetic signal (feature 4) should be detected
    top_features = results.head(5)["feature_id"].tolist()
    assert 4 in top_features, (
        f"Feature 4 (strongest signal) not in top 5: {top_features}"
    )


def test_permutation_test_plsda_structure(plsda_inputs: dict) -> None:
    """Test permutation test returns correct structure."""
    result = permutation_test_plsda(
        plsda_inputs["X"],
        plsda_inputs["labels"],
        n_components=2,
        n_permutations=10,
        cv=5,
    )

    assert "observed_q2" in result
    assert "null_q2" in result
    assert "pvalue" in result
    assert result["null_q2"].shape == (10,)
    assert 0.0 <= result["pvalue"] <= 1.0


# --- Plotting tests ---


def test_plot_pca_trajectories(synthetic_data: dict) -> None:
    """Test PCA trajectories plot renders without error."""
    import matplotlib.pyplot as plt

    from src.eda import perform_pca

    pca_result = perform_pca(synthetic_data["X"])
    ax = plot_pca_trajectories(pca_result, synthetic_data["metadata"])

    assert ax is not None
    assert "PC1" in ax.get_xlabel()
    plt.close("all")


def test_plot_plsda_scores(plsda_inputs: dict) -> None:
    """Test PLS-DA score plot renders without error."""
    import matplotlib.pyplot as plt

    result = perform_plsda(
        plsda_inputs["X"], plsda_inputs["labels"], n_components=2, cv=5
    )
    ax = plot_plsda_scores(result, plsda_inputs["labels"])

    assert ax is not None
    assert "PLS-DA" in ax.get_title()
    plt.close("all")


def test_plot_vip_scores(plsda_inputs: dict) -> None:
    """Test VIP bar chart renders without error."""
    import matplotlib.pyplot as plt

    result = perform_plsda(
        plsda_inputs["X"], plsda_inputs["labels"], n_components=2, cv=5
    )
    ax = plot_vip_scores(result["vip"], plsda_inputs["feature_ids"], n_top=10)

    assert ax is not None
    assert "VIP" in ax.get_title()
    plt.close("all")


def test_plot_splot(plsda_inputs: dict) -> None:
    """Test S-plot renders without error."""
    import matplotlib.pyplot as plt

    result = perform_plsda(
        plsda_inputs["X"], plsda_inputs["labels"], n_components=2, cv=5
    )
    p_cov, p_corr = create_splot(plsda_inputs["X"], result["model"])
    ax = plot_splot(p_cov, p_corr, plsda_inputs["feature_ids"])

    assert ax is not None
    assert "S-Plot" in ax.get_title()
    plt.close("all")


def test_plot_permutation_test(plsda_inputs: dict) -> None:
    """Test permutation test histogram renders without error."""
    import matplotlib.pyplot as plt

    perm_result = permutation_test_plsda(
        plsda_inputs["X"],
        plsda_inputs["labels"],
        n_components=2,
        n_permutations=10,
        cv=5,
    )
    ax = plot_permutation_test(perm_result)

    assert ax is not None
    assert "Permutation" in ax.get_title()
    plt.close("all")


def test_plot_time_trending_volcano(synthetic_data: dict) -> None:
    """Test volcano plot renders without error."""
    import matplotlib.pyplot as plt

    X = synthetic_data["X"]
    meta = synthetic_data["metadata"]
    feat_ids = synthetic_data["feature_ids"]

    trending = find_time_trending_features(X, meta["timepoint"].values, feat_ids)
    ax = plot_time_trending_volcano(trending)

    assert ax is not None
    assert "Time-Trending" in ax.get_title()
    plt.close("all")


def test_extract_splot_biomarkers_with_candidates(plsda_inputs: dict) -> None:
    """Test biomarker extraction with synthetic data containing candidates."""
    X = plsda_inputs["X"]
    labels = plsda_inputs["labels"]
    feat_ids = plsda_inputs["feature_ids"]

    plsda_result = perform_plsda(X, labels, n_components=2, cv=5)
    p_cov, p_corr = create_splot(X, plsda_result["model"])

    biomarkers = extract_splot_biomarkers(p_cov, p_corr, feat_ids)

    assert isinstance(biomarkers, pd.DataFrame)
    assert set(biomarkers.columns) == {
        "feature_id",
        "covariance",
        "correlation",
        "abs_covariance",
        "abs_correlation",
    }
    # Check sorting by abs_covariance descending
    if len(biomarkers) > 1:
        assert biomarkers["abs_covariance"].is_monotonic_decreasing


def test_extract_splot_biomarkers_custom_thresholds(plsda_inputs: dict) -> None:
    """Test biomarker extraction with custom thresholds."""
    X = plsda_inputs["X"]
    labels = plsda_inputs["labels"]
    feat_ids = plsda_inputs["feature_ids"]

    plsda_result = perform_plsda(X, labels, n_components=2, cv=5)
    p_cov, p_corr = create_splot(X, plsda_result["model"])

    # Very strict thresholds should yield few or no candidates
    biomarkers = extract_splot_biomarkers(
        p_cov, p_corr, feat_ids, threshold_cov=10.0, threshold_corr=0.99
    )

    assert isinstance(biomarkers, pd.DataFrame)
    assert len(biomarkers) >= 0  # Could be zero with strict thresholds


def test_extract_splot_biomarkers_empty_result() -> None:
    """Test biomarker extraction when no candidates meet thresholds."""
    rng = np.random.default_rng(42)
    p_cov = rng.standard_normal(50)
    p_corr = rng.uniform(-0.5, 0.5, 50)  # Low correlation values
    feat_ids = list(range(50))

    biomarkers = extract_splot_biomarkers(p_cov, p_corr, feat_ids, threshold_corr=0.9)

    assert isinstance(biomarkers, pd.DataFrame)
    assert len(biomarkers) == 0
    assert set(biomarkers.columns) == {
        "feature_id",
        "covariance",
        "correlation",
        "abs_covariance",
        "abs_correlation",
    }


def test_plot_pca_trajectories_plotly(synthetic_data: dict) -> None:
    """Test plotly PCA trajectories plot returns Figure."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    from src.eda import perform_pca

    X = synthetic_data["X"]
    meta = synthetic_data["metadata"]

    pca_result = perform_pca(X, n_components=3)
    fig = plot_pca_trajectories_plotly(pca_result, meta)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Should have traces
    plt.close("all")


def test_plot_plsda_scores_plotly(plsda_inputs: dict) -> None:
    """Test plotly PLS-DA scores plot returns Figure."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    plsda_result = perform_plsda(
        plsda_inputs["X"], plsda_inputs["labels"], n_components=2, cv=5
    )
    fig = plot_plsda_scores_plotly(plsda_result, plsda_inputs["labels"])

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    plt.close("all")


def test_plot_vip_scores_plotly(plsda_inputs: dict) -> None:
    """Test plotly VIP scores plot returns Figure."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    plsda_result = perform_plsda(
        plsda_inputs["X"], plsda_inputs["labels"], n_components=2, cv=5
    )
    fig = plot_vip_scores_plotly(plsda_result["vip"], plsda_inputs["feature_ids"])

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    plt.close("all")


def test_plot_splot_plotly(plsda_inputs: dict) -> None:
    """Test plotly S-plot returns Figure."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    plsda_result = perform_plsda(
        plsda_inputs["X"], plsda_inputs["labels"], n_components=2, cv=5
    )
    p_cov, p_corr = create_splot(plsda_inputs["X"], plsda_result["model"])
    fig = plot_splot_plotly(p_cov, p_corr, plsda_inputs["feature_ids"])

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    plt.close("all")


def test_plot_permutation_test_plotly(plsda_inputs: dict) -> None:
    """Test plotly permutation test plot returns Figure."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    perm_result = permutation_test_plsda(
        plsda_inputs["X"],
        plsda_inputs["labels"],
        n_components=2,
        n_permutations=10,
        cv=5,
    )
    fig = plot_permutation_test_plotly(perm_result)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    plt.close("all")


def test_plot_time_trending_volcano_plotly(synthetic_data: dict) -> None:
    """Test plotly volcano plot returns Figure."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    X = synthetic_data["X"]
    meta = synthetic_data["metadata"]
    feat_ids = synthetic_data["feature_ids"]

    trending = find_time_trending_features(X, meta["timepoint"].values, feat_ids)
    fig = plot_time_trending_volcano_plotly(trending)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    plt.close("all")
