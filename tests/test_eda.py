"""Tests for exploratory data analysis module."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.eda import plot_pca_scores_interactive


@pytest.fixture()
def synthetic_pca_data() -> dict:
    """Create synthetic PCA result and metadata for testing."""
    rng = np.random.default_rng(42)
    n_samples = 30

    scores = rng.standard_normal((n_samples, 5))
    metadata = pd.DataFrame(
        {
            "subject": [f"S{i % 5}" for i in range(n_samples)],
            "sample_type": ["QC" if i % 3 == 0 else "Study" for i in range(n_samples)],
            "timepoint": [i * 60 for i in range(n_samples)],
        }
    )

    pca_result = {
        "scores": scores,
        "explained_variance": np.array([0.45, 0.25, 0.15, 0.10, 0.05]),
    }

    return {"pca": pca_result, "metadata": metadata}


def test_plot_pca_scores_interactive_basic(synthetic_pca_data: dict) -> None:
    """Test that interactive PCA plot creates a valid Plotly figure."""
    pca_result = synthetic_pca_data["pca"]
    metadata = synthetic_pca_data["metadata"]

    fig = plot_pca_scores_interactive(pca_result, metadata, color_by="sample_type")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Has at least one trace

    # Check axis labels
    assert "PC1" in fig.layout.xaxis.title.text
    assert "45.0%" in fig.layout.xaxis.title.text
    assert "PC2" in fig.layout.yaxis.title.text
    assert "25.0%" in fig.layout.yaxis.title.text


def test_plot_pca_scores_interactive_hover_data(synthetic_pca_data: dict) -> None:
    """Test that hover tooltips contain all required fields."""
    pca_result = synthetic_pca_data["pca"]
    metadata = synthetic_pca_data["metadata"]

    fig = plot_pca_scores_interactive(pca_result, metadata, color_by="sample_type")

    # Check that customdata is present
    for trace in fig.data:
        assert trace.customdata is not None
        assert trace.customdata.shape[1] == 3  # sample_type, subject, timepoint

        # Check hover template includes all fields
        assert "Sample Type" in trace.hovertemplate
        assert "Subject" in trace.hovertemplate
        assert "Timepoint" in trace.hovertemplate
        assert "PC1" in trace.hovertemplate
        assert "PC2" in trace.hovertemplate


def test_plot_pca_scores_interactive_different_pcs(synthetic_pca_data: dict) -> None:
    """Test plotting different PC combinations."""
    pca_result = synthetic_pca_data["pca"]
    metadata = synthetic_pca_data["metadata"]

    fig = plot_pca_scores_interactive(
        pca_result, metadata, color_by="sample_type", pc_x=1, pc_y=2
    )

    # Check axis labels reflect PC2 vs PC3
    assert "PC2" in fig.layout.xaxis.title.text
    assert "25.0%" in fig.layout.xaxis.title.text
    assert "PC3" in fig.layout.yaxis.title.text
    assert "15.0%" in fig.layout.yaxis.title.text

    # Check hover template uses correct PC indices
    for trace in fig.data:
        assert "PC2" in trace.hovertemplate
        assert "PC3" in trace.hovertemplate


def test_plot_pca_scores_interactive_color_by_options(
    synthetic_pca_data: dict,
) -> None:
    """Test coloring by different metadata columns."""
    pca_result = synthetic_pca_data["pca"]
    metadata = synthetic_pca_data["metadata"]

    # Test color by sample_type
    fig1 = plot_pca_scores_interactive(pca_result, metadata, color_by="sample_type")
    assert len(fig1.data) == 2  # QC and Study

    # Test color by subject
    fig2 = plot_pca_scores_interactive(pca_result, metadata, color_by="subject")
    assert len(fig2.data) == 5  # S0, S1, S2, S3, S4

    # Test color by timepoint
    fig3 = plot_pca_scores_interactive(pca_result, metadata, color_by="timepoint")
    assert len(fig3.data) == 30  # Each timepoint unique


def test_plot_pca_scores_interactive_metadata_mismatch(
    synthetic_pca_data: dict,
) -> None:
    """Test error on metadata row count mismatch."""
    pca_result = synthetic_pca_data["pca"]
    metadata = synthetic_pca_data["metadata"].iloc[:10]  # Only 10 rows

    with pytest.raises(
        ValueError, match="Metadata row count .* does not match PCA scores"
    ):
        plot_pca_scores_interactive(pca_result, metadata, color_by="sample_type")


def test_plot_pca_scores_interactive_invalid_color_by(
    synthetic_pca_data: dict,
) -> None:
    """Test error on missing color_by column."""
    pca_result = synthetic_pca_data["pca"]
    metadata = synthetic_pca_data["metadata"]

    with pytest.raises(ValueError, match="Column 'invalid_column' not found"):
        plot_pca_scores_interactive(pca_result, metadata, color_by="invalid_column")


def test_plot_pca_scores_interactive_invalid_pc_indices(
    synthetic_pca_data: dict,
) -> None:
    """Test error on out-of-range PC indices."""
    pca_result = synthetic_pca_data["pca"]
    metadata = synthetic_pca_data["metadata"]

    with pytest.raises(ValueError, match="PC indices .* out of range"):
        plot_pca_scores_interactive(
            pca_result, metadata, color_by="sample_type", pc_x=10, pc_y=1
        )

    with pytest.raises(ValueError, match="PC indices .* out of range"):
        plot_pca_scores_interactive(
            pca_result, metadata, color_by="sample_type", pc_x=0, pc_y=10
        )
