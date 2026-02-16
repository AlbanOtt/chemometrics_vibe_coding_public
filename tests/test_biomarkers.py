"""Tests for src.biomarkers module."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.biomarkers import (
    correlate_skin_features_to_target,
    pair_skin_plasma_samples,
    plot_biomarker_summary,
    plot_candidate_heatmap,
    plot_correlation_volcano,
    plot_predicted_vs_observed,
    pls_regression_loso_cv,
    run_biomarker_discovery,
    select_biomarker_candidates,
)


@pytest.fixture()
def synthetic_paired_data() -> dict:
    """Create synthetic skin-plasma paired metabolomics data.

    5 subjects, 6 timepoints, 50 features. One DPH-like feature at m/z 256.1696,
    3 features intentionally correlated with DPH PK curve.
    """
    rng = np.random.default_rng(42)
    n_subjects = 5
    timepoints = [0, 30, 60, 120, 240, 360]
    n_features = 50

    # Generate DPH-like PK curve (peak at 60-120 min, decay)
    dph_curve = np.array([0.0, 100.0, 150.0, 120.0, 60.0, 20.0])  # Simple PK profile

    # Create metadata and data
    plasma_meta = []
    skin_meta = []
    plasma_data = []
    skin_data = []

    for subject_idx in range(1, n_subjects + 1):
        subject_id = f"S{subject_idx}"
        for tp in timepoints:
            # Plasma sample
            plasma_meta.append(
                {
                    "filename": f"plasma_{subject_id}_t{tp}.mzML",
                    "subject": subject_id,
                    "timepoint": tp,
                    "sample_type": "plasma",
                }
            )
            # Skin sample
            skin_meta.append(
                {
                    "filename": f"skin_{subject_id}_t{tp}.mzML",
                    "subject": subject_id,
                    "timepoint": tp,
                    "sample_type": "skin",
                }
            )

            # Generate plasma DPH intensity with subject variability
            tp_idx = timepoints.index(tp)
            plasma_dph = dph_curve[tp_idx] * (0.8 + 0.4 * rng.random())
            plasma_features = rng.standard_normal(n_features - 1)
            plasma_data.append(np.concatenate([[plasma_dph], plasma_features]))

            # Generate skin data: 3 features correlated with plasma DPH
            skin_features = rng.standard_normal(n_features)
            # Features 0, 1, 2 are correlated with plasma DPH
            skin_features[0] = plasma_dph * 0.7 + rng.standard_normal() * 10
            skin_features[1] = plasma_dph * 0.6 + rng.standard_normal() * 15
            skin_features[2] = plasma_dph * 0.5 + rng.standard_normal() * 20
            skin_data.append(skin_features)

    metadata = pd.DataFrame(plasma_meta + skin_meta)

    # Create feature matrix (all samples x features)
    X_all = np.vstack([plasma_data, skin_data])
    X_processed = (X_all - np.mean(X_all, axis=0)) / (
        np.std(X_all, axis=0, ddof=1) + 1e-10
    )

    # Create peak areas raw (features x samples)
    peak_areas_raw = pd.DataFrame(
        X_all.T, columns=[row["filename"] for row in plasma_meta + skin_meta]
    )
    peak_areas_raw.index.name = "row ID"

    # Create feature metadata
    feature_metadata = pd.DataFrame(
        {
            "row m/z": [256.1696] + list(200 + rng.random(n_features - 1) * 100),
            "row retention time": 5.0 + rng.random(n_features) * 10,
        }
    )
    feature_metadata.index.name = "row ID"

    feature_ids = list(range(n_features))

    return {
        "X_processed": X_processed,
        "metadata": metadata,
        "feature_metadata": feature_metadata,
        "feature_ids": feature_ids,
        "peak_areas_raw": peak_areas_raw,
        "n_subjects": n_subjects,
        "n_timepoints": len(timepoints),
    }


# ---------------------------------------------------------------------------
# Pairing tests
# ---------------------------------------------------------------------------


def test_pair_skin_plasma_samples_status(synthetic_paired_data: dict) -> None:
    """Test that pairing returns success status when DPH is found."""
    result = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    assert result["status"] == "success"
    assert result["dph_feature_id"] is not None


def test_pair_skin_plasma_samples_n_pairs(synthetic_paired_data: dict) -> None:
    """Test that pairing produces the expected number of pairs."""
    result = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    expected_pairs = (
        synthetic_paired_data["n_subjects"] * synthetic_paired_data["n_timepoints"]
    )
    assert result["n_pairs"] == expected_pairs


def test_pair_skin_plasma_samples_shapes(synthetic_paired_data: dict) -> None:
    """Test that pairing output arrays have consistent shapes."""
    result = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    n_pairs = result["n_pairs"]
    assert len(result["skin_indices"]) == n_pairs
    assert len(result["y_plasma_dph"]) == n_pairs
    assert len(result["subjects"]) == n_pairs
    assert len(result["timepoints"]) == n_pairs


def test_pair_skin_plasma_samples_dph_not_found(synthetic_paired_data: dict) -> None:
    """Test pairing handles DPH not found gracefully."""
    result = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=999.9999,  # Nonsense m/z
        tolerance_ppm=10.0,
    )

    assert result["status"] == "dph_not_found"
    assert result["n_pairs"] == 0
    assert result["dph_feature_id"] is None


def test_pair_skin_plasma_samples_y_nonnegative(synthetic_paired_data: dict) -> None:
    """Test that plasma DPH intensities are non-negative (raw peak areas)."""
    result = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    assert np.all(result["y_plasma_dph"] >= 0)


# ---------------------------------------------------------------------------
# Correlation tests
# ---------------------------------------------------------------------------


def test_correlate_skin_features_to_target_columns(
    synthetic_paired_data: dict,
) -> None:
    """Test that correlation results have expected columns."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]

    results = correlate_skin_features_to_target(
        X_skin_paired,
        y_plasma_dph,
        synthetic_paired_data["feature_ids"],
    )

    expected_cols = {
        "feature_id",
        "rho",
        "pvalue",
        "pvalue_fdr",
        "abs_rho",
        "significant",
    }
    assert set(results.columns) == expected_cols


def test_correlate_skin_features_to_target_length(
    synthetic_paired_data: dict,
) -> None:
    """Test that correlation results have one row per feature."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]

    results = correlate_skin_features_to_target(
        X_skin_paired,
        y_plasma_dph,
        synthetic_paired_data["feature_ids"],
    )

    assert len(results) == len(synthetic_paired_data["feature_ids"])


def test_correlate_skin_features_to_target_sort_order(
    synthetic_paired_data: dict,
) -> None:
    """Test that results are sorted by absolute correlation descending."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]

    results = correlate_skin_features_to_target(
        X_skin_paired,
        y_plasma_dph,
        synthetic_paired_data["feature_ids"],
    )

    abs_rho_values = results["abs_rho"].values
    assert np.all(abs_rho_values[:-1] >= abs_rho_values[1:])


def test_correlate_skin_features_to_target_detects_signal(
    synthetic_paired_data: dict,
) -> None:
    """Test that synthetic correlated features appear in top 10."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]

    results = correlate_skin_features_to_target(
        X_skin_paired,
        y_plasma_dph,
        synthetic_paired_data["feature_ids"],
    )

    top_10_ids = results.head(10)["feature_id"].tolist()
    # Features 0, 1, 2 were designed to correlate
    assert 0 in top_10_ids or 1 in top_10_ids or 2 in top_10_ids


# ---------------------------------------------------------------------------
# Candidate selection tests
# ---------------------------------------------------------------------------


def test_select_biomarker_candidates_columns(synthetic_paired_data: dict) -> None:
    """Test that candidate selection returns expected columns."""
    # Create dummy correlation results and VIP scores
    n_features = len(synthetic_paired_data["feature_ids"])
    correlation_results = pd.DataFrame(
        {
            "feature_id": synthetic_paired_data["feature_ids"],
            "rho": np.linspace(-0.8, 0.8, n_features),
            "pvalue": np.full(n_features, 0.01),
            "pvalue_fdr": np.full(n_features, 0.02),
            "abs_rho": np.abs(np.linspace(-0.8, 0.8, n_features)),
            "significant": True,
        }
    )

    vip_scores = np.linspace(0.5, 2.0, n_features)

    candidates = select_biomarker_candidates(
        correlation_results,
        vip_scores,
        synthetic_paired_data["feature_ids"],
    )

    expected_cols = {
        "feature_id",
        "rho",
        "pvalue",
        "pvalue_fdr",
        "abs_rho",
        "significant",
        "vip",
        "candidate",
        "rank_score",
    }
    assert set(candidates.columns) == expected_cols


def test_select_biomarker_candidates_filter_correctness(
    synthetic_paired_data: dict,
) -> None:
    """Test that candidates are filtered correctly by thresholds."""
    n_features = len(synthetic_paired_data["feature_ids"])

    # Create correlation results with known significance
    correlation_results = pd.DataFrame(
        {
            "feature_id": synthetic_paired_data["feature_ids"],
            "rho": np.linspace(-0.8, 0.8, n_features),
            "pvalue": np.full(n_features, 0.01),
            "pvalue_fdr": np.concatenate(
                [np.full(10, 0.01), np.full(n_features - 10, 0.1)]
            ),  # 10 significant
            "abs_rho": np.abs(np.linspace(-0.8, 0.8, n_features)),
            "significant": np.concatenate(
                [np.full(10, True), np.full(n_features - 10, False)]
            ),
        }
    )

    # VIP: half above threshold, half below
    vip_scores = np.concatenate([np.full(25, 1.5), np.full(25, 0.5)])

    candidates = select_biomarker_candidates(
        correlation_results,
        vip_scores,
        synthetic_paired_data["feature_ids"],
        fdr_threshold=0.05,
        vip_threshold=1.0,
        correlation_threshold=0.5,
    )

    # Check that filters are applied correctly
    for _, row in candidates.iterrows():
        if row["candidate"]:
            assert row["pvalue_fdr"] < 0.05
            assert row["vip"] > 1.0
            assert row["abs_rho"] > 0.5


def test_select_biomarker_candidates_sort_order(
    synthetic_paired_data: dict,
) -> None:
    """Test that candidates are sorted by rank_score descending."""
    n_features = len(synthetic_paired_data["feature_ids"])
    correlation_results = pd.DataFrame(
        {
            "feature_id": synthetic_paired_data["feature_ids"],
            "rho": np.linspace(-0.8, 0.8, n_features),
            "pvalue": np.full(n_features, 0.01),
            "pvalue_fdr": np.full(n_features, 0.02),
            "abs_rho": np.abs(np.linspace(-0.8, 0.8, n_features)),
            "significant": True,
        }
    )

    vip_scores = np.linspace(0.5, 2.0, n_features)

    candidates = select_biomarker_candidates(
        correlation_results,
        vip_scores,
        synthetic_paired_data["feature_ids"],
    )

    rank_scores = candidates["rank_score"].values
    assert np.all(rank_scores[:-1] >= rank_scores[1:])


# ---------------------------------------------------------------------------
# PLS regression tests
# ---------------------------------------------------------------------------


def test_pls_regression_loso_cv_dict_keys(synthetic_paired_data: dict) -> None:
    """Test that PLS regression returns all expected keys."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]
    subjects = pairing["subjects"]

    result = pls_regression_loso_cv(X_skin_paired, y_plasma_dph, subjects)

    expected_keys = {
        "model",
        "r2",
        "q2",
        "rmsecv",
        "bias",
        "vip",
        "y_pred_cv",
        "y_actual",
        "subjects_cv",
        "n_components",
        "n_subjects",
    }
    assert set(result.keys()) == expected_keys


def test_pls_regression_loso_cv_prediction_length(
    synthetic_paired_data: dict,
) -> None:
    """Test that CV predictions have correct length."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]
    subjects = pairing["subjects"]

    result = pls_regression_loso_cv(X_skin_paired, y_plasma_dph, subjects)

    assert len(result["y_pred_cv"]) == len(y_plasma_dph)
    assert len(result["y_actual"]) == len(y_plasma_dph)
    assert len(result["subjects_cv"]) == len(subjects)


def test_pls_regression_loso_cv_r2_positive(synthetic_paired_data: dict) -> None:
    """Test that R² is positive for this synthetic data."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]
    subjects = pairing["subjects"]

    result = pls_regression_loso_cv(X_skin_paired, y_plasma_dph, subjects)

    assert result["r2"] > 0


def test_pls_regression_loso_cv_few_subjects_edge_case(
    synthetic_paired_data: dict,
) -> None:
    """Test PLS with minimal subjects (edge case)."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]
    subjects = pairing["subjects"]

    # Use only first 2 subjects
    mask = np.isin(subjects, ["S1", "S2"])
    X_subset = X_skin_paired[mask]
    y_subset = y_plasma_dph[mask]
    subjects_subset = subjects[mask]

    result = pls_regression_loso_cv(X_subset, y_subset, subjects_subset, n_components=1)

    assert result["n_subjects"] == 2
    assert result["n_components"] == 1


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


def test_run_biomarker_discovery_full_pipeline(
    synthetic_paired_data: dict,
) -> None:
    """Test that full biomarker discovery pipeline runs successfully."""
    result = run_biomarker_discovery(
        synthetic_paired_data["X_processed"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        synthetic_paired_data["feature_ids"],
        synthetic_paired_data["peak_areas_raw"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    assert result["status"] == "success"
    assert "pairing" in result
    assert "correlation" in result
    assert "pls_result" in result
    assert "candidates" in result
    assert result["pairing"]["n_pairs"] > 0
    assert not result["correlation"].empty
    assert "vip" in result["pls_result"]


def test_run_biomarker_discovery_dph_not_found(
    synthetic_paired_data: dict,
) -> None:
    """Test that discovery handles DPH not found gracefully."""
    result = run_biomarker_discovery(
        synthetic_paired_data["X_processed"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        synthetic_paired_data["feature_ids"],
        synthetic_paired_data["peak_areas_raw"],
        target_mz=999.9999,  # Nonsense m/z
        tolerance_ppm=10.0,
    )

    assert result["status"] == "dph_not_found"
    assert result["correlation"].empty
    assert result["pls_result"] == {}
    assert result["candidates"].empty


# ---------------------------------------------------------------------------
# Plot function tests
# ---------------------------------------------------------------------------


def test_plot_correlation_volcano(synthetic_paired_data: dict) -> None:
    """Test that volcano plot is created without errors."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]

    correlation_results = correlate_skin_features_to_target(
        X_skin_paired,
        y_plasma_dph,
        synthetic_paired_data["feature_ids"],
    )

    ax = plot_correlation_volcano(correlation_results)
    assert ax is not None
    plt.close("all")


def test_plot_predicted_vs_observed(synthetic_paired_data: dict) -> None:
    """Test that predicted vs observed plot is created without errors."""
    pairing = pair_skin_plasma_samples(
        synthetic_paired_data["peak_areas_raw"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][pairing["skin_indices"]]
    y_plasma_dph = pairing["y_plasma_dph"]
    subjects = pairing["subjects"]

    pls_result = pls_regression_loso_cv(X_skin_paired, y_plasma_dph, subjects)

    ax = plot_predicted_vs_observed(pls_result, location_name="forearm")
    assert ax is not None
    plt.close("all")


def test_plot_candidate_heatmap(synthetic_paired_data: dict) -> None:
    """Test that candidate heatmap is created without errors."""
    result = run_biomarker_discovery(
        synthetic_paired_data["X_processed"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        synthetic_paired_data["feature_ids"],
        synthetic_paired_data["peak_areas_raw"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    X_skin_paired = synthetic_paired_data["X_processed"][
        result["pairing"]["skin_indices"]
    ]

    ax = plot_candidate_heatmap(
        X_skin_paired,
        result["candidates"],
        result["pairing"],
        synthetic_paired_data["feature_ids"],
        n_top=10,
    )
    assert ax is not None
    plt.close("all")


def test_plot_biomarker_summary(synthetic_paired_data: dict) -> None:
    """Test that summary plot is created without errors."""
    result = run_biomarker_discovery(
        synthetic_paired_data["X_processed"],
        synthetic_paired_data["metadata"],
        synthetic_paired_data["feature_metadata"],
        synthetic_paired_data["feature_ids"],
        synthetic_paired_data["peak_areas_raw"],
        target_mz=256.1696,
        tolerance_ppm=10.0,
    )

    # Add X_skin_paired to result for heatmap panel
    result["X_skin_paired"] = synthetic_paired_data["X_processed"][
        result["pairing"]["skin_indices"]
    ]

    fig = plot_biomarker_summary(result, location_name="forearm")
    assert fig is not None
    plt.close("all")
