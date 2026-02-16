"""Tests for src.location_comparison module."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.location_comparison import (
    compare_drug_detection,
    compare_pk_parameters,
    compare_plsda_performance,
    compute_feature_overlap,
    generate_recommendation,
    match_features_across_locations,
    plot_detection_comparison,
    plot_feature_overlap,
    plot_model_comparison,
    plot_pk_comparison_locations,
    plot_recommendation_summary,
)


@pytest.fixture()
def synthetic_feature_meta_pair() -> dict:
    """Create two synthetic feature metadata DataFrames with known overlapping features."""
    rng = np.random.default_rng(42)

    # Dataset A: 50 features
    n_a = 50
    mz_a = 100.0 + rng.uniform(0, 500, n_a)
    rt_a = rng.uniform(1, 20, n_a)
    meta_a = pd.DataFrame(
        {"row m/z": mz_a, "row retention time": rt_a},
        index=np.arange(1000, 1000 + n_a),
    )

    # Dataset B: 40 features, 20 deliberately matching A (with small noise)
    n_b = 40
    n_shared = 20
    mz_b_shared = mz_a[:n_shared] + rng.uniform(-0.0005, 0.0005, n_shared)
    rt_b_shared = rt_a[:n_shared] + rng.uniform(-0.3, 0.3, n_shared)
    mz_b_unique = 100.0 + rng.uniform(0, 500, n_b - n_shared)
    rt_b_unique = rng.uniform(1, 20, n_b - n_shared)

    mz_b = np.concatenate([mz_b_shared, mz_b_unique])
    rt_b = np.concatenate([rt_b_shared, rt_b_unique])
    meta_b = pd.DataFrame(
        {"row m/z": mz_b, "row retention time": rt_b},
        index=np.arange(2000, 2000 + n_b),
    )

    return {"meta_a": meta_a, "meta_b": meta_b, "n_shared": n_shared}


@pytest.fixture()
def synthetic_location_data() -> dict:
    """Create synthetic data for full location comparison pipeline."""
    rng = np.random.default_rng(123)

    n_features_a = 30
    n_features_b = 25
    n_subjects = 4
    timepoints = [0, 60, 120, 240]

    def make_metadata(n_subjects: int, timepoints: list[int]) -> pd.DataFrame:
        rows = []
        for s in range(1, n_subjects + 1):
            for tp in timepoints:
                for st in ["plasma", "skin"]:
                    rows.append(
                        {
                            "filename": f"S{s}_T{tp}_{st}.mzML",
                            "subject": f"S{s}",
                            "timepoint": tp,
                            "sample_type": st,
                        }
                    )
        return pd.DataFrame(rows)

    meta_a = make_metadata(n_subjects, timepoints)
    meta_b = make_metadata(n_subjects, timepoints)

    # Feature metadata with DPH-like feature in both
    mz_a = rng.uniform(100, 500, n_features_a)
    mz_a[0] = 256.1696  # DPH
    rt_a = rng.uniform(2, 18, n_features_a)
    feat_meta_a = pd.DataFrame(
        {"row m/z": mz_a, "row retention time": rt_a},
        index=np.arange(100, 100 + n_features_a),
    )

    mz_b = rng.uniform(100, 500, n_features_b)
    mz_b[0] = 256.1697  # DPH with small offset
    rt_b = rng.uniform(2, 18, n_features_b)
    feat_meta_b = pd.DataFrame(
        {"row m/z": mz_b, "row retention time": rt_b},
        index=np.arange(200, 200 + n_features_b),
    )

    # Peak areas (features x samples)
    peak_areas_a = pd.DataFrame(
        rng.uniform(1000, 50000, (n_features_a, len(meta_a))),
        index=feat_meta_a.index,
        columns=meta_a["filename"].values,
    )
    peak_areas_b = pd.DataFrame(
        rng.uniform(1000, 50000, (n_features_b, len(meta_b))),
        index=feat_meta_b.index,
        columns=meta_b["filename"].values,
    )

    # Preprocessed X matrices (samples x features)
    X_a = rng.standard_normal((len(meta_a), n_features_a))
    X_b = rng.standard_normal((len(meta_b), n_features_b))

    # Add time signal to first 3 features
    for j in range(3):
        X_a[:, j] += meta_a["timepoint"].values * 0.01 * (j + 1)
        X_b[:, j] += meta_b["timepoint"].values * 0.01 * (j + 1)

    return {
        "feat_meta_a": feat_meta_a,
        "feat_meta_b": feat_meta_b,
        "peak_areas_a": peak_areas_a,
        "peak_areas_b": peak_areas_b,
        "metadata_a": meta_a,
        "metadata_b": meta_b,
        "X_a": X_a,
        "X_b": X_b,
    }


# --- Feature matching tests ---


def test_match_features_returns_dataframe(synthetic_feature_meta_pair: dict) -> None:
    """Test that match_features_across_locations returns a DataFrame."""
    pair = synthetic_feature_meta_pair
    matches = match_features_across_locations(pair["meta_a"], pair["meta_b"])
    assert isinstance(matches, pd.DataFrame)
    expected_cols = {
        "feature_id_a",
        "feature_id_b",
        "mz_a",
        "mz_b",
        "rt_a",
        "rt_b",
        "mz_ppm_error",
        "rt_diff_min",
    }
    assert set(matches.columns) == expected_cols


def test_match_features_finds_shared(synthetic_feature_meta_pair: dict) -> None:
    """Test that matching finds approximately the expected number of shared features."""
    pair = synthetic_feature_meta_pair
    matches = match_features_across_locations(pair["meta_a"], pair["meta_b"])
    # We planted 20 shared features; expect most to be found
    assert len(matches) >= 10
    assert len(matches) <= 30  # should not find more than shared + noise


def test_match_features_respects_tolerance(
    synthetic_feature_meta_pair: dict,
) -> None:
    """Test that all matches are within the specified tolerances."""
    pair = synthetic_feature_meta_pair
    tol_ppm = 15.0
    tol_rt = 0.5
    matches = match_features_across_locations(
        pair["meta_a"], pair["meta_b"], tol_ppm, tol_rt
    )
    if not matches.empty:
        assert matches["mz_ppm_error"].max() <= tol_ppm
        assert matches["rt_diff_min"].max() <= tol_rt


def test_match_features_empty_input() -> None:
    """Test matching with empty DataFrames returns empty result."""
    empty = pd.DataFrame(columns=["row m/z", "row retention time"])
    nonempty = pd.DataFrame(
        {"row m/z": [100.0], "row retention time": [5.0]}, index=[1]
    )
    result = match_features_across_locations(empty, nonempty)
    assert len(result) == 0


# --- Overlap tests ---


def test_compute_feature_overlap_keys(synthetic_feature_meta_pair: dict) -> None:
    """Test that compute_feature_overlap returns all expected keys."""
    pair = synthetic_feature_meta_pair
    overlap = compute_feature_overlap(pair["meta_a"], pair["meta_b"])

    expected_keys = {
        "n_a",
        "n_b",
        "n_matched_a",
        "n_matched_b",
        "n_unique_a",
        "n_unique_b",
        "overlap_fraction_a",
        "overlap_fraction_b",
        "matches",
    }
    assert set(overlap.keys()) == expected_keys


def test_compute_feature_overlap_counts(synthetic_feature_meta_pair: dict) -> None:
    """Test that overlap counts are consistent."""
    pair = synthetic_feature_meta_pair
    overlap = compute_feature_overlap(pair["meta_a"], pair["meta_b"])

    assert overlap["n_a"] == 50
    assert overlap["n_b"] == 40
    assert overlap["n_matched_a"] + overlap["n_unique_a"] == overlap["n_a"]
    assert overlap["n_matched_b"] + overlap["n_unique_b"] == overlap["n_b"]
    assert 0.0 <= overlap["overlap_fraction_a"] <= 1.0
    assert 0.0 <= overlap["overlap_fraction_b"] <= 1.0


# --- Drug detection comparison tests ---


def test_compare_drug_detection_columns(synthetic_location_data: dict) -> None:
    """Test that compare_drug_detection returns expected columns."""
    data = synthetic_location_data
    target = {"TestDrug": 256.1696}
    result = compare_drug_detection(
        data["feat_meta_a"],
        data["feat_meta_b"],
        data["peak_areas_a"],
        data["peak_areas_b"],
        data["metadata_a"],
        data["metadata_b"],
        target,
    )
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "compound",
        "location",
        "n_matches",
        "best_ppm_error",
        "mean_skin_intensity",
        "detected",
    }
    assert set(result.columns) == expected_cols
    assert len(result) == 2  # 1 compound x 2 locations


# --- PK comparison tests ---


def test_compare_pk_parameters_structure(synthetic_location_data: dict) -> None:
    """Test that compare_pk_parameters returns correct structure."""
    data = synthetic_location_data
    result = compare_pk_parameters(
        data["peak_areas_a"],
        data["peak_areas_b"],
        data["metadata_a"],
        data["metadata_b"],
        data["feat_meta_a"],
        data["feat_meta_b"],
        target_mz=256.1696,
    )
    assert "location_a" in result
    assert "location_b" in result
    assert "summary" in result
    assert isinstance(result["summary"], pd.DataFrame)
    assert "cmax_corr_r" in result["summary"].columns


# --- PLS-DA comparison tests ---


def test_compare_plsda_performance_summary(synthetic_location_data: dict) -> None:
    """Test that compare_plsda_performance returns valid summary."""
    data = synthetic_location_data
    # Filter to skin-only
    skin_a = data["metadata_a"]["sample_type"].values == "skin"
    skin_b = data["metadata_b"]["sample_type"].values == "skin"

    result = compare_plsda_performance(
        data["X_a"][skin_a],
        data["X_b"][skin_b],
        data["metadata_a"][skin_a].reset_index(drop=True),
        data["metadata_b"][skin_b].reset_index(drop=True),
        time_threshold=60.0,
        n_components=2,
        cv=3,
    )

    assert "plsda_a" in result
    assert "plsda_b" in result
    assert "labels_a" in result
    assert "labels_b" in result
    assert "summary" in result
    summary = result["summary"]
    assert set(summary.columns) == {"location", "r2y", "q2", "q2_std"}
    assert len(summary) == 2


# --- Recommendation tests ---


def test_generate_recommendation_keys(synthetic_location_data: dict) -> None:
    """Test that generate_recommendation returns all expected keys."""
    data = synthetic_location_data
    skin_a = data["metadata_a"]["sample_type"].values == "skin"
    skin_b = data["metadata_b"]["sample_type"].values == "skin"

    overlap = compute_feature_overlap(data["feat_meta_a"], data["feat_meta_b"])
    detection = compare_drug_detection(
        data["feat_meta_a"],
        data["feat_meta_b"],
        data["peak_areas_a"],
        data["peak_areas_b"],
        data["metadata_a"],
        data["metadata_b"],
        {"DPH": 256.1696},
    )
    pk = compare_pk_parameters(
        data["peak_areas_a"],
        data["peak_areas_b"],
        data["metadata_a"],
        data["metadata_b"],
        data["feat_meta_a"],
        data["feat_meta_b"],
        target_mz=256.1696,
    )
    plsda = compare_plsda_performance(
        data["X_a"][skin_a],
        data["X_b"][skin_b],
        data["metadata_a"][skin_a].reset_index(drop=True),
        data["metadata_b"][skin_b].reset_index(drop=True),
        time_threshold=60.0,
        n_components=2,
        cv=3,
    )

    rec = generate_recommendation(overlap, detection, pk, plsda)

    assert set(rec.keys()) == {"scores", "recommended", "rationale", "raw_metrics"}
    assert rec["recommended"] in {"forearm", "forehead"}
    assert isinstance(rec["scores"], pd.DataFrame)
    assert "total" in rec["scores"].columns


# --- Plotting tests ---


def test_plot_feature_overlap(synthetic_feature_meta_pair: dict) -> None:
    """Test feature overlap plot renders without error."""
    import matplotlib.pyplot as plt

    pair = synthetic_feature_meta_pair
    overlap = compute_feature_overlap(pair["meta_a"], pair["meta_b"])
    ax = plot_feature_overlap(overlap)

    assert ax is not None
    assert "Overlap" in ax.get_title()
    plt.close("all")


def test_plot_detection_comparison(synthetic_location_data: dict) -> None:
    """Test detection comparison plot renders without error."""
    import matplotlib.pyplot as plt

    data = synthetic_location_data
    detection = compare_drug_detection(
        data["feat_meta_a"],
        data["feat_meta_b"],
        data["peak_areas_a"],
        data["peak_areas_b"],
        data["metadata_a"],
        data["metadata_b"],
        {"DPH": 256.1696},
    )
    ax = plot_detection_comparison(detection)

    assert ax is not None
    assert "Detection" in ax.get_title()
    plt.close("all")


def test_plot_pk_comparison_locations(synthetic_location_data: dict) -> None:
    """Test PK comparison location plot renders without error."""
    import matplotlib.pyplot as plt

    data = synthetic_location_data
    pk = compare_pk_parameters(
        data["peak_areas_a"],
        data["peak_areas_b"],
        data["metadata_a"],
        data["metadata_b"],
        data["feat_meta_a"],
        data["feat_meta_b"],
        target_mz=256.1696,
    )
    fig = plot_pk_comparison_locations(pk)

    assert fig is not None
    plt.close("all")


def test_plot_model_comparison(synthetic_location_data: dict) -> None:
    """Test model comparison plot renders without error."""
    import matplotlib.pyplot as plt

    data = synthetic_location_data
    skin_a = data["metadata_a"]["sample_type"].values == "skin"
    skin_b = data["metadata_b"]["sample_type"].values == "skin"

    plsda = compare_plsda_performance(
        data["X_a"][skin_a],
        data["X_b"][skin_b],
        data["metadata_a"][skin_a].reset_index(drop=True),
        data["metadata_b"][skin_b].reset_index(drop=True),
        time_threshold=60.0,
        n_components=2,
        cv=3,
    )
    fig = plot_model_comparison(plsda)

    assert fig is not None
    plt.close("all")


def test_plot_recommendation_summary(synthetic_location_data: dict) -> None:
    """Test recommendation summary plot renders without error."""
    import matplotlib.pyplot as plt

    data = synthetic_location_data
    skin_a = data["metadata_a"]["sample_type"].values == "skin"
    skin_b = data["metadata_b"]["sample_type"].values == "skin"

    overlap = compute_feature_overlap(data["feat_meta_a"], data["feat_meta_b"])
    detection = compare_drug_detection(
        data["feat_meta_a"],
        data["feat_meta_b"],
        data["peak_areas_a"],
        data["peak_areas_b"],
        data["metadata_a"],
        data["metadata_b"],
        {"DPH": 256.1696},
    )
    pk = compare_pk_parameters(
        data["peak_areas_a"],
        data["peak_areas_b"],
        data["metadata_a"],
        data["metadata_b"],
        data["feat_meta_a"],
        data["feat_meta_b"],
        target_mz=256.1696,
    )
    plsda = compare_plsda_performance(
        data["X_a"][skin_a],
        data["X_b"][skin_b],
        data["metadata_a"][skin_a].reset_index(drop=True),
        data["metadata_b"][skin_b].reset_index(drop=True),
        time_threshold=60.0,
        n_components=2,
        cv=3,
    )
    rec = generate_recommendation(overlap, detection, pk, plsda)
    ax = plot_recommendation_summary(rec)

    assert ax is not None
    assert "Recommendation" in ax.get_title()
    plt.close("all")
