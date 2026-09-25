import numpy as np
import pytest

from src.terrain_error_bound import (
    calculate_plane_residuals,
    calculate_terrain_error_metrics,
    decide_terrain_error_bound,
)


def make_flat_points():
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_plane_residuals_for_flat_plane():
    points = make_flat_points()

    residuals = calculate_plane_residuals(
        points
    )

    assert residuals.shape == (4,)

    assert np.allclose(
        residuals,
        0.0,
        atol=1e-6,
    )


def test_plane_residuals_for_tilted_plane():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )

    residuals = calculate_plane_residuals(
        points
    )

    assert residuals.shape == (4,)

    assert np.all(
        residuals >= 0.0
    )


def test_flat_region_has_low_error():
    points = make_flat_points()

    metrics = calculate_terrain_error_metrics(
        points
    )

    assert metrics["geometry_status"] == "valid"
    assert metrics["point_count"] == 4

    assert metrics["max_plane_residual"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert metrics["rms_plane_residual"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert metrics["elevation_variation"] == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_sparse_region_is_reported():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    metrics = calculate_terrain_error_metrics(
        points
    )

    assert metrics["geometry_status"] == "sparse"
    assert metrics["point_count"] == 2
    assert metrics["max_plane_residual"] is None


def test_high_elevation_variation_requests_split():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = decide_terrain_error_bound(
        points=points,
        max_plane_residual=0.01,
        max_elevation_variation=0.5,
    )

    assert result["split"] is True

    assert (
        "elevation variation exceeds threshold"
        in result["reason"]
    )


def test_low_error_region_does_not_split():
    points = make_flat_points()

    result = decide_terrain_error_bound(
        points=points,
        max_plane_residual=0.01,
        max_elevation_variation=0.5,
    )

    assert result["split"] is False

    assert result["reason"] == (
        "terrain error within bounds"
    )


def test_point_count_can_request_split():
    points = make_flat_points()

    result = decide_terrain_error_bound(
        points=points,
        max_plane_residual=0.01,
        max_elevation_variation=0.5,
        max_points_per_region=3,
    )

    assert result["split"] is True

    assert (
        "point count exceeds threshold"
        in result["reason"]
    )


def test_sparse_region_does_not_split():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = decide_terrain_error_bound(
        points=points,
        max_plane_residual=0.01,
        max_elevation_variation=0.5,
    )

    assert result["split"] is False

    assert result["metrics"]["geometry_status"] == (
        "sparse"
    )


def test_invalid_residual_threshold():
    points = make_flat_points()

    with pytest.raises(ValueError):
        decide_terrain_error_bound(
            points=points,
            max_plane_residual=0.0,
            max_elevation_variation=0.5,
        )


def test_invalid_elevation_threshold():
    points = make_flat_points()

    with pytest.raises(ValueError):
        decide_terrain_error_bound(
            points=points,
            max_plane_residual=0.01,
            max_elevation_variation=0.0,
        )