import numpy as np

from src.terrain_geometry import (
    fit_local_plane_pca,
    calculate_local_slope,
    calculate_roughness,
    calculate_elevation_variation,
    analyze_terrain_geometry,
)


def test_flat_terrain_has_near_zero_slope():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    slope = calculate_local_slope(points)

    assert slope < 1e-6


def test_tilted_terrain_has_nonzero_slope():
    # z = 0.5x
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.5],
        ],
        dtype=np.float64,
    )

    slope = calculate_local_slope(points)

    expected_slope = np.degrees(
        np.arctan(0.5)
    )

    assert np.isclose(
        slope,
        expected_slope,
        atol=1e-6,
    )


def test_flat_terrain_has_low_roughness():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    roughness = calculate_roughness(points)

    assert roughness < 1e-6


def test_uneven_terrain_has_nonzero_roughness():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.4],
            [0.5, 0.5, 1.3],
        ],
        dtype=np.float64,
    )

    roughness = calculate_roughness(points)

    assert roughness > 0.0


def test_elevation_variation():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.2],
            [0.0, 1.0, 1.5],
        ],
        dtype=np.float64,
    )

    variation = calculate_elevation_variation(points)

    assert np.isclose(
        variation,
        0.5,
    )


def test_analyze_terrain_geometry_for_sparse_cell():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    result = analyze_terrain_geometry(points)

    assert result["geometry_status"] == "sparse"
    assert result["slope"] is None
    assert result["roughness"] is None
    assert result["elevation_variation"] is None


def test_pca_plane_contains_four_coefficients():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    plane = fit_local_plane_pca(points)

    assert plane.shape == (4,)

    a, b, c, d = plane

    assert np.isclose(
        a * 0.0 + b * 0.0 + c * 1.0 + d,
        0.0,
        atol=1e-6,
    )