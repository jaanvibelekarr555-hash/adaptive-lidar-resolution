import numpy as np

from src.road_geometry import (
    analyze_road_cell,
    calculate_height_variation,
    classify_road_cell,
)


def test_flat_road_has_small_height_variation():
    points = np.array(
        [
            [1.0, 1.0, 1.00, 0.5],
            [1.1, 1.0, 1.01, 0.5],
            [1.0, 1.1, 1.02, 0.5],
            [1.1, 1.1, 1.01, 0.5],
        ],
        dtype=np.float32,
    )

    variation = calculate_height_variation(points)

    assert np.isclose(variation, 0.02)


def test_uneven_road_has_large_height_variation():
    points = np.array(
        [
            [1.0, 1.0, 1.00, 0.5],
            [1.1, 1.0, 1.01, 0.5],
            [1.0, 1.1, 0.75, 0.5],
            [1.1, 1.1, 1.02, 0.5],
        ],
        dtype=np.float32,
    )

    variation = calculate_height_variation(points)

    assert np.isclose(variation, 0.27)


def test_flat_road_cell_is_drivable():
    points = np.array(
        [
            [1.0, 1.0, 1.00, 0.5],
            [1.1, 1.0, 1.01, 0.5],
            [1.0, 1.1, 1.02, 0.5],
            [1.1, 1.1, 1.01, 0.5],
        ],
        dtype=np.float32,
    )

    result = classify_road_cell(
        points,
        height_threshold=0.15,
    )

    assert result == "drivable"


def test_uneven_road_cell_is_non_drivable():
    points = np.array(
        [
            [1.0, 1.0, 1.00, 0.5],
            [1.1, 1.0, 1.01, 0.5],
            [1.0, 1.1, 0.75, 0.5],
            [1.1, 1.1, 1.02, 0.5],
        ],
        dtype=np.float32,
    )

    result = classify_road_cell(
        points,
        height_threshold=0.15,
    )

    assert result == "non-drivable"


def test_analyze_road_cell_returns_cell_information():
    points = np.array(
        [
            [1.0, 1.0, 1.00, 0.5],
            [1.1, 1.0, 1.01, 0.5],
            [1.0, 1.1, 1.02, 0.5],
            [1.1, 1.1, 1.01, 0.5],
            [1.05, 1.05, 1.00, 0.5],
        ],
        dtype=np.float32,
    )

    result = analyze_road_cell(
        points,
        cell_x=3,
        cell_y=4,
    )

    assert result["cell_x"] == 3
    assert result["cell_y"] == 4
    assert result["point_count"] == 5
    assert np.isclose(result["height_variation"], 0.02)
    assert result["classification"] == "drivable"