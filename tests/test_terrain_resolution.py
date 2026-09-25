import pytest

from src.terrain_resolution import (
    build_base_resolution_by_cell,
    calculate_cell_base_resolution,
    calculate_cell_distance,
)


def make_cell(x_min, x_max, y_min, y_max):
    return {
        "cell_id": (0, 0),
        "bbox": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "point_indices": [],
        "point_count": 0,
    }


def test_calculate_cell_distance_uses_cell_center():
    cell = make_cell(
        x_min=0.0,
        x_max=2.0,
        y_min=0.0,
        y_max=2.0,
    )

    assert calculate_cell_distance(cell) == pytest.approx(
        2**0.5
    )


def test_cell_distance_uses_horizontal_xy_only():
    cell = make_cell(
        x_min=3.0,
        x_max=4.0,
        y_min=4.0,
        y_max=5.0,
    )

    expected = ((3.5**2) + (4.5**2)) ** 0.5

    assert calculate_cell_distance(cell) == pytest.approx(
        expected
    )


def test_near_cell_gets_fine_resolution():
    cell = make_cell(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    assert calculate_cell_base_resolution(cell) == 0.05


def test_medium_distance_cell_gets_correct_resolution():
    cell = make_cell(
        x_min=18.0,
        x_max=19.0,
        y_min=0.0,
        y_max=1.0,
    )

    assert calculate_cell_base_resolution(cell) == 0.10


def test_far_cell_gets_coarse_resolution():
    cell = make_cell(
        x_min=60.0,
        x_max=61.0,
        y_min=0.0,
        y_max=1.0,
    )

    assert calculate_cell_base_resolution(cell) == 0.50


def test_cell_without_bbox_is_rejected():
    with pytest.raises(ValueError):
        calculate_cell_distance({})


def test_invalid_bbox_is_rejected():
    cell = make_cell(
        x_min=2.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    with pytest.raises(ValueError):
        calculate_cell_distance(cell)


def test_build_base_resolution_by_cell():
    cells = {
        (0, 0): make_cell(
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        ),
        (20, 0): make_cell(
            x_min=20.0,
            x_max=21.0,
            y_min=0.0,
            y_max=1.0,
        ),
        (60, 0): make_cell(
            x_min=60.0,
            x_max=61.0,
            y_min=0.0,
            y_max=1.0,
        ),
    }

    result = build_base_resolution_by_cell(cells)

    assert result[(0, 0)] == 0.05
    assert result[(20, 0)] == 0.10
    assert result[(60, 0)] == 0.50


def test_build_base_resolution_by_cell_empty():
    assert build_base_resolution_by_cell({}) == {}


def test_build_base_resolution_by_cell_rejects_invalid_input():
    with pytest.raises(ValueError):
        build_base_resolution_by_cell([])