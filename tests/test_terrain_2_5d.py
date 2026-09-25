import numpy as np
import pytest

from src.terrain_2_5d import (
    build_terrain_2_5d_cell,
    build_terrain_2_5d_map,
    validate_terrain_2_5d_point_conservation,
)


def make_result(
    data_indices,
    bbox=None,
    depth=2,
    cell_size=0.25,
):
    if bbox is None:
        bbox = {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        }

    return {
        "cell_id": (0, 0),
        "bbox": bbox,
        "requested_resolution": cell_size,
        "leaves": [
            {
                "bbox": bbox,
                "point_indices": data_indices,
                "point_count": len(data_indices),
                "depth": depth,
                "cell_size": cell_size,
                "error_bound_split": False,
                "error_bound_reason": (
                    "terrain error within bounds"
                ),
                "error_metrics": {},
            }
        ],
    }


def test_build_terrain_2_5d_cell():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.9, 0.1, 1.1],
            [0.1, 0.9, 1.2],
            [0.9, 0.9, 1.3],
        ],
        dtype=np.float32,
    )

    result = make_result(
        [0, 1, 2, 3]
    )

    cells = build_terrain_2_5d_cell(
        data=data,
        terrain_quadtree_result=result,
    )

    assert len(cells) == 1

    cell = cells[0]

    assert cell["cell_id"] == (0, 0)
    assert "leaf_id" in cell
    assert "bbox" in cell
    assert "centroid" in cell

    assert cell["z_mean"] == pytest.approx(
        1.15
    )

    assert cell["z_min"] == pytest.approx(
        1.0
    )

    assert cell["z_max"] == pytest.approx(
        1.3
    )

    assert cell["elevation"] == pytest.approx(
        1.15
    )

    assert cell["point_count"] == 4
    assert cell["resolution"] == pytest.approx(
        0.25
    )
    assert cell["occupied"] is True


def test_2_5d_cell_centroid():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.9, 0.9, 1.0],
            [0.2, 0.8, 1.0],
        ],
        dtype=np.float32,
    )

    result = make_result(
        [0, 1, 2]
    )

    cell = build_terrain_2_5d_cell(
        data=data,
        terrain_quadtree_result=result,
    )[0]

    assert cell["centroid"]["x"] == pytest.approx(
        0.5
    )

    assert cell["centroid"]["y"] == pytest.approx(
        0.5
    )


def test_2_5d_cell_contains_terrain_features():
    data = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = make_result(
        [0, 1, 2, 3]
    )

    cell = build_terrain_2_5d_cell(
        data=data,
        terrain_quadtree_result=result,
    )[0]

    assert cell["slope"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert cell["roughness"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert cell["elevation_variation"] == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert cell["traversability"] == (
        "DRIVABLE"
    )


def test_sparse_leaf_is_handled():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )

    result = make_result(
        [0]
    )

    cell = build_terrain_2_5d_cell(
        data=data,
        terrain_quadtree_result=result,
    )[0]

    assert cell["geometry_status"] == (
        "sparse"
    )

    assert cell["terrain_complexity"] is None
    assert cell["traversability"] == (
        "SPARSE / UNKNOWN"
    )
    assert cell["terrain_confidence"] == 0.0
    assert cell["terrain_priority"] == 1.0


def test_build_terrain_2_5d_map():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.9, 0.9, 1.0],
            [1.1, 0.1, 2.0],
            [1.9, 0.9, 2.0],
        ],
        dtype=np.float32,
    )

    result_a = make_result(
        [0, 1],
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    result_b = make_result(
        [2, 3],
        bbox={
            "x_min": 1.0,
            "x_max": 2.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    result_b["cell_id"] = (1, 0)

    terrain_map = build_terrain_2_5d_map(
        data=data,
        terrain_quadtree_map={
            (0, 0): result_a,
            (1, 0): result_b,
        },
    )

    assert len(terrain_map) == 2


def test_2_5d_point_conservation():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.9, 0.1, 1.0],
            [0.1, 0.9, 1.0],
            [0.9, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    result = make_result(
        [0, 1, 2, 3]
    )

    terrain_map = build_terrain_2_5d_map(
        data=data,
        terrain_quadtree_map={
            (0, 0): result
        },
    )

    expected = np.arange(
        4,
        dtype=np.int64,
    )

    assert validate_terrain_2_5d_point_conservation(
        terrain_map,
        expected,
    )


def test_2_5d_point_conservation_rejects_duplicates():
    terrain_map = [
        {
            "point_indices": [0, 1],
        },
        {
            "point_indices": [1, 2],
        },
    ]

    expected = np.array(
        [0, 1, 2],
        dtype=np.int64,
    )

    assert (
        validate_terrain_2_5d_point_conservation(
            terrain_map,
            expected,
        )
        is False
    )


def test_invalid_data_is_rejected():
    with pytest.raises(ValueError):
        build_terrain_2_5d_map(
            data=np.array([]),
            terrain_quadtree_map={},
        )


def test_empty_terrain_quadtree_map_is_supported():
    data = np.array(
        [
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = build_terrain_2_5d_map(
        data=data,
        terrain_quadtree_map={},
    )

    assert result == []


def test_empty_leaf_is_not_added():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )

    result = {
        "cell_id": (0, 0),
        "bbox": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
        "leaves": [
            {
                "bbox": {
                    "x_min": 0.0,
                    "x_max": 1.0,
                    "y_min": 0.0,
                    "y_max": 1.0,
                },
                "point_indices": [],
                "point_count": 0,
                "depth": 0,
                "cell_size": 1.0,
            }
        ],
    }

    cells = build_terrain_2_5d_cell(
        data=data,
        terrain_quadtree_result=result,
    )

    assert cells == []