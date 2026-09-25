import numpy as np

from src.terrain_cells import build_terrain_cells


def test_build_terrain_cells_preserves_original_point_indices():
    data = np.array(
        [
            [0.10, 0.10, 1.0, 0.5],
            [0.20, 0.20, 1.1, 0.5],
            [0.90, 0.90, 1.2, 0.5],
            [1.10, 0.10, 1.3, 0.5],
            [1.20, 0.20, 1.4, 0.5],
        ],
        dtype=np.float32,
    )

    ground_mask = np.array(
        [True, True, False, True, True]
    )

    point_id = np.arange(
        len(data),
        dtype=np.int64,
    )

    cells = build_terrain_cells(
        data,
        ground_mask,
        point_id,
        cell_size=1.0,
    )

    assert len(cells) == 2

    assert (0, 0) in cells
    assert (1, 0) in cells

    assert cells[(0, 0)]["point_indices"] == [0, 1]

    assert cells[(1, 0)]["point_indices"] == [3, 4]


def test_build_terrain_cells_creates_correct_bounding_boxes():
    data = np.array(
        [
            [0.10, 0.10, 1.0, 0.5],
            [1.10, 0.10, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    ground_mask = np.array(
        [True, True]
    )

    point_id = np.arange(
        len(data),
        dtype=np.int64,
    )

    cells = build_terrain_cells(
        data,
        ground_mask,
        point_id,
        cell_size=1.0,
    )

    assert cells[(0, 0)]["bbox"] == {
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
    }

    assert cells[(1, 0)]["bbox"] == {
        "x_min": 1.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 1.0,
    }


def test_build_terrain_cells_returns_empty_for_no_ground():
    data = np.array(
        [
            [1.0, 1.0, 1.0, 0.5],
            [2.0, 2.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    ground_mask = np.array(
        [False, False]
    )

    point_id = np.arange(
        len(data),
        dtype=np.int64,
    )

    cells = build_terrain_cells(
        data,
        ground_mask,
        point_id,
    )

    assert cells == {}