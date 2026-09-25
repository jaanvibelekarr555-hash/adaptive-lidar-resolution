import numpy as np
import pytest

from src.terrain_quadtree import (
    build_terrain_quadtree_for_cell,
    build_terrain_quadtree_map,
    collect_terrain_leaf_point_ids,
    validate_terrain_quadtree_point_conservation,
)


def make_cell(
    cell_id,
    point_indices,
    x_min,
    x_max,
    y_min,
    y_max,
):
    return {
        "cell_id": cell_id,
        "bbox": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "point_indices": point_indices,
        "point_count": len(point_indices),
    }


def make_request(
    resolution,
    priority=0.5,
    reason="test",
):
    return {
        "region": {
            "cell_id": (0, 0),
        },
        "requested_resolution": resolution,
        "priority": priority,
        "reason": reason,
    }


def test_build_terrain_quadtree_for_cell():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.0],
            [0.9, 0.1, 1.0, 0.0],
            [0.1, 0.9, 1.0, 0.0],
            [0.9, 0.9, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(0, 0),
        point_indices=[0, 1, 2, 3],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    request = make_request(
        resolution=0.25
    )

    result = build_terrain_quadtree_for_cell(
        data=data,
        cell=cell,
        refinement_request=request,
    )

    assert result["cell_id"] == (0, 0)
    assert result["requested_resolution"] == 0.25
    assert result["point_count"] == 4
    assert result["leaf_point_count"] == 4
    assert len(result["leaves"]) > 0


def test_requested_resolution_is_upper_bound():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.9, 0.1, 1.0],
            [0.1, 0.9, 1.0],
            [0.9, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(0, 0),
        point_indices=[0, 1, 2, 3],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    request = make_request(
        resolution=0.10
    )

    result = build_terrain_quadtree_for_cell(
        data=data,
        cell=cell,
        refinement_request=request,
    )

    for leaf in result["leaves"]:
        assert leaf["cell_size"] <= 0.10


def test_point_ids_are_preserved():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.9, 0.1, 1.0],
            [0.1, 0.9, 1.0],
            [0.9, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(5, 7),
        point_indices=[10, 20, 30, 40],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    # Expand data so these IDs are valid.
    expanded_data = np.zeros(
        (41, 3),
        dtype=np.float32,
    )

    expanded_data[10] = data[0]
    expanded_data[20] = data[1]
    expanded_data[30] = data[2]
    expanded_data[40] = data[3]

    result = build_terrain_quadtree_for_cell(
        data=expanded_data,
        cell=cell,
        refinement_request=make_request(0.25),
    )

    actual_ids = collect_terrain_leaf_point_ids(
        {(5, 7): result}
    )

    assert np.array_equal(
        np.sort(actual_ids),
        np.array(
            [10, 20, 30, 40],
            dtype=np.int64,
        ),
    )


def test_terrain_quadtree_point_conservation():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 1.0],
            [0.8, 0.2, 1.0],
            [0.9, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(0, 0),
        point_indices=[0, 1, 2, 3],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    result = build_terrain_quadtree_for_cell(
        data=data,
        cell=cell,
        refinement_request=make_request(0.125),
    )

    terrain_map = {
        (0, 0): result
    }

    expected = np.arange(
        4,
        dtype=np.int64,
    )

    assert validate_terrain_quadtree_point_conservation(
        terrain_map,
        expected,
    )


def test_build_multiple_terrain_quadtrees():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.9, 0.9, 1.0],
            [1.1, 0.1, 1.0],
            [1.9, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    cells = {
        (0, 0): make_cell(
            cell_id=(0, 0),
            point_indices=[0, 1],
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        ),
        (1, 0): make_cell(
            cell_id=(1, 0),
            point_indices=[2, 3],
            x_min=1.0,
            x_max=2.0,
            y_min=0.0,
            y_max=1.0,
        ),
    }

    analyzed_cells = {
        (0, 0): {
            "refinement_request": make_request(
                0.25
            )
        },
        (1, 0): {
            "refinement_request": make_request(
                0.10
            )
        },
    }

    terrain_map = build_terrain_quadtree_map(
        data=data,
        cells=cells,
        analyzed_cells=analyzed_cells,
    )

    assert set(
        terrain_map.keys()
    ) == {(0, 0), (1, 0)}

    expected = np.arange(
        4,
        dtype=np.int64,
    )

    assert validate_terrain_quadtree_point_conservation(
        terrain_map,
        expected,
    )


def test_missing_refinement_request_is_rejected():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(0, 0),
        point_indices=[0],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    analyzed_cells = {
        (0, 0): {
            "refinement_request": None
        }
    }

    with pytest.raises(ValueError):
        build_terrain_quadtree_map(
            data=data,
            cells={(0, 0): cell},
            analyzed_cells=analyzed_cells,
        )


def test_invalid_point_indices_are_rejected():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(0, 0),
        point_indices=[10],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    with pytest.raises(ValueError):
        build_terrain_quadtree_for_cell(
            data=data,
            cell=cell,
            refinement_request=make_request(0.25),
        )


def test_invalid_requested_resolution_is_rejected():
    data = np.array(
        [
            [0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(0, 0),
        point_indices=[0],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    with pytest.raises(ValueError):
        build_terrain_quadtree_for_cell(
            data=data,
            cell=cell,
            refinement_request=make_request(0.0),
        )
def test_error_bound_causes_additional_subdivision():
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(0, 0),
        point_indices=[0, 1, 2, 3],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    result = build_terrain_quadtree_for_cell(
        data=data,
        cell=cell,
        refinement_request=make_request(
            resolution=1.0
        ),
    )

    assert len(result["leaves"]) == 4

    assert all(
        leaf["cell_size"] == pytest.approx(0.5)
        for leaf in result["leaves"]
    )

    assert result["point_count"] == 4
    assert result["leaf_point_count"] == 4


def test_error_bound_does_not_split_flat_region():
    data = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    cell = make_cell(
        cell_id=(0, 0),
        point_indices=[0, 1, 2, 3],
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )

    result = build_terrain_quadtree_for_cell(
        data=data,
        cell=cell,
        refinement_request=make_request(
            resolution=1.0
        ),
    )

    assert len(result["leaves"]) == 1
    assert result["leaves"][0]["cell_size"] == pytest.approx(
        1.0
    )

    assert result["point_count"] == 4
    assert result["leaf_point_count"] == 4
