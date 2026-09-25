import numpy as np

from src.terrain_cells import build_terrain_cells
from src.terrain_analysis import analyze_terrain_cells
from src.terrain_resolution import build_base_resolution_by_cell


def test_terrain_cells_to_refinement_request():
    data = np.array(
        [
            [0.10, 0.10, 1.00, 0.5],
            [0.20, 0.20, 1.01, 0.5],
            [0.30, 0.30, 1.02, 0.5],
            [18.10, 0.10, 2.00, 0.5],
            [18.20, 0.20, 2.01, 0.5],
            [18.30, 0.30, 2.02, 0.5],
        ],
        dtype=np.float32,
    )

    ground_mask = np.array(
        [True, True, True, True, True, True]
    )

    point_id = np.arange(
        len(data),
        dtype=np.int64,
    )

    cells = build_terrain_cells(
        data=data,
        ground_mask=ground_mask,
        point_id=point_id,
        cell_size=1.0,
    )

    base_resolution_by_cell = (
        build_base_resolution_by_cell(cells)
    )

    analyzed_cells = analyze_terrain_cells(
        data=data,
        cells=cells,
        base_resolution_by_cell=base_resolution_by_cell,
    )

    assert len(analyzed_cells) == 2

    for cell in analyzed_cells.values():
        assert cell["terrain_priority"] is not None
        assert cell["refinement_request"] is not None

        request = cell["refinement_request"]

        assert "region" in request
        assert "requested_resolution" in request
        assert "priority" in request
        assert "reason" in request

        assert request["requested_resolution"] > 0
        assert 0.0 <= request["priority"] <= 1.0


def test_terrain_analysis_without_resolution_still_works():
    data = np.array(
        [
            [0.10, 0.10, 1.00, 0.5],
            [0.20, 0.20, 1.01, 0.5],
            [0.30, 0.30, 1.02, 0.5],
        ],
        dtype=np.float32,
    )

    ground_mask = np.array(
        [True, True, True]
    )

    point_id = np.arange(
        len(data),
        dtype=np.int64,
    )

    cells = build_terrain_cells(
        data=data,
        ground_mask=ground_mask,
        point_id=point_id,
        cell_size=1.0,
    )

    analyzed_cells = analyze_terrain_cells(
        data=data,
        cells=cells,
    )

    for cell in analyzed_cells.values():
        assert cell["terrain_priority"] is not None
        assert cell["refinement_request"] is None