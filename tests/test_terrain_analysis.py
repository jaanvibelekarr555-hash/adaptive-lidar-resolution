import numpy as np

from src.terrain_analysis import analyze_terrain_cells


def test_analyze_terrain_cells_adds_geometry_features():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.5],
            [0.9, 0.1, 1.0, 0.5],
            [0.1, 0.9, 1.0, 0.5],
            [0.9, 0.9, 1.0, 0.5],
        ],
        dtype=np.float64,
    )

    cells = {
        (0, 0): {
            "cell_id": (0, 0),
            "bbox": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "point_indices": [0, 1, 2, 3],
            "point_count": 4,
        }
    }

    results = analyze_terrain_cells(
        data,
        cells,
    )

    result = results[(0, 0)]

    assert "cell_id" in result
    assert "bbox" in result
    assert "point_indices" in result
    assert "point_count" in result

    assert "slope" in result
    assert "roughness" in result
    assert "elevation_variation" in result
    assert "geometry_status" in result
    assert "terrain_complexity" in result
    assert "traversability" in result
    assert "terrain_confidence" in result
    assert "terrain_priority" in result
    assert "refinement_request" in result

    assert result["geometry_status"] == "valid"

    assert result["slope"] < 1e-6

    assert result["roughness"] < 1e-6

    assert np.isclose(
        result["elevation_variation"],
        0.0,
    )

    assert np.isclose(
        result["terrain_complexity"],
        0.0,
    )

    assert result["traversability"] == "DRIVABLE"

    assert np.isclose(
        result["terrain_confidence"],
        0.2,
    )

    assert np.isclose(
        result["terrain_priority"],
        0.16,
    )

    assert result["refinement_request"] is None


def test_analyze_terrain_cells_uses_original_point_indices():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.5],
            [0.9, 0.1, 1.0, 0.5],
            [5.0, 5.0, 10.0, 0.5],
            [0.1, 0.9, 1.0, 0.5],
            [0.9, 0.9, 1.0, 0.5],
        ],
        dtype=np.float64,
    )

    cells = {
        (0, 0): {
            "cell_id": (0, 0),
            "bbox": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "point_indices": [0, 1, 3, 4],
            "point_count": 4,
        }
    }

    results = analyze_terrain_cells(
        data,
        cells,
    )

    result = results[(0, 0)]

    assert np.isclose(
        result["elevation_variation"],
        0.0,
    )

    assert np.isclose(
        result["terrain_complexity"],
        0.0,
    )

    assert result["traversability"] == "DRIVABLE"

    assert np.isclose(
        result["terrain_confidence"],
        0.2,
    )

    assert np.isclose(
        result["terrain_priority"],
        0.16,
    )

    assert result["refinement_request"] is None


def test_analyze_terrain_cells_handles_sparse_cell():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.5],
            [0.9, 0.1, 1.2, 0.5],
        ],
        dtype=np.float64,
    )

    cells = {
        (0, 0): {
            "cell_id": (0, 0),
            "bbox": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "point_indices": [0, 1],
            "point_count": 2,
        }
    }

    results = analyze_terrain_cells(
        data,
        cells,
    )

    result = results[(0, 0)]

    assert result["geometry_status"] == "sparse"

    assert result["slope"] is None

    assert result["roughness"] is None

    assert result["elevation_variation"] is None

    assert result["terrain_complexity"] is None

    assert result["traversability"] == "SPARSE / UNKNOWN"

    assert np.isclose(
        result["terrain_confidence"],
        0.0,
    )

    assert np.isclose(
        result["terrain_priority"],
        1.0,
    )

    assert result["refinement_request"] is None


def test_analyze_terrain_cells_rejects_invalid_point_indices():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.5],
            [0.9, 0.1, 1.0, 0.5],
            [0.1, 0.9, 1.0, 0.5],
        ],
        dtype=np.float64,
    )

    cells = {
        (0, 0): {
            "cell_id": (0, 0),
            "bbox": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "point_indices": [0, 1, 10],
            "point_count": 3,
        }
    }

    try:
        analyze_terrain_cells(
            data,
            cells,
        )
        assert False
    except ValueError:
        pass


def test_analyze_terrain_cells_builds_refinement_request():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.5],
            [0.9, 0.1, 1.0, 0.5],
            [0.1, 0.9, 1.0, 0.5],
            [0.9, 0.9, 1.0, 0.5],
        ],
        dtype=np.float64,
    )

    cells = {
        (0, 0): {
            "cell_id": (0, 0),
            "bbox": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "point_indices": [0, 1, 2, 3],
            "point_count": 4,
        }
    }

    base_resolution_by_cell = {
        (0, 0): 0.20,
    }

    results = analyze_terrain_cells(
        data,
        cells,
        base_resolution_by_cell=base_resolution_by_cell,
    )

    result = results[(0, 0)]

    assert result["refinement_request"] is not None

    request = result["refinement_request"]

    assert "region" in request
    assert "requested_resolution" in request
    assert "priority" in request
    assert "reason" in request

    assert request["region"]["cell_id"] == (0, 0)

    assert request["region"]["bbox"] == {
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
    }

    assert np.isclose(
        request["requested_resolution"],
        0.20,
    )

    assert np.isclose(
        request["priority"],
        0.16,
    )

    assert "low terrain confidence" in request["reason"]


def test_analyze_terrain_cells_builds_sparse_refinement_request():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.5],
            [0.9, 0.1, 1.2, 0.5],
        ],
        dtype=np.float64,
    )

    cells = {
        (0, 0): {
            "cell_id": (0, 0),
            "bbox": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "point_indices": [0, 1],
            "point_count": 2,
        }
    }

    base_resolution_by_cell = {
        (0, 0): 0.10,
    }

    results = analyze_terrain_cells(
        data,
        cells,
        base_resolution_by_cell=base_resolution_by_cell,
    )

    result = results[(0, 0)]

    request = result["refinement_request"]

    assert request is not None

    assert np.isclose(
        request["priority"],
        1.0,
    )

    assert np.isclose(
        request["requested_resolution"],
        0.05,
    )

    assert "sparse or unknown terrain" in request["reason"]


def test_analyze_terrain_cells_rejects_missing_base_resolution():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.5],
            [0.9, 0.1, 1.0, 0.5],
            [0.1, 0.9, 1.0, 0.5],
        ],
        dtype=np.float64,
    )

    cells = {
        (0, 0): {
            "cell_id": (0, 0),
            "bbox": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "point_indices": [0, 1, 2],
            "point_count": 3,
        }
    }

    base_resolution_by_cell = {}

    try:
        analyze_terrain_cells(
            data,
            cells,
            base_resolution_by_cell=base_resolution_by_cell,
        )
        assert False
    except ValueError:
        pass


def test_analyze_terrain_cells_rejects_invalid_base_resolution():
    data = np.array(
        [
            [0.1, 0.1, 1.0, 0.5],
            [0.9, 0.1, 1.0, 0.5],
            [0.1, 0.9, 1.0, 0.5],
        ],
        dtype=np.float64,
    )

    cells = {
        (0, 0): {
            "cell_id": (0, 0),
            "bbox": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "point_indices": [0, 1, 2],
            "point_count": 3,
        }
    }

    base_resolution_by_cell = {
        (0, 0): 0.0,
    }

    try:
        analyze_terrain_cells(
            data,
            cells,
            base_resolution_by_cell=base_resolution_by_cell,
        )
        assert False
    except ValueError:
        pass