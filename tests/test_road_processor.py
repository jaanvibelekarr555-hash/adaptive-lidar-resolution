import numpy as np

from src.road_processor import (
    process_road_frame,
    extract_ground_mask,
)


def test_process_road_frame_returns_cell_results(tmp_path):
    file_path = tmp_path / "sample.bin"

    # Synthetic ground-like LiDAR points.
    points = np.array(
        [
            [5.0, 0.0, 1.00, 0.5],
            [5.1, 0.0, 1.01, 0.5],
            [5.0, 0.1, 1.02, 0.5],
            [5.1, 0.1, 1.01, 0.5],
            [5.0, 0.2, 1.00, 0.5],
            [5.1, 0.2, 1.01, 0.5],
            [5.0, 0.3, 1.02, 0.5],
            [5.1, 0.3, 1.01, 0.5],
        ],
        dtype=np.float32,
    )

    points.tofile(file_path)

    results = process_road_frame(
        str(file_path),
        height_threshold=0.20,
    )

    assert isinstance(results, list)
    assert len(results) > 0

    for result in results:
        assert "cell_x" in result
        assert "cell_y" in result
        assert "point_count" in result
        assert "height_variation" in result
        assert "classification" in result

        assert result["classification"] in {
            "drivable",
            "non-drivable",
            "sparse",
        }


def test_process_road_frame_returns_valid_point_counts(tmp_path):
    file_path = tmp_path / "sample.bin"

    points = np.array(
        [
            [5.0, 0.0, 1.00, 0.5],
            [5.1, 0.0, 1.01, 0.5],
            [5.0, 0.1, 1.02, 0.5],
            [5.1, 0.1, 1.01, 0.5],
            [5.0, 0.2, 1.00, 0.5],
            [5.1, 0.2, 1.01, 0.5],
        ],
        dtype=np.float32,
    )

    points.tofile(file_path)

    results = process_road_frame(
        str(file_path),
        height_threshold=0.20,
    )

    for result in results:
        assert result["point_count"] >= 1


def test_extract_ground_mask_preserves_original_point_ids(tmp_path):
    file_path = tmp_path / "sample.bin"

    points = np.array(
        [
            [5.0, 0.0, 1.00, 0.5],
            [5.1, 0.0, 1.01, 0.5],
            [5.0, 0.1, 1.02, 0.5],
            [5.1, 0.1, 1.01, 0.5],
            [5.0, 0.2, 1.00, 0.5],
            [5.1, 0.2, 1.01, 0.5],
            [5.0, 0.3, 1.02, 0.5],
            [5.1, 0.3, 1.01, 0.5],
        ],
        dtype=np.float32,
    )

    points.tofile(file_path)

    data = np.fromfile(
        file_path,
        dtype=np.float32,
    ).reshape(-1, 4)

    ground_mask, point_id, plane_model = extract_ground_mask(
        data
    )

    # Ground mask must correspond to every original LiDAR point.
    assert ground_mask.shape == (len(data),)
    assert ground_mask.dtype == np.bool_

    # Point IDs must preserve the original point ordering.
    expected_ids = np.arange(
        len(data),
        dtype=np.int64,
    )

    assert np.array_equal(
        point_id,
        expected_ids,
    )

    # RANSAC should detect ground points.
    assert np.sum(ground_mask) > 0

    # Plane model must contain [a, b, c, d].
    assert plane_model.shape == (4,)