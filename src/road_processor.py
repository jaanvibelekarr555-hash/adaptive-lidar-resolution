import numpy as np
import open3d as o3d

from src.config import ROAD_CELL_SIZE
from src.road_geometry import analyze_road_cells


def process_road_frame(
    file_path: str,
    height_threshold: float = 0.20,
) -> list[dict]:
    """
    Process one LiDAR frame and classify local ground/road cells.

    Returns:
        List of dictionaries containing:
        - cell_x
        - cell_y
        - point_count
        - height_variation
        - classification
    """

    # --------------------------------------------------
    # 1. Load LiDAR frame
    # --------------------------------------------------

    data = np.fromfile(
        file_path,
        dtype=np.float32,
    ).reshape(-1, 4)

    if len(data) == 0:
        raise ValueError("LiDAR file is empty.")

    # --------------------------------------------------
    # 2. Select region in front of the vehicle
    # --------------------------------------------------

    points = data[
        (data[:, 0] > 0)
        & (data[:, 0] < 30)
        & (np.abs(data[:, 1]) < 10)
    ]

    if len(points) == 0:
        raise ValueError(
            "No LiDAR points found in the selected region."
        )

    xyz = points[:, :3]

    # --------------------------------------------------
    # 3. Estimate dominant ground plane
    # --------------------------------------------------

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.15,
        ransac_n=3,
        num_iterations=1000,
    )

    ground_points = xyz[inliers]

    if len(ground_points) == 0:
        raise ValueError("No ground points detected.")

    # --------------------------------------------------
    # 4. Divide ground into local cells
    # --------------------------------------------------

    cell_size = ROAD_CELL_SIZE

    x_min = ground_points[:, 0].min()
    y_min = ground_points[:, 1].min()

    cell_x = np.floor(
        (ground_points[:, 0] - x_min) / cell_size
    ).astype(int)

    cell_y = np.floor(
        (ground_points[:, 1] - y_min) / cell_size
    ).astype(int)

    cells = {}

    for i in range(len(ground_points)):

        key = (
            int(cell_x[i]),
            int(cell_y[i]),
        )

        if key not in cells:
            cells[key] = []

        cells[key].append(ground_points[i])

    # --------------------------------------------------
    # 5. Analyze all road cells
    # --------------------------------------------------

    results = analyze_road_cells(
        cells,
        height_threshold=height_threshold,
    )

    return results