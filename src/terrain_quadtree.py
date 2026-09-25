from __future__ import annotations

import numpy as np

from src.quadtree import (
    build_quadtree,
    collect_leaf_point_ids,
    get_leaf_nodes,
    validate_point_conservation,
)
from src.terrain_error_bound import (
    DEFAULT_MAX_ELEVATION_VARIATION,
    DEFAULT_MAX_PLANE_RESIDUAL,
    decide_terrain_error_bound,
)


DEFAULT_MIN_TERRAIN_RESOLUTION = 0.05


def _validate_data(data: np.ndarray) -> None:
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            "data must have shape (N, 3) or (N, 4)."
        )

    if len(data) == 0:
        raise ValueError(
            "data must not be empty."
        )


def _validate_point_indices(
    data: np.ndarray,
    point_indices: np.ndarray,
) -> np.ndarray:
    point_indices = np.asarray(
        point_indices,
        dtype=np.int64,
    )

    if point_indices.ndim != 1:
        raise ValueError(
            "point_indices must have shape (N,)."
        )

    if np.any(point_indices < 0) or np.any(
        point_indices >= len(data)
    ):
        raise ValueError(
            "point_indices contain invalid data indices."
        )

    return point_indices


def _validate_resolution(
    resolution: float,
    name: str,
) -> float:
    resolution = float(resolution)

    if (
        not np.isfinite(resolution)
        or resolution <= 0
    ):
        raise ValueError(
            f"{name} must be finite and greater than zero."
        )

    return resolution


def _get_node_points(
    points: np.ndarray,
    point_ids: np.ndarray,
    node_point_ids: list[int],
) -> np.ndarray:
    """
    Convert original point IDs stored by a node into
    the corresponding rows of the local points array.
    """

    point_ids = np.asarray(
        point_ids,
        dtype=np.int64,
    )

    node_point_ids = np.asarray(
        node_point_ids,
        dtype=np.int64,
    )

    id_to_local_index = {
        int(point_id): index
        for index, point_id in enumerate(point_ids)
    }

    try:
        local_indices = np.asarray(
            [
                id_to_local_index[int(point_id)]
                for point_id in node_point_ids
            ],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"Node contains unknown point ID: {exc.args[0]}"
        ) from exc

    return np.asarray(
        points[local_indices, :3],
        dtype=float,
    )


def _refine_node_to_requested_resolution(
    node,
    points: np.ndarray,
    point_ids: np.ndarray,
    target_resolution: float,
) -> None:
    """
    Refine a node until its cell size is less than or equal
    to the requested terrain resolution.

    The requested resolution is treated as an upper bound
    on occupied leaf cell size.
    """

    if node.point_count == 0:
        return

    if node.cell_size <= target_resolution:
        return

    before_count = node.point_count

    node.subdivide(
        points,
        point_ids=point_ids,
    )

    after_count = sum(
        child.point_count
        for child in node.children
    )

    if after_count != before_count:
        raise RuntimeError(
            "Requested-resolution refinement changed "
            "the point count."
        )

    for child in node.children:
        _refine_node_to_requested_resolution(
            node=child,
            points=points,
            point_ids=point_ids,
            target_resolution=target_resolution,
        )


def _refine_node_with_error_bound(
    node,
    points: np.ndarray,
    point_ids: np.ndarray,
    min_resolution: float,
    max_plane_residual: float,
    max_elevation_variation: float,
) -> None:
    """
    Apply terrain error-bound refinement to one quadtree node.

    A node is subdivided when its geometric error exceeds
    the configured thresholds.

    Splitting stops when another subdivision would make
    the child cell size smaller than min_resolution.
    """

    if node.point_count == 0:
        return

    next_cell_size = node.cell_size / 2.0

    if next_cell_size < min_resolution:
        return

    node_points = _get_node_points(
        points=points,
        point_ids=point_ids,
        node_point_ids=node.point_indices,
    )

    decision = decide_terrain_error_bound(
        points=node_points,
        max_plane_residual=max_plane_residual,
        max_elevation_variation=max_elevation_variation,
    )

    if not decision["split"]:
        return

    before_count = node.point_count

    node.subdivide(
        points,
        point_ids=point_ids,
    )

    after_count = sum(
        child.point_count
        for child in node.children
    )

    if after_count != before_count:
        raise RuntimeError(
            "Error-bound refinement changed "
            "the point count."
        )

    for child in node.children:
        _refine_node_with_error_bound(
            node=child,
            points=points,
            point_ids=point_ids,
            min_resolution=min_resolution,
            max_plane_residual=max_plane_residual,
            max_elevation_variation=max_elevation_variation,
        )


def build_terrain_quadtree_for_cell(
    data: np.ndarray,
    cell: dict,
    refinement_request: dict,
    max_plane_residual: float = (
        DEFAULT_MAX_PLANE_RESIDUAL
    ),
    max_elevation_variation: float = (
        DEFAULT_MAX_ELEVATION_VARIATION
    ),
    min_resolution: float = (
        DEFAULT_MIN_TERRAIN_RESOLUTION
    ),
) -> dict:
    """
    Build a quadtree for one terrain cell.

    Refinement happens in two stages:

    1. Reach the terrain-requested resolution.
    2. Apply the terrain geometric error bound for
       additional local refinement.
    """

    _validate_data(data)

    if not isinstance(cell, dict):
        raise ValueError(
            "cell must be a dictionary."
        )

    if "cell_id" not in cell:
        raise ValueError(
            "cell must contain cell_id."
        )

    if "bbox" not in cell:
        raise ValueError(
            "cell must contain bbox."
        )

    if "point_indices" not in cell:
        raise ValueError(
            "cell must contain point_indices."
        )

    if not isinstance(
        refinement_request,
        dict,
    ):
        raise ValueError(
            "refinement_request must be a dictionary."
        )

    if "requested_resolution" not in refinement_request:
        raise ValueError(
            "refinement_request must contain "
            "requested_resolution."
        )

    requested_resolution = _validate_resolution(
        refinement_request["requested_resolution"],
        "requested_resolution",
    )

    max_plane_residual = _validate_resolution(
        max_plane_residual,
        "max_plane_residual",
    )

    max_elevation_variation = _validate_resolution(
        max_elevation_variation,
        "max_elevation_variation",
    )

    min_resolution = _validate_resolution(
        min_resolution,
        "min_resolution",
    )

    point_indices = _validate_point_indices(
        data,
        cell["point_indices"],
    )

    if len(point_indices) == 0:
        raise ValueError(
            "Terrain cell must contain at least one point."
        )

    points = np.asarray(
        data[point_indices, :3],
        dtype=float,
    )

    root = build_quadtree(
        points=points,
        point_ids=point_indices,
        bbox=cell["bbox"],
    )

    # --------------------------------------------------
    # Stage 1:
    # Reach terrain-requested resolution.
    # --------------------------------------------------

    _refine_node_to_requested_resolution(
        node=root,
        points=points,
        point_ids=point_indices,
        target_resolution=requested_resolution,
    )

    # --------------------------------------------------
    # Stage 2:
    # Apply terrain error-bound refinement.
    # --------------------------------------------------

    leaves_before_error_refinement = get_leaf_nodes(
        root
    )

    for leaf in leaves_before_error_refinement:

        if leaf.point_count == 0:
            continue

        _refine_node_with_error_bound(
            node=leaf,
            points=points,
            point_ids=point_indices,
            min_resolution=min_resolution,
            max_plane_residual=max_plane_residual,
            max_elevation_variation=max_elevation_variation,
        )

    # --------------------------------------------------
    # Point conservation check.
    # --------------------------------------------------

    if not validate_point_conservation(
        root,
        point_indices,
    ):
        raise RuntimeError(
            "Terrain quadtree changed the point set."
        )

    leaves = get_leaf_nodes(root)

    occupied_leaves = [
        leaf
        for leaf in leaves
        if leaf.point_count > 0
    ]

    leaf_records = []

    for leaf in occupied_leaves:

        leaf_points = _get_node_points(
            points=points,
            point_ids=point_indices,
            node_point_ids=leaf.point_indices,
        )

        error_bound = decide_terrain_error_bound(
            points=leaf_points,
            max_plane_residual=max_plane_residual,
            max_elevation_variation=max_elevation_variation,
        )

        leaf_records.append(
            {
                "bbox": dict(leaf.bbox),
                "point_indices": list(
                    leaf.point_indices
                ),
                "point_count": leaf.point_count,
                "depth": leaf.depth,
                "cell_size": leaf.cell_size,
                "error_bound_split": error_bound[
                    "split"
                ],
                "error_bound_reason": error_bound[
                    "reason"
                ],
                "error_metrics": error_bound[
                    "metrics"
                ],
            }
        )

    return {
        "cell_id": cell["cell_id"],
        "bbox": dict(cell["bbox"]),
        "requested_resolution": requested_resolution,
        "priority": refinement_request.get(
            "priority"
        ),
        "reason": refinement_request.get(
            "reason"
        ),
        "root": root,
        "leaves": leaf_records,
        "point_count": len(point_indices),
        "leaf_point_count": sum(
            leaf["point_count"]
            for leaf in leaf_records
        ),
        "error_bound": {
            "max_plane_residual": max_plane_residual,
            "max_elevation_variation": (
                max_elevation_variation
            ),
            "min_resolution": min_resolution,
        },
    }


def build_terrain_quadtree_map(
    data: np.ndarray,
    cells: dict[tuple[int, int], dict],
    analyzed_cells: dict[tuple[int, int], dict],
    max_plane_residual: float = (
        DEFAULT_MAX_PLANE_RESIDUAL
    ),
    max_elevation_variation: float = (
        DEFAULT_MAX_ELEVATION_VARIATION
    ),
    min_resolution: float = (
        DEFAULT_MIN_TERRAIN_RESOLUTION
    ),
) -> dict[tuple[int, int], dict]:
    """
    Build terrain quadtrees for all analyzed terrain cells.
    """

    _validate_data(data)

    if not isinstance(cells, dict):
        raise ValueError(
            "cells must be a dictionary."
        )

    if not isinstance(
        analyzed_cells,
        dict,
    ):
        raise ValueError(
            "analyzed_cells must be a dictionary."
        )

    terrain_quadtree_map = {}

    for cell_id, cell in cells.items():

        if cell_id not in analyzed_cells:
            raise ValueError(
                f"No analyzed terrain cell found for {cell_id}."
            )

        analyzed_cell = analyzed_cells[
            cell_id
        ]

        refinement_request = analyzed_cell.get(
            "refinement_request"
        )

        if refinement_request is None:
            raise ValueError(
                f"Terrain cell {cell_id} has no refinement request."
            )

        terrain_quadtree_map[cell_id] = (
            build_terrain_quadtree_for_cell(
                data=data,
                cell=cell,
                refinement_request=refinement_request,
                max_plane_residual=max_plane_residual,
                max_elevation_variation=max_elevation_variation,
                min_resolution=min_resolution,
            )
        )

    return terrain_quadtree_map


def collect_terrain_leaf_point_ids(
    terrain_quadtree_map: dict[tuple[int, int], dict],
) -> np.ndarray:
    """
    Collect all original point IDs from all terrain
    quadtree leaves.
    """

    if not isinstance(
        terrain_quadtree_map,
        dict,
    ):
        raise ValueError(
            "terrain_quadtree_map must be a dictionary."
        )

    all_point_ids = []

    for result in terrain_quadtree_map.values():

        if "root" not in result:
            raise ValueError(
                "Each terrain quadtree result must contain root."
            )

        all_point_ids.extend(
            collect_leaf_point_ids(
                result["root"]
            ).tolist()
        )

    return np.asarray(
        all_point_ids,
        dtype=np.int64,
    )


def validate_terrain_quadtree_point_conservation(
    terrain_quadtree_map: dict[tuple[int, int], dict],
    expected_point_ids: np.ndarray,
) -> bool:
    """
    Verify that the complete terrain quadtree map contains
    every expected original point ID exactly once.
    """

    expected = np.asarray(
        expected_point_ids,
        dtype=np.int64,
    )

    if expected.ndim != 1:
        raise ValueError(
            "expected_point_ids must be one-dimensional."
        )

    if len(np.unique(expected)) != len(expected):
        return False

    actual = collect_terrain_leaf_point_ids(
        terrain_quadtree_map
    )

    if len(np.unique(actual)) != len(actual):
        return False

    if len(actual) != len(expected):
        return False

    return bool(
        np.array_equal(
            np.sort(actual),
            np.sort(expected),
        )
    )