import numpy as np
import pytest

from src.quadtree import (
    QuadTreeNode,
    build_quadtree,
    collect_leaf_point_ids,
    get_leaf_nodes,
    refine_node_to_resolution,
    validate_point_conservation,
)


def test_build_quadtree_without_splitting():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )

    root = build_quadtree(
        points,
        max_points_per_leaf=10,
    )

    assert root.is_leaf
    assert root.point_count == 3
    assert root.depth == 0
    assert len(get_leaf_nodes(root)) == 1


def test_quadtree_splits_into_four_children():
    points = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9],
        ]
    )

    root = build_quadtree(
        points,
        max_points_per_leaf=1,
        max_depth=1,
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    assert not root.is_leaf
    assert len(root.children) == 4

    counts = sorted(
        child.point_count
        for child in root.children
    )

    assert counts == [1, 1, 1, 1]


def test_original_point_ids_are_preserved():
    points = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9],
        ]
    )

    point_ids = np.array(
        [10, 20, 30, 40],
        dtype=np.int64,
    )

    root = build_quadtree(
        points,
        point_ids=point_ids,
        max_points_per_leaf=1,
        max_depth=1,
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    actual = collect_leaf_point_ids(root)

    assert np.array_equal(
        np.sort(actual),
        np.sort(point_ids),
    )


def test_point_conservation_after_recursive_splitting():
    points = np.array(
        [
            [0.05, 0.05],
            [0.10, 0.10],
            [0.20, 0.20],
            [0.70, 0.10],
            [0.80, 0.20],
            [0.10, 0.80],
            [0.20, 0.90],
            [0.75, 0.75],
            [0.90, 0.90],
            [0.95, 0.95],
        ]
    )

    point_ids = np.arange(
        100,
        110,
        dtype=np.int64,
    )

    root = build_quadtree(
        points,
        point_ids=point_ids,
        max_points_per_leaf=2,
        max_depth=5,
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    assert validate_point_conservation(
        root,
        point_ids,
    )

    assert len(
        collect_leaf_point_ids(root)
    ) == len(point_ids)


def test_leaf_point_ids_are_unique():
    points = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9],
        ]
    )

    point_ids = np.array(
        [4, 8, 15, 16],
        dtype=np.int64,
    )

    root = build_quadtree(
        points,
        point_ids=point_ids,
        max_points_per_leaf=1,
        max_depth=1,
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    actual = collect_leaf_point_ids(root)

    assert len(
        np.unique(actual)
    ) == len(actual)


def test_invalid_points_shape_is_rejected():
    with pytest.raises(ValueError):
        build_quadtree(
            np.array([1.0, 2.0, 3.0])
        )


def test_invalid_point_id_shape_is_rejected():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(ValueError):
        build_quadtree(
            points,
            point_ids=np.array([[0, 1]]),
        )


def test_invalid_bbox_is_rejected():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(ValueError):
        build_quadtree(
            points,
            bbox={
                "x_min": 0.0,
                "x_max": 2.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
        )


def test_empty_points_are_rejected():
    with pytest.raises(ValueError):
        build_quadtree(
            np.empty((0, 2))
        )


def test_leaf_point_count_matches_ids():
    points = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9],
        ]
    )

    root = build_quadtree(
        points,
        max_points_per_leaf=1,
        max_depth=1,
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    for leaf in get_leaf_nodes(root):
        assert leaf.point_count == len(
            leaf.point_indices
        )


def test_subdivide_rejects_second_split():
    points = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.9],
        ]
    )

    root = QuadTreeNode(
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
        point_indices=[0, 1],
    )

    root.subdivide(points)

    with pytest.raises(ValueError):
        root.subdivide(points)


def test_refine_node_to_resolution():
    points = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9],
        ]
    )

    point_ids = np.arange(
        4,
        dtype=np.int64,
    )

    root = build_quadtree(
        points,
        point_ids=point_ids,
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    refine_node_to_resolution(
        node=root,
        points=points,
        point_ids=point_ids,
        target_resolution=0.25,
    )

    leaves = get_leaf_nodes(root)

    assert all(
        leaf.cell_size <= 0.25
        for leaf in leaves
    )

    assert validate_point_conservation(
        root,
        point_ids,
    )


def test_refine_node_to_resolution_does_not_split_when_already_fine():
    points = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.9],
        ]
    )

    root = build_quadtree(
        points,
        bbox={
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
    )

    refine_node_to_resolution(
        node=root,
        points=points,
        point_ids=np.array([0, 1]),
        target_resolution=1.0,
    )

    assert root.is_leaf