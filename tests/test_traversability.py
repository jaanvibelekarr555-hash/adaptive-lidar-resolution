import numpy as np
import pytest

from src.traversability import (
    classify_traversability,
    calculate_traversability_confidence,
)


def test_flat_terrain_is_drivable():
    result = classify_traversability(
        slope=2.0,
        roughness=0.01,
        elevation_variation=0.03,
        geometry_status="valid",
    )

    assert result == "DRIVABLE"


def test_steep_terrain_is_non_drivable():
    result = classify_traversability(
        slope=35.0,
        roughness=0.05,
        elevation_variation=0.20,
        geometry_status="valid",
    )

    assert result == "NON-DRIVABLE"


def test_rough_terrain_is_non_drivable():
    result = classify_traversability(
        slope=5.0,
        roughness=0.25,
        elevation_variation=0.20,
        geometry_status="valid",
    )

    assert result == "NON-DRIVABLE"


def test_high_elevation_variation_is_non_drivable():
    result = classify_traversability(
        slope=5.0,
        roughness=0.05,
        elevation_variation=0.60,
        geometry_status="valid",
    )

    assert result == "NON-DRIVABLE"


def test_sparse_terrain_is_unknown():
    result = classify_traversability(
        slope=None,
        roughness=None,
        elevation_variation=None,
        geometry_status="sparse",
    )

    assert result == "SPARSE / UNKNOWN"


def test_custom_thresholds():
    result = classify_traversability(
        slope=15.0,
        roughness=0.10,
        elevation_variation=0.20,
        geometry_status="valid",
        max_slope=10.0,
    )

    assert result == "NON-DRIVABLE"


def test_valid_terrain_confidence():
    confidence = calculate_traversability_confidence(
        point_count=20,
        complexity=0.3,
        geometry_status="valid",
    )

    assert np.isclose(
        confidence,
        1.0,
    )


def test_low_point_count_has_lower_confidence():
    confidence = calculate_traversability_confidence(
        point_count=5,
        complexity=0.3,
        geometry_status="valid",
    )

    assert np.isclose(
        confidence,
        0.25,
    )


def test_sparse_terrain_has_zero_confidence():
    confidence = calculate_traversability_confidence(
        point_count=2,
        complexity=None,
        geometry_status="sparse",
    )

    assert np.isclose(
        confidence,
        0.0,
    )


def test_invalid_complexity_is_rejected():
    with pytest.raises(ValueError):
        calculate_traversability_confidence(
            point_count=10,
            complexity=1.5,
            geometry_status="valid",
        )


def test_invalid_geometry_status_is_rejected():
    with pytest.raises(ValueError):
        classify_traversability(
            slope=5.0,
            roughness=0.05,
            elevation_variation=0.10,
            geometry_status="unknown",
        )