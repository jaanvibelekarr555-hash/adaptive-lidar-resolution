import numpy as np
import pytest

from src.terrain_refinement import (
    calculate_requested_resolution,
    build_terrain_refinement_request,
)


def test_low_priority_keeps_base_resolution():
    resolution = calculate_requested_resolution(
        base_resolution=0.10,
        priority=0.20,
    )

    assert np.isclose(
        resolution,
        0.10,
    )


def test_medium_priority_requests_finer_resolution():
    resolution = calculate_requested_resolution(
        base_resolution=0.20,
        priority=0.50,
    )

    assert np.isclose(
        resolution,
        0.15,
    )


def test_high_priority_requests_finer_resolution():
    resolution = calculate_requested_resolution(
        base_resolution=0.20,
        priority=0.80,
    )

    assert np.isclose(
        resolution,
        0.10,
    )


def test_resolution_respects_minimum_resolution():
    resolution = calculate_requested_resolution(
        base_resolution=0.05,
        priority=1.0,
    )

    assert np.isclose(
        resolution,
        0.05,
    )


def test_custom_refinement_factors():
    resolution = calculate_requested_resolution(
        base_resolution=0.20,
        priority=0.80,
        high_refinement_factor=0.25,
        min_resolution=0.01,
    )

    assert np.isclose(
        resolution,
        0.05,
    )


def test_invalid_base_resolution_is_rejected():
    with pytest.raises(ValueError):
        calculate_requested_resolution(
            base_resolution=0.0,
            priority=0.5,
        )


def test_invalid_priority_is_rejected():
    with pytest.raises(ValueError):
        calculate_requested_resolution(
            base_resolution=0.10,
            priority=1.5,
        )


def test_build_refinement_request_for_complex_terrain():
    region = {
        "x_min": 10.0,
        "x_max": 11.0,
        "y_min": 2.0,
        "y_max": 3.0,
    }

    request = build_terrain_refinement_request(
        region=region,
        base_resolution=0.20,
        priority=0.80,
        traversability="NON-DRIVABLE",
        terrain_complexity=0.85,
        terrain_confidence=0.90,
    )

    assert request["region"] == region

    assert np.isclose(
        request["requested_resolution"],
        0.10,
    )

    assert np.isclose(
        request["priority"],
        0.80,
    )

    assert "non-drivable terrain" in request["reason"]

    assert "high terrain complexity" in request["reason"]

    assert "high terrain refinement priority" in request["reason"]


def test_build_refinement_request_for_sparse_terrain():
    region = {
        "x_min": 20.0,
        "x_max": 21.0,
        "y_min": 4.0,
        "y_max": 5.0,
    }

    request = build_terrain_refinement_request(
        region=region,
        base_resolution=0.10,
        priority=1.0,
        traversability="SPARSE / UNKNOWN",
        terrain_complexity=None,
        terrain_confidence=0.0,
    )

    assert request["region"] == region

    assert np.isclose(
        request["requested_resolution"],
        0.05,
    )

    assert np.isclose(
        request["priority"],
        1.0,
    )

    assert "sparse or unknown terrain" in request["reason"]

    assert "low terrain confidence" in request["reason"]

    assert "high terrain refinement priority" in request["reason"]


def test_build_refinement_request_for_normal_terrain():
    region = {
        "x_min": 5.0,
        "x_max": 6.0,
        "y_min": 0.0,
        "y_max": 1.0,
    }

    request = build_terrain_refinement_request(
        region=region,
        base_resolution=0.10,
        priority=0.10,
        traversability="DRIVABLE",
        terrain_complexity=0.10,
        terrain_confidence=1.0,
    )

    assert request["region"] == region

    assert np.isclose(
        request["requested_resolution"],
        0.10,
    )

    assert np.isclose(
        request["priority"],
        0.10,
    )

    assert (
        request["reason"]
        == "low-priority terrain; retain base resolution"
    )


def test_invalid_traversability_is_rejected():
    region = {
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
    }

    with pytest.raises(ValueError):
        build_terrain_refinement_request(
            region=region,
            base_resolution=0.10,
            priority=0.5,
            traversability="UNKNOWN",
            terrain_complexity=0.5,
            terrain_confidence=0.8,
        )


def test_invalid_confidence_is_rejected():
    region = {
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
    }

    with pytest.raises(ValueError):
        build_terrain_refinement_request(
            region=region,
            base_resolution=0.10,
            priority=0.5,
            traversability="DRIVABLE",
            terrain_complexity=0.5,
            terrain_confidence=1.2,
        )