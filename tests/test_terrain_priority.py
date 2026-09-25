import numpy as np
import pytest

from src.terrain_priority import (
    calculate_terrain_priority,
)


def test_simple_drivable_terrain_has_low_priority():
    priority = calculate_terrain_priority(
        terrain_complexity=0.0,
        traversability="DRIVABLE",
        terrain_confidence=1.0,
    )

    assert np.isclose(
        priority,
        0.0,
    )


def test_complex_terrain_has_higher_priority():
    priority = calculate_terrain_priority(
        terrain_complexity=0.8,
        traversability="DRIVABLE",
        terrain_confidence=1.0,
    )

    assert priority > 0.0


def test_non_drivable_terrain_has_higher_priority():
    priority = calculate_terrain_priority(
        terrain_complexity=0.8,
        traversability="NON-DRIVABLE",
        terrain_confidence=1.0,
    )

    assert priority > 0.0


def test_sparse_unknown_terrain_requests_information():
    priority = calculate_terrain_priority(
        terrain_complexity=None,
        traversability="SPARSE / UNKNOWN",
        terrain_confidence=0.0,
    )

    assert np.isclose(
        priority,
        1.0,
    )


def test_low_confidence_increases_priority():
    high_confidence = calculate_terrain_priority(
        terrain_complexity=0.4,
        traversability="DRIVABLE",
        terrain_confidence=1.0,
    )

    low_confidence = calculate_terrain_priority(
        terrain_complexity=0.4,
        traversability="DRIVABLE",
        terrain_confidence=0.2,
    )

    assert low_confidence > high_confidence


def test_higher_complexity_increases_priority():
    simple = calculate_terrain_priority(
        terrain_complexity=0.2,
        traversability="DRIVABLE",
        terrain_confidence=1.0,
    )

    complex_terrain = calculate_terrain_priority(
        terrain_complexity=0.8,
        traversability="DRIVABLE",
        terrain_confidence=1.0,
    )

    assert complex_terrain > simple


def test_custom_weights():
    priority = calculate_terrain_priority(
        terrain_complexity=0.0,
        traversability="NON-DRIVABLE",
        terrain_confidence=1.0,
        complexity_weight=0.0,
        risk_weight=1.0,
        uncertainty_weight=0.0,
    )

    assert np.isclose(
        priority,
        1.0,
    )


def test_priority_is_between_zero_and_one():
    priority = calculate_terrain_priority(
        terrain_complexity=0.7,
        traversability="NON-DRIVABLE",
        terrain_confidence=0.3,
    )

    assert 0.0 <= priority <= 1.0


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError):
        calculate_terrain_priority(
            terrain_complexity=0.5,
            traversability="DRIVABLE",
            terrain_confidence=1.5,
        )


def test_invalid_complexity_is_rejected():
    with pytest.raises(ValueError):
        calculate_terrain_priority(
            terrain_complexity=1.5,
            traversability="DRIVABLE",
            terrain_confidence=1.0,
        )


def test_invalid_traversability_is_rejected():
    with pytest.raises(ValueError):
        calculate_terrain_priority(
            terrain_complexity=0.5,
            traversability="UNKNOWN",
            terrain_confidence=1.0,
        )


def test_zero_weights_are_rejected():
    with pytest.raises(ValueError):
        calculate_terrain_priority(
            terrain_complexity=0.5,
            traversability="DRIVABLE",
            terrain_confidence=1.0,
            complexity_weight=0.0,
            risk_weight=0.0,
            uncertainty_weight=0.0,
        )