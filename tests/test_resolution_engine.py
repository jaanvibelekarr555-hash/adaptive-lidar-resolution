import numpy as np
import pytest

from src.resolution_engine import (
    distance_array_to_resolution,
    distance_to_resolution,
)


def test_distance_to_resolution_at_zero():
    assert distance_to_resolution(0.0) == 0.05


def test_distance_to_resolution_near_10_boundary():
    assert distance_to_resolution(9.999999) == 0.05
    assert distance_to_resolution(10.0) == 0.10
    assert distance_to_resolution(10.000001) == 0.10


def test_distance_to_resolution_near_25_boundary():
    assert distance_to_resolution(24.999999) == 0.10
    assert distance_to_resolution(25.0) == 0.25
    assert distance_to_resolution(25.000001) == 0.25


def test_distance_to_resolution_near_50_boundary():
    assert distance_to_resolution(49.999999) == 0.25
    assert distance_to_resolution(50.0) == 0.50
    assert distance_to_resolution(50.000001) == 0.50


def test_distance_to_resolution_at_100():
    assert distance_to_resolution(100.0) == 0.50


def test_distance_to_resolution_rejects_negative():
    with pytest.raises(ValueError):
        distance_to_resolution(-0.001)


def test_distance_to_resolution_rejects_above_max_range():
    with pytest.raises(ValueError):
        distance_to_resolution(100.000001)


def test_distance_to_resolution_rejects_non_finite():
    with pytest.raises(ValueError):
        distance_to_resolution(float("inf"))

    with pytest.raises(ValueError):
        distance_to_resolution(float("nan"))


def test_vectorized_distance_to_resolution():
    distances = np.array(
        [0.0, 9.999999, 10.0, 24.999999, 25.0, 49.999999, 50.0, 100.0]
    )

    expected = np.array(
        [0.05, 0.05, 0.10, 0.10, 0.25, 0.25, 0.50, 0.50]
    )

    result = distance_array_to_resolution(distances)

    np.testing.assert_array_equal(result, expected)


def test_vectorized_distance_to_resolution_rejects_invalid_values():
    with pytest.raises(ValueError):
        distance_array_to_resolution(np.array([-1.0, 5.0]))

    with pytest.raises(ValueError):
        distance_array_to_resolution(np.array([5.0, 101.0]))


def test_vectorized_distance_to_resolution_rejects_non_1d_array():
    with pytest.raises(ValueError):
        distance_array_to_resolution(
            np.array([[5.0, 10.0]])
        )