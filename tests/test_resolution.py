import numpy as np
import pytest

from src.resolution_engine import distance_array_to_resolution, distance_to_resolution


def test_distance_to_resolution_at_zero():
    assert distance_to_resolution(0.0) == 0.05


def test_distance_to_resolution_first_boundary():
    assert distance_to_resolution(9.999999) == 0.05
    assert distance_to_resolution(10.0) == 0.10
    assert distance_to_resolution(10.000001) == 0.10


def test_distance_to_resolution_second_boundary():
    assert distance_to_resolution(24.999999) == 0.10
    assert distance_to_resolution(25.0) == 0.25
    assert distance_to_resolution(25.000001) == 0.25


def test_distance_to_resolution_third_boundary():
    assert distance_to_resolution(49.999999) == 0.25
    assert distance_to_resolution(50.0) == 0.50
    assert distance_to_resolution(50.000001) == 0.50


def test_distance_to_resolution_at_max_range():
    assert distance_to_resolution(100.0) == 0.50


def test_distance_to_resolution_example():
    assert distance_to_resolution(18.3) == 0.10


def test_distance_to_resolution_rejects_negative():
    with pytest.raises(ValueError):
        distance_to_resolution(-1.0)


def test_distance_to_resolution_rejects_above_max_range():
    with pytest.raises(ValueError):
        distance_to_resolution(100.000001)


def test_distance_to_resolution_rejects_non_finite():
    with pytest.raises(ValueError):
        distance_to_resolution(np.inf)

    with pytest.raises(ValueError):
        distance_to_resolution(np.nan)


def test_distance_to_resolution_vectorized():
    distances = np.array([
        5.0,
        18.3,
        30.0,
        75.0,
        100.0,
    ])

    expected = np.array([
        0.05,
        0.10,
        0.25,
        0.50,
        0.50,
    ])

    result = distance_array_to_resolution(distances)

    np.testing.assert_array_equal(result, expected)
