import numpy as np
import pytest

from src.terrain_complexity import (
    normalize_feature,
    calculate_terrain_complexity,
)


def test_normalize_feature():
    assert np.isclose(
        normalize_feature(0.5, 1.0),
        0.5,
    )


def test_normalize_feature_caps_at_one():
    assert normalize_feature(2.0, 1.0) == 1.0


def test_flat_terrain_has_zero_complexity():
    complexity = calculate_terrain_complexity(
        slope=0.0,
        roughness=0.0,
        elevation_variation=0.0,
    )

    assert np.isclose(
        complexity,
        0.0,
    )


def test_hilly_terrain_has_higher_complexity():
    flat = calculate_terrain_complexity(
        slope=2.0,
        roughness=0.01,
        elevation_variation=0.03,
    )

    hilly = calculate_terrain_complexity(
        slope=20.0,
        roughness=0.10,
        elevation_variation=0.30,
    )

    assert hilly > flat


def test_highly_complex_terrain_approaches_one():
    complexity = calculate_terrain_complexity(
        slope=50.0,
        roughness=0.50,
        elevation_variation=1.00,
    )

    assert np.isclose(
        complexity,
        1.0,
    )


def test_custom_weights():
    complexity = calculate_terrain_complexity(
        slope=30.0,
        roughness=0.0,
        elevation_variation=0.0,
        weights=(1.0, 0.0, 0.0),
    )

    assert np.isclose(
        complexity,
        1.0,
    )


def test_invalid_feature_value():
    with pytest.raises(ValueError):
        calculate_terrain_complexity(
            slope=-1.0,
            roughness=0.1,
            elevation_variation=0.1,
        )


def test_invalid_weights():
    with pytest.raises(ValueError):
        calculate_terrain_complexity(
            slope=10.0,
            roughness=0.1,
            elevation_variation=0.1,
            weights=(0.0, 0.0, 0.0),
        )