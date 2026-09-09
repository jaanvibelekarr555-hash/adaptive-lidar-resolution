import numpy as np

SAMPLE_POINTS = np.array([
    [1.0, 1.0, 0.10, 0.50],
    [1.1, 1.0, 0.11, 0.45],
    [1.0, 1.1, 0.09, 0.60],
    [5.0, 5.0, 1.00, 0.30],
    [5.1, 5.0, 1.02, 0.35],
    [5.0, 5.1, 1.01, 0.40],
], dtype=np.float32)

SAMPLE_LABELS = np.array([
    40,
    40,
    40,
    10,
    10,
    10
], dtype=np.uint32)