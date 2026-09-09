DATA_ROOT = "data/semantic_kitti"
SEQUENCE = "00"
FRAME = "000000"

RESULTS_DIR = "results"

DISTANCE_BANDS = [
    (0.0, 10.0, 0.10),
    (10.0, 30.0, 0.20),
    (30.0, 50.0, 0.35),
    (50.0, float("inf"), 0.50),
]