"""Configuration and dataset paths for IRE Assignment 3."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"

# Dataset paths
SIFT_FILE = "sift-128-euclidean.hdf5"
GLOVE_FILE = "glove-100-angular.hdf5"

SIFT_PATH = DATA_DIR / SIFT_FILE
GLOVE_PATH = DATA_DIR / GLOVE_FILE

# Dataset specifications
DATASET_METADATA = {
    "sift": {
        "filename": SIFT_FILE,
        "path": SIFT_PATH,
        "dim": 128,
        "metric": "euclidean",
        "num_train": 1_000_000,
        "num_queries": 10_000,
        "normalize": False,
    },
    "glove": {
        "filename": GLOVE_FILE,
        "path": GLOVE_PATH,
        "dim": 100,
        "metric": "angular",
        "num_train": 1_183_514,
        "num_queries": 10_000,
        "normalize": True,
    },
}

# Query split parameters (first 5,000 tuning, last 5,000 evaluation)
TUNING_SPLIT_END = 5_000
TOTAL_QUERIES = 10_000

# Evaluation parameters
DEFAULT_K = 10
DEFAULT_EPSILON = 1e-4  # Tolerance for floating point distance ties
