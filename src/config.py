"""Configuration, dataset paths, and parameter sweep grids for IRE Assignment 3."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
SWEEPS_DIR = FIGURES_DIR / "sweeps"

RESULTS_CACHE_DIR = RESULTS_DIR / "cache"
PART3_FIGURES_DIR = FIGURES_DIR / "part3"
PART4_FIGURES_DIR = FIGURES_DIR / "part4"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
PART3_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PART4_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

# --------------------------------------------------------------------------
# Part 2: Index Build and Search Parameter Sweep Grids
# --------------------------------------------------------------------------

# HNSW
HNSW_M_LIST = [8, 16, 32]
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH_LIST = [8, 16, 32, 64, 128, 256, 512]

# IVF-Flat
IVF_FLAT_NLIST_LIST = [1024, 2048, 4096]
IVF_FLAT_NPROBE_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# IVF-PQ
IVF_PQ_NLIST = 1024
IVF_PQ_NBITS = 8
SIFT_PQ_M_LIST = [16, 32, 64]      # SIFT d=128 (8x, 4x, 2x compression)
GLOVE_PQ_M_LIST = [10, 20, 25, 50]  # GloVe d=100 (10x, 5x, 4x, 2x compression)
IVF_PQ_NPROBE_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Fast / Quick mode settings (for rapid verification in under 5 minutes)
QUICK_HNSW_M_LIST = [8, 16]
QUICK_HNSW_EF_SEARCH = [8, 32, 128]
QUICK_IVF_NLIST_LIST = [512, 1024]
QUICK_IVF_NPROBE = [1, 8, 32, 128]
QUICK_SIFT_PQ_M = [16, 32]
QUICK_GLOVE_PQ_M = [10, 20]

# --------------------------------------------------------------------------
# Part 3: Scaling Experiment (SIFT 100K, 300K, 1M)
# --------------------------------------------------------------------------
SCALING_DATASET_SIZES = [100_000, 300_000, 1_000_000]

PART3_HNSW_M_LIST = [8, 16, 32]
PART3_HNSW_EF_CONSTRUCTION = 64
PART3_HNSW_EF_SEARCH_LIST = [8, 16, 32, 64, 128, 256, 512]

PART3_IVF_NLIST_MAP = {
    100_000: [512, 1024, 2048],
    300_000: [512, 1024, 2048],
    1_000_000: [1024, 2048, 4096],
}
PART3_IVF_NPROBE_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]

QUICK_PART3_SIZES = [100_000]
QUICK_PART3_HNSW_M = [8, 16]
QUICK_PART3_HNSW_EF_SEARCH = [8, 32, 128]
QUICK_PART3_IVF_NLIST = [512, 1024]
QUICK_PART3_IVF_NPROBE = [1, 8, 32, 128]

# --------------------------------------------------------------------------
# Part 4: Hard Geometry (Cluster Separation) Experiment
# --------------------------------------------------------------------------
PART4_DIM = 64
PART4_N_CLUSTERS = 20
PART4_N_VECTORS = 100_000
PART4_N_QUERIES = 2_000
PART4_TUNING_SPLIT_END = 1_000  # 1,000 tune, 1,000 eval
PART4_SEPARATIONS = [0.5, 1.0, 2.0, 4.0]
PART4_CONTROL_SEPARATION = 1.0
PART4_SEED = 42

# Fixed build parameters for Part 4
PART4_HNSW_M = 16
PART4_HNSW_EF_CONSTRUCTION = 64
PART4_IVF_NLIST = 512

# Search parameter sweeps for Part 4 (refined around threshold: efSearch=[80, 96, 112], nprobe=[17, 20, 24, 28])
PART4_HNSW_EF_SEARCH_LIST = [4, 8, 16, 32, 64, 80, 96, 112, 128, 256, 512]
PART4_IVF_NPROBE_LIST = [1, 2, 4, 8, 16, 17, 20, 24, 28, 32, 64, 128, 256]

# Quick mode settings for Part 4
QUICK_PART4_N_VECTORS = 10_000
QUICK_PART4_N_QUERIES = 500
QUICK_PART4_TUNING_SPLIT_END = 250
QUICK_PART4_SEPARATIONS = [0.5, 1.0, 4.0]
QUICK_PART4_IVF_NLIST = 128
QUICK_PART4_HNSW_EF_SEARCH = [4, 16, 64, 256]
QUICK_PART4_IVF_NPROBE = [1, 4, 16, 64]

