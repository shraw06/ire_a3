"""Data loading, normalization, and query split utilities for SIFT and GloVe."""

from dataclasses import dataclass
from typing import Optional, Tuple
import h5py
import numpy as np
import faiss

from src.config import DATASET_METADATA, TUNING_SPLIT_END, RESULTS_CACHE_DIR


@dataclass
class DatasetSplits:
    """Holds dataset training vectors and split test queries with ground-truth."""
    name: str
    dim: int
    train: np.ndarray
    test_tune: np.ndarray
    test_eval: np.ndarray
    gt_neighbors_tune: np.ndarray
    gt_neighbors_eval: np.ndarray
    gt_distances_tune: np.ndarray
    gt_distances_eval: np.ndarray

    @property
    def test_full(self) -> np.ndarray:
        return np.vstack([self.test_tune, self.test_eval])

    @property
    def gt_neighbors_full(self) -> np.ndarray:
        return np.vstack([self.gt_neighbors_tune, self.gt_neighbors_eval])

    @property
    def gt_distances_full(self) -> np.ndarray:
        return np.vstack([self.gt_distances_tune, self.gt_distances_eval])


def normalize_vectors(v: np.ndarray) -> np.ndarray:
    """In-place L2 normalization of vectors using FAISS or numpy fallback."""
    v = np.ascontiguousarray(v, dtype=np.float32)
    faiss.normalize_L2(v)
    return v


def compute_ground_truth(
    train: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes exact brute-force ground-truth nearest neighbors and distances for queries.

    Args:
        train: Database vectors of shape (N, D).
        queries: Query vectors of shape (Q, D).
        k: Number of nearest neighbors to return (default 100).
        metric: 'euclidean' or 'angular'.

    Returns:
        Tuple of (neighbors, distances):
            neighbors: int32 array of shape (Q, k)
            distances: float32 array of shape (Q, k) in aligned metric
    """
    train = np.ascontiguousarray(train, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    dim = train.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(train)
    raw_dists, indices = index.search(queries, k)

    if metric == "euclidean":
        dists = np.sqrt(np.maximum(raw_dists, 0.0))
    elif metric == "angular":
        dists = raw_dists / 2.0
    else:
        dists = raw_dists

    return indices.astype(np.int32), dists.astype(np.float32)


def load_dataset(
    name: str,
    max_train: Optional[int] = None,
    normalize_glove: bool = True,
    tuning_split_end: int = TUNING_SPLIT_END,
    force_recompute_gt: bool = False,
) -> DatasetSplits:
    """
    Loads dataset vectors and pre-computed nearest neighbours.
    When max_train is specified and less than the total vectors, computes and caches
    exact ground truth specifically for the subset to avoid using 1M-specific ground truth.

    Args:
        name: 'sift' or 'glove'
        max_train: Optional slice limit for train vectors (e.g. 100_000, 300_000)
        normalize_glove: If True and dataset is glove, normalizes train and test vectors
        tuning_split_end: Index at which to split tuning and evaluation queries (default: 5000)
        force_recompute_gt: If True, recalculates ground truth even if cache exists.

    Returns:
        DatasetSplits dataclass instance containing train, tune, and eval data.
    """
    name = name.lower()
    if name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_METADATA.keys())}")

    meta = DATASET_METADATA[name]
    path = meta["path"]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {path}. Please ensure it is present in the repository."
        )

    with h5py.File(path, "r") as f:
        train = np.array(f["train"][:max_train], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        if max_train is None or max_train >= meta["num_train"]:
            neighbors = np.array(f["neighbors"], dtype=np.int32)
            distances = np.array(f["distances"], dtype=np.float32)
        else:
            neighbors = None
            distances = None

    # Normalization for GloVe (angular distance on unit sphere corresponds to Euclidean distance)
    if name == "glove" and normalize_glove:
        normalize_vectors(train)
        normalize_vectors(test)

    train = np.ascontiguousarray(train, dtype=np.float32)
    test = np.ascontiguousarray(test, dtype=np.float32)

    # For subset training vectors, ensure ground truth is computed specifically for this subset
    if neighbors is None or distances is None:
        RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = RESULTS_CACHE_DIR / f"{name}_gt_{max_train}.npz"
        if cache_path.exists() and not force_recompute_gt:
            cached = np.load(cache_path)
            neighbors = cached["neighbors"]
            distances = cached["distances"]
        else:
            neighbors, distances = compute_ground_truth(
                train, test, k=100, metric=meta["metric"]
            )
            np.savez_compressed(cache_path, neighbors=neighbors, distances=distances)

    test_tune = np.ascontiguousarray(test[:tuning_split_end], dtype=np.float32)
    test_eval = np.ascontiguousarray(test[tuning_split_end:], dtype=np.float32)

    gt_neighbors_tune = np.ascontiguousarray(neighbors[:tuning_split_end], dtype=np.int32)
    gt_neighbors_eval = np.ascontiguousarray(neighbors[tuning_split_end:], dtype=np.int32)

    gt_distances_tune = np.ascontiguousarray(distances[:tuning_split_end], dtype=np.float32)
    gt_distances_eval = np.ascontiguousarray(distances[tuning_split_end:], dtype=np.float32)

    return DatasetSplits(
        name=name,
        dim=meta["dim"],
        train=train,
        test_tune=test_tune,
        test_eval=test_eval,
        gt_neighbors_tune=gt_neighbors_tune,
        gt_neighbors_eval=gt_neighbors_eval,
        gt_distances_tune=gt_distances_tune,
        gt_distances_eval=gt_distances_eval,
    )
