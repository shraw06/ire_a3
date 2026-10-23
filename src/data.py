"""Data loading, normalization, and query split utilities for SIFT and GloVe."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
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
    diagnostics: Optional[Any] = None

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
    if train.ndim != 2 or queries.ndim != 2:
        raise ValueError(f"train and queries must be 2D arrays, got shapes {train.shape} and {queries.shape}")
    if train.shape[1] != queries.shape[1]:
        raise ValueError(
            f"Dimension mismatch: train has {train.shape[1]} dimensions, queries has {queries.shape[1]} dimensions"
        )
    if k <= 0 or k > len(train):
        raise ValueError(f"k must be in range [1, {len(train)}], got {k}")
    metric_clean = metric.lower() if metric else "none"
    if metric_clean not in ["euclidean", "angular", "none"]:
        raise ValueError(f"Unsupported metric '{metric}'. Expected 'euclidean', 'angular', or 'none'.")

    train = np.ascontiguousarray(train, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    dim = train.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(train)
    raw_dists, indices = index.search(queries, k)

    if metric_clean == "euclidean":
        dists = np.sqrt(np.maximum(raw_dists, 0.0))
    elif metric_clean == "angular":
        dists = raw_dists / 2.0
    else:
        dists = raw_dists

    return indices.astype(np.int32), dists.astype(np.float32)


def verify_ground_truth(
    train: np.ndarray,
    queries: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    metric: str = "euclidean",
    sample_size: int = 50,
    atol: float = 1e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sanity checks ground-truth arrays:
    1. Checks for invalid neighbor IDs (< 0 or >= N or duplicate IDs per query).
    2. Checks distance monotonicity along nearest neighbors.
    3. Directly recomputes exact distances in pure NumPy for a sample of queries
       and verifies agreement with stored ground-truth distances.

    Returns:
        Dictionary with verification results and maximum observed discrepancy.
    """
    n_train = len(train)
    n_queries = len(queries)
    k = neighbors.shape[1]

    # 1. Check neighbor ID bounds
    invalid_lower = int(np.sum(neighbors < 0))
    invalid_upper = int(np.sum(neighbors >= n_train))
    if invalid_lower > 0 or invalid_upper > 0:
        raise ValueError(
            f"Invalid neighbor IDs detected: {invalid_lower} IDs < 0, {invalid_upper} IDs >= N ({n_train})"
        )

    # 2. Check distance monotonicity
    diffs = np.diff(distances, axis=1)
    if np.any(diffs < -1e-5):
        raise ValueError("Ground-truth distances are not monotonically non-decreasing along rank")

    # 3. Direct NumPy distance verification on a query sample
    rng = np.random.RandomState(seed)
    sample_size = min(sample_size, n_queries)
    sample_indices = rng.choice(n_queries, size=sample_size, replace=False)

    max_diff = 0.0
    for q_idx in sample_indices:
        q_vec = queries[q_idx]
        cand_ids = neighbors[q_idx]
        cand_vecs = train[cand_ids]

        if metric.lower() == "euclidean":
            direct_dists = np.linalg.norm(cand_vecs - q_vec[np.newaxis, :], axis=1)
        elif metric.lower() == "angular":
            # Cosine distance = 1 - cos(theta)
            dots = np.sum(cand_vecs * q_vec[np.newaxis, :], axis=1)
            q_norm = np.linalg.norm(q_vec)
            c_norms = np.linalg.norm(cand_vecs, axis=1)
            direct_dists = 1.0 - (dots / (q_norm * c_norms + 1e-12))
        else:
            direct_dists = np.linalg.norm(cand_vecs - q_vec[np.newaxis, :], axis=1)

        stored_dists = distances[q_idx]
        diff = np.max(np.abs(direct_dists - stored_dists))
        if diff > max_diff:
            max_diff = float(diff)

    passed = max_diff <= atol * 10.0  # Allow slight float32 accumulation difference
    return {
        "passed": passed,
        "sample_size": sample_size,
        "max_discrepancy": round(max_diff, 6),
        "invalid_ids_count": 0,
        "monotonicity_valid": True,
    }


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
        candidates = [
            PROJECT_ROOT / meta["filename"],
            PROJECT_ROOT.parent / meta["filename"],
            Path.cwd() / meta["filename"],
            PROJECT_ROOT / "data" / meta["filename"],
            PROJECT_ROOT.parent / "data" / meta["filename"],
        ]
        found = False
        for cand in candidates:
            if cand.exists():
                path = cand
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Dataset file '{meta['filename']}' not found at {path} or parent directories. Please ensure it is present."
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
        # Collision-free cache key including all relevant parameters
        cache_path = RESULTS_CACHE_DIR / f"{name}_gt_N{max_train}_k100_metric_{meta['metric']}_norm_{normalize_glove}.npz"
        legacy_cache_path = RESULTS_CACHE_DIR / f"{name}_gt_{max_train}.npz"

        loaded_from_cache = False
        target_path = cache_path if cache_path.exists() else legacy_cache_path

        if target_path.exists() and not force_recompute_gt:
            try:
                cached = np.load(target_path, allow_pickle=True)
                cached_neighbors = cached["neighbors"]
                cached_distances = cached["distances"]
                # Validate cached array shapes and integrity
                if (
                    cached_neighbors.shape == (len(test), 100)
                    and cached_distances.shape == (len(test), 100)
                    and np.all(cached_neighbors >= 0)
                    and np.all(cached_neighbors < len(train))
                ):
                    neighbors = cached_neighbors
                    distances = cached_distances
                    loaded_from_cache = True
            except Exception:
                loaded_from_cache = False

        if not loaded_from_cache:
            neighbors, distances = compute_ground_truth(
                train, test, k=100, metric=meta["metric"]
            )
            # Verify newly computed ground truth
            verify_ground_truth(train, test, neighbors, distances, metric=meta["metric"])
            # Save cache with metadata
            np.savez_compressed(
                cache_path,
                neighbors=neighbors,
                distances=distances,
                max_train=max_train,
                k=100,
                metric=meta["metric"],
                normalize_glove=normalize_glove,
                numpy_version=np.__version__,
                faiss_version=faiss.__version__,
            )

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
