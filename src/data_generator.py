"""Synthetic clustered dataset generator with controlled separation for Part 4."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import (
    DEFAULT_EPSILON,
    DEFAULT_K,
    PART4_CONTROL_SEPARATION,
    PART4_DIM,
    PART4_N_CLUSTERS,
    PART4_N_QUERIES,
    PART4_N_VECTORS,
    PART4_SEED,
    PART4_SEPARATIONS,
    PART4_TUNING_SPLIT_END,
    RESULTS_CACHE_DIR,
)
from src.data import DatasetSplits, compute_ground_truth


"""Synthetic clustered dataset generator with controlled separation and diagnostics for Part 4."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import faiss
import numpy as np

from src.config import (
    DEFAULT_EPSILON,
    DEFAULT_K,
    PART4_CONTROL_SEPARATION,
    PART4_DIM,
    PART4_N_CLUSTERS,
    PART4_N_QUERIES,
    PART4_N_VECTORS,
    PART4_SEED,
    PART4_SEPARATIONS,
    PART4_TUNING_SPLIT_END,
    RESULTS_CACHE_DIR,
)
from src.data import DatasetSplits, compute_ground_truth, verify_ground_truth


@dataclass
class ClusterGeometryDiagnostics:
    """Rigorous geometric and verification diagnostics for a clustered synthetic dataset."""
    separation: float
    dim: int
    n_clusters: int
    n_vectors: int
    n_queries: int
    mean_inter_cluster_dist: float
    min_inter_cluster_dist: float
    max_inter_cluster_dist: float
    mean_within_cluster_spread: float
    std_within_cluster_spread: float
    relative_separation_ratio: float
    cluster_size_min: int
    cluster_size_max: int
    cluster_size_mean: float
    cluster_size_std: float
    gt_sanity_check_passed: bool
    gt_max_discrepancy: float
    seed: int
    numpy_version: str
    faiss_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_generation_parameters(
    n_vectors: int,
    n_queries: int,
    dim: int,
    n_clusters: int,
    separations: List[float],
    tuning_split_end: int,
    seed: int,
) -> None:
    """Validates that all generation parameters are mathematically well-formed."""
    if n_vectors <= 0:
        raise ValueError(f"n_vectors must be positive, got {n_vectors}")
    if n_queries <= 0:
        raise ValueError(f"n_queries must be positive, got {n_queries}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
    if n_clusters > n_vectors:
        raise ValueError(f"n_clusters ({n_clusters}) cannot exceed database vectors ({n_vectors})")
    if tuning_split_end <= 0 or tuning_split_end >= n_queries:
        raise ValueError(
            f"tuning_split_end must be strictly between 1 and {n_queries - 1}, got {tuning_split_end}"
        )
    if not separations:
        raise ValueError("separations list cannot be empty")
    for s in separations:
        if s <= 0:
            raise ValueError(f"Separation multiplier must be strictly positive, got {s}")
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")


def compute_geometry_diagnostics(
    train: np.ndarray,
    queries: np.ndarray,
    gt_neighbors: np.ndarray,
    gt_distances: np.ndarray,
    base_centers: np.ndarray,
    train_assignments: np.ndarray,
    separation: float,
    dim: int,
    n_clusters: int,
    seed: int,
) -> ClusterGeometryDiagnostics:
    """
    Computes rigorous geometric diagnostics and sanity checks:
    1. Inter-cluster distance: Mean, min, max pairwise Euclidean distance between cluster centres.
    2. Within-cluster spread: Mean and std distance of database vectors to their assigned centre.
    3. Relative separation ratio: Mean inter-cluster distance / mean within-cluster spread.
    4. Cluster sizes: Min, max, mean, std of vector counts across clusters.
    5. Ground-truth sanity check: Verifies ID bounds and direct NumPy distance agreement on query sample.
    """
    # Scaled cluster centres
    centers = (separation * base_centers).astype(np.float32)

    # 1. Pairwise inter-cluster distances
    # Distance matrix between all pairs of centres
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    upper_tri = np.triu_indices(n_clusters, k=1)
    pair_dists = dist_matrix[upper_tri]
    mean_inter = float(np.mean(pair_dists))
    min_inter = float(np.min(pair_dists))
    max_inter = float(np.max(pair_dists))

    # 2. Within-cluster spread
    assigned_centers = centers[train_assignments]
    spreads = np.linalg.norm(train - assigned_centers, axis=1)
    mean_within = float(np.mean(spreads))
    std_within = float(np.std(spreads))

    # 3. Relative separation ratio
    rel_ratio = float(mean_inter / mean_within) if mean_within > 0 else 0.0

    # 4. Cluster size distributions
    counts = np.bincount(train_assignments, minlength=n_clusters)
    size_min = int(np.min(counts))
    size_max = int(np.max(counts))
    size_mean = float(np.mean(counts))
    size_std = float(np.std(counts))

    # 5. Ground-truth sanity check
    gt_res = verify_ground_truth(
        train=train,
        queries=queries,
        neighbors=gt_neighbors,
        distances=gt_distances,
        metric="euclidean",
        sample_size=50,
        seed=seed,
    )

    return ClusterGeometryDiagnostics(
        separation=separation,
        dim=dim,
        n_clusters=n_clusters,
        n_vectors=len(train),
        n_queries=len(queries),
        mean_inter_cluster_dist=round(mean_inter, 4),
        min_inter_cluster_dist=round(min_inter, 4),
        max_inter_cluster_dist=round(max_inter, 4),
        mean_within_cluster_spread=round(mean_within, 4),
        std_within_cluster_spread=round(std_within, 4),
        relative_separation_ratio=round(rel_ratio, 4),
        cluster_size_min=size_min,
        cluster_size_max=size_max,
        cluster_size_mean=round(size_mean, 2),
        cluster_size_std=round(size_std, 2),
        gt_sanity_check_passed=gt_res["passed"],
        gt_max_discrepancy=gt_res["max_discrepancy"],
        seed=seed,
        numpy_version=np.__version__,
        faiss_version=faiss.__version__,
    )


def verify_dataset_consistency(
    datasets: Dict[float, DatasetSplits],
    base_centers: np.ndarray,
    train_assignments: np.ndarray,
    query_assignments: np.ndarray,
) -> Dict[str, Any]:
    """
    Verifies dataset consistency across separation settings:
    - Verifies matching array shapes and dimensions across all datasets.
    - Mathematically asserts that:
        (X_{s2} - X_{s1}) == (s2 - s1) * base_centers[train_assignments]
      proving that cluster assignments and random noise are strictly preserved.
    """
    seps = sorted(datasets.keys())
    if len(seps) < 2:
        return {"passed": True, "num_settings": len(seps), "max_residual": 0.0}

    ref_dim = datasets[seps[0]].dim
    ref_n_train = len(datasets[seps[0]].train)
    ref_n_queries = len(datasets[seps[0]].test_full)

    max_residual = 0.0
    for s in seps:
        ds = datasets[s]
        if ds.dim != ref_dim:
            raise ValueError(f"Dimension mismatch in setting s={s}: expected {ref_dim}, got {ds.dim}")
        if len(ds.train) != ref_n_train:
            raise ValueError(f"Train size mismatch in setting s={s}: expected {ref_n_train}, got {len(ds.train)}")
        if len(ds.test_full) != ref_n_queries:
            raise ValueError(f"Query size mismatch in setting s={s}: expected {ref_n_queries}, got {len(ds.test_full)}")

    # Check pairwise assignment & noise invariant: (X2 - X1) == (s2 - s1) * C
    for i in range(len(seps) - 1):
        s1 = seps[i]
        s2 = seps[i + 1]
        X1 = datasets[s1].train
        X2 = datasets[s2].train
        expected_diff = (s2 - s1) * base_centers[train_assignments]
        actual_diff = X2 - X1
        res = float(np.max(np.abs(actual_diff - expected_diff)))
        if res > max_residual:
            max_residual = res

    passed = max_residual < 1e-4
    if not passed:
        raise ValueError(
            f"Dataset consistency check failed: assignments/noise not shared (max residual {max_residual})"
        )

    return {
        "passed": passed,
        "num_settings": len(seps),
        "max_residual": round(max_residual, 6),
    }


def generate_clustered_dataset(
    n_vectors: int = PART4_N_VECTORS,
    n_queries: int = PART4_N_QUERIES,
    dim: int = PART4_DIM,
    n_clusters: int = PART4_N_CLUSTERS,
    separations: Optional[List[float]] = None,
    seed: int = PART4_SEED,
    tuning_split_end: Optional[int] = None,
    force_recompute: bool = False,
    cache_dir: Optional[Path] = None,
    return_diagnostics: bool = False,
) -> Union[Dict[float, DatasetSplits], Tuple[Dict[float, DatasetSplits], Dict[float, ClusterGeometryDiagnostics]]]:
    """
    Generates synthetic clustered datasets with controlled cluster separation.

    Crucially, cluster assignments and Gaussian noise are generated ONCE and
    reused across all separation multipliers, ensuring that ONLY the inter-cluster
    distance varies between settings.

    Args:
        n_vectors: Number of database vectors per setting (e.g. 100,000).
        n_queries: Number of query vectors (e.g. 2,000: 1,000 tune, 1,000 eval).
        dim: Dimensionality of vectors (e.g. 64).
        n_clusters: Number of Gaussian clusters (e.g. 20).
        separations: List of separation multipliers (e.g. [0.5, 1.0, 2.0, 4.0]).
        seed: Random seed for exact reproducibility.
        tuning_split_end: Index at which to split tune and eval queries.
        force_recompute: If True, regenerates data and ground truth even if cached.
        cache_dir: Directory for storing cached .npz files.
        return_diagnostics: If True, returns (datasets, diagnostics_dict).

    Returns:
        Dictionary mapping separation multiplier -> DatasetSplits object,
        or tuple of (datasets, diagnostics_dict) if return_diagnostics=True.
    """
    if separations is None:
        separations = PART4_SEPARATIONS

    if cache_dir is None:
        cache_dir = RESULTS_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if tuning_split_end is None:
        tuning_split_end = n_queries // 2

    # 1. Rigorous input validation
    validate_generation_parameters(
        n_vectors=n_vectors,
        n_queries=n_queries,
        dim=dim,
        n_clusters=n_clusters,
        separations=separations,
        tuning_split_end=tuning_split_end,
        seed=seed,
    )

    # 2. Deterministic generation of shared geometric components
    rng = np.random.RandomState(seed)

    # Base cluster centers on a hypersphere shell of radius R = 3 * sqrt(dim)
    base_centers = rng.randn(n_clusters, dim).astype(np.float32)
    norms = np.linalg.norm(base_centers, axis=1, keepdims=True)
    base_centers = (base_centers / norms) * (np.sqrt(dim) * 3.0)

    # Shared cluster assignments
    train_assignments = rng.randint(0, n_clusters, size=n_vectors)
    query_assignments = rng.randint(0, n_clusters, size=n_queries)

    # Shared within-cluster Gaussian noise (unit variance)
    train_noise = rng.randn(n_vectors, dim).astype(np.float32)
    query_noise = rng.randn(n_queries, dim).astype(np.float32)

    datasets: Dict[float, DatasetSplits] = {}
    diagnostics_dict: Dict[float, ClusterGeometryDiagnostics] = {}

    for s in separations:
        # Collision-free cache filename including dim and n_clusters
        cache_file = cache_dir / f"synthetic_sep_{s}_N{n_vectors}_Q{n_queries}_D{dim}_K{n_clusters}_seed{seed}.npz"
        legacy_cache_file = cache_dir / f"synthetic_sep_{s}_N{n_vectors}_Q{n_queries}_seed{seed}.npz"

        target_cache = cache_file if cache_file.exists() else legacy_cache_file
        loaded_from_cache = False

        if target_cache.exists() and not force_recompute:
            try:
                cached = np.load(target_cache, allow_pickle=True)
                c_train = cached["train"]
                c_queries = cached["queries"]
                c_neighbors = cached["gt_neighbors"]
                c_distances = cached["gt_distances"]

                # Validate shapes and data bounds
                if (
                    c_train.shape == (n_vectors, dim)
                    and c_queries.shape == (n_queries, dim)
                    and c_neighbors.shape == (n_queries, 100)
                    and c_distances.shape == (n_queries, 100)
                    and np.all(c_neighbors >= 0)
                    and np.all(c_neighbors < n_vectors)
                ):
                    # Check metadata if present
                    if "seed" in cached and int(cached["seed"]) != seed:
                        raise ValueError("Cached seed mismatch")
                    if "dim" in cached and int(cached["dim"]) != dim:
                        raise ValueError("Cached dim mismatch")
                    if "n_clusters" in cached and int(cached["n_clusters"]) != n_clusters:
                        raise ValueError("Cached n_clusters mismatch")

                    train = c_train
                    queries = c_queries
                    gt_neighbors = c_neighbors
                    gt_distances = c_distances
                    loaded_from_cache = True
            except Exception:
                loaded_from_cache = False

        if not loaded_from_cache:
            # Apply separation multiplier: only the cluster center distance changes!
            train = (s * base_centers[train_assignments] + train_noise).astype(np.float32)
            queries = (s * base_centers[query_assignments] + query_noise).astype(np.float32)

            # Compute exact ground truth for this specific geometry
            gt_neighbors, gt_distances = compute_ground_truth(
                train=train,
                queries=queries,
                k=100,
                metric="euclidean",
            )

            # Cache arrays along with reproducibility metadata
            np.savez_compressed(
                cache_file,
                train=train,
                queries=queries,
                gt_neighbors=gt_neighbors,
                gt_distances=gt_distances,
                seed=seed,
                dim=dim,
                n_clusters=n_clusters,
                n_vectors=n_vectors,
                n_queries=n_queries,
                separation=s,
                k=100,
                metric="euclidean",
                numpy_version=np.__version__,
                faiss_version=faiss.__version__,
            )

        # Compute diagnostics for this dataset setting
        diag = compute_geometry_diagnostics(
            train=train,
            queries=queries,
            gt_neighbors=gt_neighbors,
            gt_distances=gt_distances,
            base_centers=base_centers,
            train_assignments=train_assignments,
            separation=s,
            dim=dim,
            n_clusters=n_clusters,
            seed=seed,
        )
        diagnostics_dict[s] = diag

        test_tune = np.ascontiguousarray(queries[:tuning_split_end], dtype=np.float32)
        test_eval = np.ascontiguousarray(queries[tuning_split_end:], dtype=np.float32)

        gt_neighbors_tune = np.ascontiguousarray(gt_neighbors[:tuning_split_end], dtype=np.int32)
        gt_neighbors_eval = np.ascontiguousarray(gt_neighbors[tuning_split_end:], dtype=np.int32)

        gt_distances_tune = np.ascontiguousarray(gt_distances[:tuning_split_end], dtype=np.float32)
        gt_distances_eval = np.ascontiguousarray(gt_distances[tuning_split_end:], dtype=np.float32)

        dataset_name = f"synthetic_sep_{s}"
        datasets[s] = DatasetSplits(
            name=dataset_name,
            dim=dim,
            train=train,
            test_tune=test_tune,
            test_eval=test_eval,
            gt_neighbors_tune=gt_neighbors_tune,
            gt_neighbors_eval=gt_neighbors_eval,
            gt_distances_tune=gt_distances_tune,
            gt_distances_eval=gt_distances_eval,
            diagnostics=diag,
        )

    # 3. Overall Dataset Consistency Verification across settings
    verify_dataset_consistency(
        datasets=datasets,
        base_centers=base_centers,
        train_assignments=train_assignments,
        query_assignments=query_assignments,
    )

    if return_diagnostics:
        return datasets, diagnostics_dict
    return datasets
