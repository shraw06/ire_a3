"""Evaluation metrics for approximate and exact nearest neighbour search.

Implements distance-threshold recall@k using distances returned by the index (FAISS)
and ground-truth distances, with consistent metric alignment and tolerance.

A retrieved candidate is counted as a valid nearest neighbour if:
    dist(q, p) <= d_k + eps
where d_k = gt_distances[k - 1] is the distance of the k-th true neighbour.

This guarantees that alternative database vectors tied at the k-th neighbour boundary
are credited as valid hits without requiring an ID-set fallback.
"""

from typing import Optional, Tuple, Union
import numpy as np


def align_faiss_distances(
    faiss_distances: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Converts distances returned by FAISS IndexFlatL2 to the native dataset metric:
    - 'euclidean': FAISS returns squared L2 (||u - v||^2). Native metric is L2 (sqrt).
    - 'angular': On unit-normalized vectors, FAISS returns ||u - v||^2 = 2 * (1 - cos(theta)).
                 Native metric is angular distance (1 - cos(theta)) = faiss_dist / 2.0.

    Args:
        faiss_distances: Distances array returned directly by FAISS search.
        metric: 'euclidean', 'angular', or None (no-op).

    Returns:
        Array of candidate distances aligned to the dataset ground-truth metric.
    """
    faiss_distances = np.asarray(faiss_distances, dtype=np.float32)
    metric = metric.lower() if metric else "none"

    if metric == "euclidean":
        return np.sqrt(np.maximum(faiss_distances, 0.0))
    elif metric == "angular":
        return faiss_distances / 2.0
    elif metric == "none":
        return faiss_distances
    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected 'euclidean', 'angular', or 'none'.")


def query_recall_at_k(
    retrieved_distances: np.ndarray,
    gt_distances: np.ndarray,
    retrieved_indices: Optional[np.ndarray] = None,
    metric: Optional[str] = None,
    k: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Computes recall@k for a single query using distance thresholding.

    Args:
        retrieved_distances: 1D array of candidate distances (length >= k).
        gt_distances: 1D array of ground-truth distances (sorted ascending, length >= k).
        retrieved_indices: Optional 1D array of candidate IDs used to prevent duplicate counting.
        metric: Optional metric name ('euclidean', 'angular') to align FAISS distances.
        k: Rank threshold (default 10).
        eps: Tolerance for floating-point distance comparisons.

    Returns:
        Recall score in [0.0, 1.0].
    """
    retrieved_distances = np.asarray(retrieved_distances, dtype=np.float32)[:k]
    gt_distances = np.asarray(gt_distances, dtype=np.float32)

    if len(retrieved_distances) < k or len(gt_distances) < k:
        raise ValueError(f"Inputs must have at least k={k} elements")

    # Align distances to ground-truth metric if metric is specified
    if metric is not None and metric.lower() != "none":
        retrieved_distances = align_faiss_distances(retrieved_distances, metric=metric)

    # Threshold distance at the k-th neighbor (0-indexed position k-1)
    threshold = gt_distances[k - 1] + eps

    # Deduplicate candidate IDs if provided to ensure unique hits
    if retrieved_indices is not None:
        retrieved_indices = np.asarray(retrieved_indices)[:k]
        seen = set()
        hits = 0
        for idx, dist in zip(retrieved_indices, retrieved_distances):
            if idx not in seen:
                seen.add(idx)
                if dist <= threshold:
                    hits += 1
    else:
        hits = int(np.sum(retrieved_distances <= threshold))

    return min(1.0, hits / float(k))


def recall_at_k(
    retrieved_distances: np.ndarray,
    gt_distances: np.ndarray,
    retrieved_indices: Optional[np.ndarray] = None,
    metric: Optional[str] = None,
    k: int = 10,
    eps: float = 1e-4,
    return_per_query: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Computes recall@k across a batch of queries using distance thresholding.

    Args:
        retrieved_distances: (N, >=k) or (>=k,) array of candidate distances.
        gt_distances: (N, >=k) or (>=k,) array of ground-truth distances.
        retrieved_indices: Optional (N, >=k) array of candidate IDs for deduplication.
        metric: Optional metric ('euclidean', 'angular') to align raw FAISS distances.
        k: Rank threshold (default: 10).
        eps: Floating point comparison tolerance.
        return_per_query: If True, returns (mean_recall, per_query_recalls).

    Returns:
        Mean recall score across queries, or tuple of (mean_recall, per_query_recalls).
    """
    retrieved_distances = np.asarray(retrieved_distances, dtype=np.float32)
    gt_distances = np.asarray(gt_distances, dtype=np.float32)

    # Handle 1D input (single query)
    if retrieved_distances.ndim == 1:
        score = query_recall_at_k(
            retrieved_distances=retrieved_distances,
            gt_distances=gt_distances,
            retrieved_indices=retrieved_indices,
            metric=metric,
            k=k,
            eps=eps,
        )
        if return_per_query:
            return score, np.array([score], dtype=np.float64)
        return score

    # Align batch of distances if metric is specified
    if metric is not None and metric.lower() != "none":
        retrieved_distances = align_faiss_distances(retrieved_distances, metric=metric)

    n_queries = len(retrieved_distances)
    recalls = np.zeros(n_queries, dtype=np.float64)

    # Check if we can vectorize across all queries (when no duplicate check is needed)
    if retrieved_indices is None:
        thresholds = gt_distances[:, k - 1] + eps
        hits = np.sum(retrieved_distances[:, :k] <= thresholds[:, None], axis=1)
        recalls = np.minimum(1.0, hits / float(k))
    else:
        retrieved_indices = np.asarray(retrieved_indices)
        for i in range(n_queries):
            recalls[i] = query_recall_at_k(
                retrieved_distances=retrieved_distances[i],
                gt_distances=gt_distances[i],
                retrieved_indices=retrieved_indices[i],
                metric=None,  # Already aligned above
                k=k,
                eps=eps,
            )

    mean_recall = float(np.mean(recalls))
    if return_per_query:
        return mean_recall, recalls
    return mean_recall
