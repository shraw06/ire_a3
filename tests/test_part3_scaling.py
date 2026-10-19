"""Unit and integration tests for Part 3 Scaling Experiment and subset ground truth computation."""

import numpy as np
import pytest

from src.data import compute_ground_truth, load_dataset
from src.metrics import recall_at_k
from src.index.exact import ExactFlatIndex
from src.config import RESULTS_CACHE_DIR


def test_compute_ground_truth_synthetic():
    """Verify compute_ground_truth against exact numpy pairwise distances."""
    np.random.seed(42)
    N, Q, D = 100, 10, 16
    train = np.random.randn(N, D).astype(np.float32)
    queries = np.random.randn(Q, D).astype(np.float32)

    k = 10
    indices, dists = compute_ground_truth(train, queries, k=k, metric="euclidean")

    assert indices.shape == (Q, k)
    assert dists.shape == (Q, k)

    # Brute-force numpy calculation
    for q_idx in range(Q):
        diff = train - queries[q_idx]
        np_dists = np.linalg.norm(diff, axis=1)
        np_sorted_indices = np.argsort(np_dists)[:k]
        np_sorted_dists = np_dists[np_sorted_indices]

        # Check distances match within floating-point tolerance
        np.testing.assert_allclose(dists[q_idx], np_sorted_dists, rtol=1e-5, atol=1e-5)
        # Check indices match (assuming no distance ties in random floats)
        np.testing.assert_array_equal(indices[q_idx], np_sorted_indices)


def test_subset_ground_truth_id_bounds_and_exact_agreement():
    """
    Verify that subset ground truth IDs never exceed N-1 and that
    ExactFlatIndex on the subset achieves 100% recall@10 against the subset ground truth.
    """
    subset_size = 5_000
    splits = load_dataset("sift", max_train=subset_size)

    assert len(splits.train) == subset_size
    assert splits.gt_neighbors_eval.max() < subset_size
    assert splits.gt_neighbors_eval.min() >= 0
    assert splits.gt_neighbors_tune.max() < subset_size

    # Verify cache file was created
    cache_file = RESULTS_CACHE_DIR / f"sift_gt_{subset_size}.npz"
    assert cache_file.exists()

    # Exact search on this subset must achieve 1.0000 recall
    exact_idx = ExactFlatIndex(splits.dim)
    exact_idx.build(splits.train)

    # Search on a slice of evaluation queries
    query_sample = splits.test_eval[:100]
    gt_dists_sample = splits.gt_distances_eval[:100]

    dists, retrieved_ids = exact_idx.search(query_sample, k=10)
    recalls = recall_at_k(dists, gt_dists_sample, retrieved_ids, metric="euclidean", k=10)
    assert np.mean(recalls) == 1.0
