"""Unit tests for distance-threshold recall@k using FAISS and ground-truth distances."""

import pytest
import numpy as np

from src.metrics import align_faiss_distances, query_recall_at_k, recall_at_k


def test_align_faiss_distances():
    """Verify that align_faiss_distances correctly transforms squared L2 to native metrics."""
    # Euclidean: sqrt(L2^2)
    faiss_euc = np.array([0.0, 4.0, 9.0, 16.0], dtype=np.float32)
    aligned_euc = align_faiss_distances(faiss_euc, metric="euclidean")
    np.testing.assert_allclose(aligned_euc, [0.0, 2.0, 3.0, 4.0])

    # Angular: on normalized vectors, L2^2 = 2 * (1 - cos) -> ang = L2^2 / 2
    faiss_ang = np.array([0.0, 0.4, 0.8, 1.0], dtype=np.float32)
    aligned_ang = align_faiss_distances(faiss_ang, metric="angular")
    np.testing.assert_allclose(aligned_ang, [0.0, 0.2, 0.4, 0.5])


def test_perfect_recall():
    gt_dists = np.arange(1.0, 21.0, dtype=np.float32)
    retrieved_dists = np.arange(1.0, 11.0, dtype=np.float32)
    retrieved_indices = np.arange(10, dtype=np.int32)

    score = query_recall_at_k(retrieved_dists, gt_dists, retrieved_indices=retrieved_indices, k=10)
    assert score == pytest.approx(1.0)


def test_zero_recall():
    gt_dists = np.arange(1.0, 21.0, dtype=np.float32)
    # Retrieved candidates have distances far beyond d_10 = 10.0
    retrieved_dists = np.arange(50.0, 60.0, dtype=np.float32)
    retrieved_indices = np.arange(10, dtype=np.int32)

    score = query_recall_at_k(retrieved_dists, gt_dists, retrieved_indices=retrieved_indices, k=10)
    assert score == pytest.approx(0.0)


def test_partial_recall():
    gt_dists = np.arange(1.0, 21.0, dtype=np.float32)
    # 7 candidates <= d_10 (10.0), 3 candidates > d_10
    retrieved_dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 15.0, 16.0, 17.0], dtype=np.float32)
    retrieved_indices = np.arange(10, dtype=np.int32)

    score = query_recall_at_k(retrieved_dists, gt_dists, retrieved_indices=retrieved_indices, k=10)
    assert score == pytest.approx(0.7)


def test_tie_at_10th_boundary():
    """
    10th true neighbour has distance 10.0.
    Retrieved candidate list has distance 10.0 for its 10th item (tied with 10th neighbour).
    Must yield 1.0 recall.
    """
    gt_dists = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 15.0], dtype=np.float32
    )
    # All 10 retrieved candidates are <= 10.0
    retrieved_dists = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32
    )
    score = query_recall_at_k(retrieved_dists, gt_dists, k=10, eps=1e-4)
    assert score == pytest.approx(1.0)


def test_alternative_tied_database_vector_credited():
    """
    Alternative database vectors tied at the k-th boundary (d <= d_k + eps)
    are recognized as valid nearest neighbours purely by distance.
    """
    gt_dists = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # Alternative database vector with distance 1.0
    retrieved_dists = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    retrieved_indices = np.array([0, 1, 999], dtype=np.int32)  # ID 999 not in ground truth

    score = query_recall_at_k(
        retrieved_distances=retrieved_dists,
        gt_distances=gt_dists,
        retrieved_indices=retrieved_indices,
        k=3,
        eps=1e-4,
    )
    assert score == pytest.approx(1.0)


def test_worse_item_rejected():
    """Candidate with distance strictly greater than d_10 + eps is rejected."""
    gt_dists = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.005], dtype=np.float32
    )
    # 10th item is 10.005 (beyond tolerance eps=1e-4)
    retrieved_dists = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.005], dtype=np.float32
    )
    score = query_recall_at_k(retrieved_dists, gt_dists, k=10, eps=1e-4)
    assert score == pytest.approx(0.9)


def test_duplicate_retrieved_protection():
    """Duplicate candidate IDs cannot artificially inflate recall."""
    gt_dists = np.arange(1.0, 21.0, dtype=np.float32)
    retrieved_dists = np.arange(1.0, 11.0, dtype=np.float32)
    # Candidate list has repeated ID 0
    retrieved_indices = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)

    score = query_recall_at_k(
        retrieved_distances=retrieved_dists,
        gt_distances=gt_dists,
        retrieved_indices=retrieved_indices,
        k=10,
    )
    assert score == pytest.approx(0.9)


def test_batch_recall():
    """Test batch recall across multiple queries with FAISS distance metric alignment."""
    gt_dists = np.tile(np.arange(1.0, 21.0, dtype=np.float32), (3, 1))

    # FAISS returns squared L2 distances
    raw_faiss_dists = np.array([
        np.arange(1.0, 11.0, dtype=np.float32) ** 2,                                # 10/10 = 1.0
        np.array([1, 4, 9, 16, 25, 36, 49, 1000, 1000, 1000], dtype=np.float32),    # 7/10 = 0.7
        np.arange(50.0, 60.0, dtype=np.float32) ** 2,                              # 0/10 = 0.0
    ])

    mean_recall, per_query = recall_at_k(
        retrieved_distances=raw_faiss_dists,
        gt_distances=gt_dists,
        metric="euclidean",
        k=10,
        return_per_query=True,
    )
    assert mean_recall == pytest.approx((1.0 + 0.7 + 0.0) / 3.0)
    assert per_query[0] == pytest.approx(1.0)
    assert per_query[1] == pytest.approx(0.7)
    assert per_query[2] == pytest.approx(0.0)


def test_custom_k():
    """Test custom k=5."""
    gt_dists = np.arange(1.0, 11.0, dtype=np.float32)
    retrieved_dists = np.array([1.0, 2.0, 3.0, 4.0, 50.0], dtype=np.float32)
    score = query_recall_at_k(retrieved_dists, gt_dists, k=5)
    assert score == pytest.approx(4.0 / 5.0)
