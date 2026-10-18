"""Integration tests for ExactFlatIndex and ground truth agreement."""

import numpy as np
import pytest

from src.data import load_dataset
from src.index.exact import ExactFlatIndex
from src.metrics import recall_at_k


def test_exact_flat_synthetic():
    """Verify ExactFlatIndex on deterministic synthetic vectors."""
    dim = 8
    n_train = 50
    n_queries = 5

    rng = np.random.default_rng(42)
    train = rng.standard_normal((n_train, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    index = ExactFlatIndex(dim=dim)
    index.build(train)

    assert index.is_trained
    assert index.num_indexed == n_train
    assert index.get_size_bytes() > 0

    # Test single query search
    d_single, idx_single = index.search(queries[0], k=5)
    assert d_single.shape == (5,)
    assert idx_single.shape == (5,)

    # Test batch search
    d_batch, idx_batch = index.search(queries, k=5)
    assert d_batch.shape == (n_queries, 5)
    assert idx_batch.shape == (n_queries, 5)
    assert np.array_equal(idx_single, idx_batch[0])


def test_sift_exact_agreement_sample():
    """Verify that FAISS exact search achieves recall@10 = 1.0 on a sample of SIFT test queries using distances."""
    splits = load_dataset("sift")
    # Take first 50 eval queries
    sample_queries = splits.test_eval[:50]
    sample_gt_distances = splits.gt_distances_eval[:50]

    index = ExactFlatIndex(dim=splits.dim)
    index.build(splits.train)

    distances, indices = index.search(sample_queries, k=10)

    # Compute recall@10 using FAISS distances aligned to Euclidean metric
    mean_recall, per_query = recall_at_k(
        retrieved_distances=distances,
        gt_distances=sample_gt_distances,
        retrieved_indices=indices,
        metric="euclidean",
        k=10,
        return_per_query=True,
    )

    assert mean_recall == pytest.approx(1.0)
    assert np.all(per_query == 1.0)


def test_glove_exact_agreement_sample():
    """Verify that normalized GloVe exact search achieves recall@10 = 1.0 on a sample of GloVe test queries using distances."""
    splits = load_dataset("glove", normalize_glove=True)
    # Take first 50 eval queries
    sample_queries = splits.test_eval[:50]
    sample_gt_distances = splits.gt_distances_eval[:50]

    index = ExactFlatIndex(dim=splits.dim)
    index.build(splits.train)

    distances, indices = index.search(sample_queries, k=10)

    # Compute recall@10 using FAISS distances aligned to Angular metric
    mean_recall, per_query = recall_at_k(
        retrieved_distances=distances,
        gt_distances=sample_gt_distances,
        retrieved_indices=indices,
        metric="angular",
        k=10,
        return_per_query=True,
    )

    assert mean_recall == pytest.approx(1.0)
    assert np.all(per_query == 1.0)
