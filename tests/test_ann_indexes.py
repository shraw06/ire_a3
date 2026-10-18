"""Unit tests for HNSW, IVF-Flat, IVF-PQ indexes and Pareto frontier calculation."""

import numpy as np
import pytest

from src.benchmark.plotting import compute_pareto_frontier, interpolate_qps_at_recall
from src.index.hnsw import HNSWIndex
from src.index.ivf_flat import IVFFlatIndex
from src.index.ivf_pq import IVFPQIndex


def test_hnsw_synthetic():
    dim = 16
    n_train = 200
    n_queries = 10

    rng = np.random.default_rng(42)
    train = rng.standard_normal((n_train, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    index = HNSWIndex(dim=dim, M=8, efConstruction=32)
    index.build(train, num_threads=1)

    assert index.is_trained
    assert index.num_indexed == n_train
    assert index.get_size_bytes() > 0

    d, idx = index.search(queries, k=5, efSearch=16)
    assert d.shape == (n_queries, 5)
    assert idx.shape == (n_queries, 5)


def test_ivf_flat_synthetic():
    dim = 16
    n_train = 500
    n_queries = 10

    rng = np.random.default_rng(42)
    train = rng.standard_normal((n_train, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    index = IVFFlatIndex(dim=dim, nlist=16)
    index.build(train, num_threads=1)

    assert index.is_trained
    assert index.num_indexed == n_train
    assert index.get_size_bytes() > 0

    d, idx = index.search(queries, k=5, nprobe=4)
    assert d.shape == (n_queries, 5)
    assert idx.shape == (n_queries, 5)


def test_ivf_pq_synthetic():
    dim = 16
    n_train = 500
    n_queries = 10

    rng = np.random.default_rng(42)
    train = rng.standard_normal((n_train, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    index_flat = IVFFlatIndex(dim=dim, nlist=16)
    index_flat.build(train, num_threads=1)

    index_pq = IVFPQIndex(dim=dim, nlist=16, M_pq=4, nbits=8)
    index_pq.build(train, num_threads=1)

    assert index_pq.is_trained
    assert index_pq.num_indexed == n_train
    # IVF-PQ size must be significantly smaller than IVF-Flat
    assert index_pq.get_size_bytes() < index_flat.get_size_bytes()

    d, idx = index_pq.search(queries, k=5, nprobe=4)
    assert d.shape == (n_queries, 5)
    assert idx.shape == (n_queries, 5)


def test_pareto_frontier_logic():
    # Points with (recall, qps)
    pts = [
        {"recall_at_10": 0.50, "qps": 1000.0, "name": "p1"},
        {"recall_at_10": 0.60, "qps": 800.0, "name": "p2"},
        {"recall_at_10": 0.55, "qps": 700.0, "name": "p3 (dominated by p2)"},
        {"recall_at_10": 0.90, "qps": 500.0, "name": "p4"},
        {"recall_at_10": 0.95, "qps": 300.0, "name": "p5"},
        {"recall_at_10": 0.96, "qps": 100.0, "name": "p6"},
        {"recall_at_10": 0.94, "qps": 200.0, "name": "p7 (dominated by p5)"},
    ]

    pareto = compute_pareto_frontier(pts)
    pareto_names = [p["name"] for p in pareto]
    assert "p1" in pareto_names
    assert "p2" in pareto_names
    assert "p4" in pareto_names
    assert "p5" in pareto_names
    assert "p6" in pareto_names
    assert "p3 (dominated by p2)" not in pareto_names
    assert "p7 (dominated by p5)" not in pareto_names

    # Test interpolation at 0.90 (exact match)
    q_090 = interpolate_qps_at_recall(pareto, 0.90)
    assert q_090 == pytest.approx(500.0)

    # Test interpolation at 0.95 (exact match)
    q_095 = interpolate_qps_at_recall(pareto, 0.95)
    assert q_095 == pytest.approx(300.0)

    # Test interpolation at 0.99 (exceeds max recall 0.96)
    q_099 = interpolate_qps_at_recall(pareto, 0.99)
    assert q_099 is None
