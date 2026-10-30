"""Unit tests for Part 4 synthetic clustered data generator and geometry properties."""

import numpy as np
import pytest

from src.data_generator import generate_clustered_dataset
from src.index.exact import ExactFlatIndex
from src.index.hnsw import HNSWIndex
from src.index.ivf_flat import IVFFlatIndex
from src.metrics import recall_at_k


def test_generator_determinism():
    """Verify that calling the generator with the same seed produces bit-exact identical arrays."""
    d1 = generate_clustered_dataset(
        n_vectors=500, n_queries=50, dim=16, n_clusters=4, separations=[1.0], seed=99, force_recompute=True
    )
    d2 = generate_clustered_dataset(
        n_vectors=500, n_queries=50, dim=16, n_clusters=4, separations=[1.0], seed=99, force_recompute=True
    )

    np.testing.assert_array_equal(d1[1.0].train, d2[1.0].train)
    np.testing.assert_array_equal(d1[1.0].test_eval, d2[1.0].test_eval)
    np.testing.assert_array_equal(d1[1.0].gt_neighbors_eval, d2[1.0].gt_neighbors_eval)
    np.testing.assert_allclose(d1[1.0].gt_distances_eval, d2[1.0].gt_distances_eval, rtol=1e-5)


def test_noise_and_assignment_reuse():
    """
    Verify that noise vectors and cluster assignments are strictly reused across settings,
    so that: (X_{s=2.0} - 2.0 * C_z) == (X_{s=1.0} - 1.0 * C_z) == eps.
    """
    datasets = generate_clustered_dataset(
        n_vectors=1000, n_queries=100, dim=16, n_clusters=5, separations=[0.5, 1.0, 2.0], seed=42, force_recompute=True
    )

    X_05 = datasets[0.5].train
    X_10 = datasets[1.0].train
    X_20 = datasets[2.0].train

    # Since X^{(s)} = s * C_z + eps, we have:
    # X^{(2.0)} - X^{(1.0)} = (2.0 - 1.0) * C_z = C_z
    # X^{(1.0)} - X^{(0.5)} = (1.0 - 0.5) * C_z = 0.5 * C_z
    # Therefore: (X^{(2.0)} - X^{(1.0)}) should equal 2 * (X^{(1.0)} - X^{(0.5)})
    diff_21 = X_20 - X_10
    diff_10 = X_10 - X_05
    np.testing.assert_allclose(diff_21, 2.0 * diff_10, rtol=1e-5, atol=1e-5)


def test_exact_ground_truth_accuracy():
    """Verify that ExactFlatIndex achieves 100% recall on synthetic data."""
    datasets = generate_clustered_dataset(
        n_vectors=1000, n_queries=100, dim=16, n_clusters=4, separations=[1.0], seed=42, force_recompute=True
    )
    data = datasets[1.0]

    exact_idx = ExactFlatIndex(dim=data.dim)
    exact_idx.build(data.train)

    dists, ids = exact_idx.search(data.test_eval, k=10)
    recalls = recall_at_k(dists, data.gt_distances_eval, ids, metric="euclidean", k=10)
    assert np.mean(recalls) == 1.0


def test_synthetic_ann_search():
    """Integration check: test HNSW and IVF-Flat search on synthetic clustered data."""
    datasets = generate_clustered_dataset(
        n_vectors=2000, n_queries=100, dim=16, n_clusters=4, separations=[1.0], seed=42, force_recompute=True
    )
    data = datasets[1.0]

    # Test HNSW
    hnsw = HNSWIndex(dim=data.dim, M=8, efConstruction=32)
    hnsw.build(data.train)
    dists, ids = hnsw.search(data.test_eval, k=10, efSearch=64)
    rec_hnsw = np.mean(recall_at_k(dists, data.gt_distances_eval, ids, metric="euclidean", k=10))
    assert rec_hnsw > 0.85

    # Test IVF-Flat
    ivf = IVFFlatIndex(dim=data.dim, nlist=32)
    ivf.build(data.train)
    dists, ids = ivf.search(data.test_eval, k=10, nprobe=16)
    rec_ivf = np.mean(recall_at_k(dists, data.gt_distances_eval, ids, metric="euclidean", k=10))
    assert rec_ivf > 0.85


def test_input_validation_errors():
    """Verify that generate_clustered_dataset raises ValueError on invalid inputs."""
    # Negative / zero vectors
    with pytest.raises(ValueError, match="n_vectors must be positive"):
        generate_clustered_dataset(n_vectors=0)

    # Clusters exceed vectors
    with pytest.raises(ValueError, match="cannot exceed"):
        generate_clustered_dataset(n_vectors=50, n_clusters=100)

    # Invalid dimension
    with pytest.raises(ValueError, match="dim must be positive"):
        generate_clustered_dataset(dim=0)

    # Invalid tuning split
    with pytest.raises(ValueError, match="tuning_split_end must be strictly between"):
        generate_clustered_dataset(n_queries=100, tuning_split_end=150)

    # Invalid separation
    with pytest.raises(ValueError, match="Separation multiplier must be strictly positive"):
        generate_clustered_dataset(separations=[-1.0])


def test_cache_key_includes_dim_and_clusters(tmp_path):
    """Verify that cache filenames include dim and n_clusters to prevent key collisions."""
    ds = generate_clustered_dataset(
        n_vectors=200, n_queries=20, dim=8, n_clusters=3, separations=[1.0], seed=77, cache_dir=tmp_path
    )
    expected_cache_file = tmp_path / "synthetic_sep_1.0_N200_Q20_D8_K3_seed77.npz"
    assert expected_cache_file.exists()


def test_cache_validity_and_metadata_validation(tmp_path):
    """Verify that corrupt or parameter-mismatched caches are invalidated and recomputed."""
    # 1. First generate valid cache
    generate_clustered_dataset(
        n_vectors=300, n_queries=30, dim=8, n_clusters=3, separations=[1.0], seed=42, cache_dir=tmp_path
    )
    cache_file = tmp_path / "synthetic_sep_1.0_N300_Q30_D8_K3_seed42.npz"
    assert cache_file.exists()

    # 2. Intentionally corrupt cache with invalid array shape
    np.savez_compressed(
        cache_file,
        train=np.zeros((10, 8), dtype=np.float32),  # Wrong N=10 instead of 300
        queries=np.zeros((30, 8), dtype=np.float32),
        gt_neighbors=np.zeros((30, 10), dtype=np.int32),
        gt_distances=np.zeros((30, 10), dtype=np.float32),
    )

    # 3. Reloading should detect shape mismatch, recompute, and produce valid dataset
    ds = generate_clustered_dataset(
        n_vectors=300, n_queries=30, dim=8, n_clusters=3, separations=[1.0], seed=42, cache_dir=tmp_path
    )
    assert ds[1.0].train.shape == (300, 8)


def test_geometry_diagnostics_and_ground_truth_verification():
    """Verify geometry diagnostics, relative separation scaling, and GT verification."""
    datasets, diags = generate_clustered_dataset(
        n_vectors=1000,
        n_queries=100,
        dim=16,
        n_clusters=5,
        separations=[0.5, 1.0, 2.0],
        seed=42,
        force_recompute=True,
        return_diagnostics=True,
    )

    d_05 = diags[0.5]
    d_10 = diags[1.0]
    d_20 = diags[2.0]

    # Inter-cluster distance must scale strictly with s
    assert np.isclose(d_10.mean_inter_cluster_dist, 2.0 * d_05.mean_inter_cluster_dist, rtol=1e-3)
    assert np.isclose(d_20.mean_inter_cluster_dist, 2.0 * d_10.mean_inter_cluster_dist, rtol=1e-3)

    # Within-cluster spread must be invariant to separation multiplier
    assert np.isclose(d_05.mean_within_cluster_spread, d_10.mean_within_cluster_spread, rtol=1e-3)
    assert np.isclose(d_10.mean_within_cluster_spread, d_20.mean_within_cluster_spread, rtol=1e-3)

    # Relative separation ratio must double with s
    assert np.isclose(d_10.relative_separation_ratio, 2.0 * d_05.relative_separation_ratio, rtol=1e-2)
    assert np.isclose(d_20.relative_separation_ratio, 2.0 * d_10.relative_separation_ratio, rtol=1e-2)

    # Cluster sizes must sum to N
    assert d_10.cluster_size_mean == 200.0
    assert d_10.cluster_size_min > 0

    # Ground-truth sanity check passed with low discrepancy
    assert d_05.gt_sanity_check_passed is True
    assert d_10.gt_sanity_check_passed is True
    assert d_20.gt_sanity_check_passed is True
    assert d_10.gt_max_discrepancy < 1e-4
