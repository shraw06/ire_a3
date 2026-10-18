"""Data loading, normalization, and query split utilities for SIFT and GloVe."""

from dataclasses import dataclass
from typing import Optional, Tuple
import h5py
import numpy as np
import faiss

from src.config import DATASET_METADATA, TUNING_SPLIT_END


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


def load_dataset(
    name: str,
    max_train: Optional[int] = None,
    normalize_glove: bool = True,
    tuning_split_end: int = TUNING_SPLIT_END,
) -> DatasetSplits:
    """
    Loads dataset vectors and pre-computed nearest neighbours from HDF5 file.
    
    Args:
        name: 'sift' or 'glove'
        max_train: Optional slice limit for train vectors (e.g. for quick mode)
        normalize_glove: If True and dataset is glove, normalizes train and test vectors
        tuning_split_end: Index at which to split tuning and evaluation queries (default: 5000)
    
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
        neighbors = np.array(f["neighbors"], dtype=np.int32)
        distances = np.array(f["distances"], dtype=np.float32)

    # Normalization for GloVe (angular distance on unit sphere corresponds to Euclidean distance)
    if name == "glove" and normalize_glove:
        normalize_vectors(train)
        normalize_vectors(test)

    train = np.ascontiguousarray(train, dtype=np.float32)
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
