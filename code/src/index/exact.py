"""Exact nearest neighbour search baseline using FAISS IndexFlatL2."""

import time
from typing import Any, Optional, Tuple
import faiss
import numpy as np

from src.index.base import BaseIndex


class ExactFlatIndex(BaseIndex):
    """
    Exact brute-force nearest neighbour search using FAISS IndexFlatL2.
    Serves as the reference ground-truth baseline.
    """

    def __init__(self, dim: int, metric: str = "euclidean"):
        super().__init__(dim=dim, metric=metric, name="Exact-Flat")
        self.index: Optional[faiss.IndexFlatL2] = None

    def build(self, train_vectors: np.ndarray, **kwargs: Any) -> None:
        """
        Adds vectors to the FAISS Flat index and records build time.
        """
        train_vectors = np.ascontiguousarray(train_vectors, dtype=np.float32)
        if train_vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dim}, got {train_vectors.shape[1]}"
            )

        t0 = time.perf_counter()
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(train_vectors)
        self.build_time = time.perf_counter() - t0
        self.num_indexed = self.index.ntotal
        self.is_trained = True

    def search(
        self, queries: np.ndarray, k: int = 10, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches top-k nearest neighbours.
        
        Args:
            queries: 2D array of shape (N, dim) or 1D array of shape (dim,).
            k: Top-k neighbours to retrieve.
        
        Returns:
            Tuple of (distances, indices).
        """
        if self.index is None:
            raise RuntimeError("Index is not built yet. Call build() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[np.newaxis, :]

        distances, indices = self.index.search(queries, k)

        if is_single:
            return distances[0], indices[0]
        return distances, indices

    def get_size_bytes(self) -> int:
        """Computes index size in bytes using FAISS serialization."""
        if self.index is None:
            return 0
        serialized = faiss.serialize_index(self.index)
        return len(serialized)
