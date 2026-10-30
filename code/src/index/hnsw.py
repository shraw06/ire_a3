"""Hierarchical Navigable Small World (HNSW) index implementation using FAISS."""

import time
from typing import Any, Optional, Tuple
import faiss
import numpy as np

from src.index.base import BaseIndex


class HNSWIndex(BaseIndex):
    """
    HNSW graph-based approximate nearest neighbour index using FAISS IndexHNSWFlat.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        efConstruction: int = 64,
        metric: str = "euclidean",
    ):
        super().__init__(dim=dim, metric=metric, name="HNSW")
        self.M = M
        self.efConstruction = efConstruction
        self.index: Optional[faiss.IndexHNSWFlat] = None

    def build(
        self,
        train_vectors: np.ndarray,
        num_threads: int = 16,
        **kwargs: Any,
    ) -> None:
        """
        Builds the HNSW graph index on the provided database vectors.
        """
        train_vectors = np.ascontiguousarray(train_vectors, dtype=np.float32)
        if train_vectors.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {train_vectors.shape[1]}")

        faiss.omp_set_num_threads(num_threads)
        t0 = time.perf_counter()

        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.add(train_vectors)

        self.build_time = time.perf_counter() - t0
        self.num_indexed = self.index.ntotal
        self.is_trained = True

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        efSearch: int = 32,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches top-k nearest neighbours with dynamic efSearch.
        """
        if self.index is None:
            raise RuntimeError("Index is not built yet. Call build() first.")

        self.index.hnsw.efSearch = efSearch

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        is_single = queries.ndim == 1
        if is_single:
            queries = queries[np.newaxis, :]

        distances, indices = self.index.search(queries, k)

        if is_single:
            return distances[0], indices[0]
        return distances, indices

    def get_size_bytes(self) -> int:
        if self.index is None:
            return 0
        return len(faiss.serialize_index(self.index))
