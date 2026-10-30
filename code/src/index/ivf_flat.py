"""Inverted File with Flat vectors (IVF-Flat) index implementation using FAISS."""

import time
from typing import Any, Optional, Tuple
import faiss
import numpy as np

from src.index.base import BaseIndex


class IVFFlatIndex(BaseIndex):
    """
    Inverted File with Flat vectors index using FAISS IndexIVFFlat.
    Partitions the space into nlist Voronoi cells and probes nprobe cells at query time.
    """

    def __init__(
        self,
        dim: int,
        nlist: int = 1024,
        metric: str = "euclidean",
    ):
        super().__init__(dim=dim, metric=metric, name="IVF-Flat")
        self.nlist = nlist
        self.quantizer: Optional[faiss.IndexFlatL2] = None
        self.index: Optional[faiss.IndexIVFFlat] = None

    def build(
        self,
        train_vectors: np.ndarray,
        num_threads: int = 16,
        **kwargs: Any,
    ) -> None:
        """
        Trains coarse centroids and adds database vectors to inverted lists.
        """
        train_vectors = np.ascontiguousarray(train_vectors, dtype=np.float32)
        if train_vectors.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {train_vectors.shape[1]}")

        faiss.omp_set_num_threads(num_threads)
        t0 = time.perf_counter()

        self.quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        self.index.train(train_vectors)
        self.index.add(train_vectors)

        self.build_time = time.perf_counter() - t0
        self.num_indexed = self.index.ntotal
        self.is_trained = True

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int = 1,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches top-k nearest neighbours with dynamic nprobe.
        """
        if self.index is None:
            raise RuntimeError("Index is not built yet. Call build() first.")

        self.index.nprobe = min(nprobe, self.nlist)

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
