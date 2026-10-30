"""Abstract base class for all nearest neighbour indexes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseIndex(ABC):
    """Abstract interface for exact and approximate nearest neighbour indexes."""

    def __init__(self, dim: int, metric: str = "euclidean", name: str = "BaseIndex"):
        self.dim = dim
        self.metric = metric
        self.name = name
        self.build_time: float = 0.0
        self.is_trained: bool = False
        self.num_indexed: int = 0

    @abstractmethod
    def build(self, train_vectors: np.ndarray, **kwargs: Any) -> None:
        """
        Builds / trains the index on the provided vectors.

        Args:
            train_vectors: 2D numpy array of shape (N, dim), dtype float32.
        """
        pass

    @abstractmethod
    def search(
        self, queries: np.ndarray, k: int = 10, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries the index for the top-k nearest neighbours.

        Args:
            queries: 2D numpy array of shape (N_q, dim) or 1D array of shape (dim,).
            k: Number of nearest neighbours to retrieve.
            **kwargs: Query-time parameters (e.g. efSearch, nprobe).

        Returns:
            Tuple of (distances, indices) as numpy arrays of shape (N_q, k).
        """
        pass

    @abstractmethod
    def get_size_bytes(self) -> int:
        """Returns the memory or serialized size of the index in bytes."""
        pass

    def get_bytes_per_vector(self) -> float:
        """Returns the average bytes per indexed vector."""
        if self.num_indexed == 0:
            return 0.0
        return self.get_size_bytes() / float(self.num_indexed)

    def get_info(self) -> Dict[str, Any]:
        """Returns index metadata and size statistics."""
        return {
            "name": self.name,
            "dim": self.dim,
            "metric": self.metric,
            "num_indexed": self.num_indexed,
            "build_time_s": self.build_time,
            "size_bytes": self.get_size_bytes(),
            "bytes_per_vector": self.get_bytes_per_vector(),
        }
