"""High-precision timing utilities for nearest neighbour search benchmarking."""

from dataclasses import dataclass, field
import time
from typing import Any, List, Optional, Tuple
import faiss
import numpy as np

from src.index.base import BaseIndex


@dataclass
class TimingResult:
    """Stores timing metrics for a benchmark run."""
    query_mode: str  # 'single' or 'batch'
    threads: int
    num_queries: int
    total_time_s: float
    qps: float
    latency_ms_per_query: float
    run_times: List[float] = field(default_factory=list)


def time_batch_search(
    index: BaseIndex,
    queries: np.ndarray,
    k: int = 10,
    num_runs: int = 3,
    threads: int = 1,
    **kwargs: Any,
) -> Tuple[TimingResult, np.ndarray, np.ndarray]:
    """
    Measures search time when querying all vectors in a single batch.
    Repeats num_runs times and selects the median time.
    """
    faiss.omp_set_num_threads(threads)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    n_queries = queries.shape[0]

    run_times: List[float] = []
    last_dists: Optional[np.ndarray] = None
    last_indices: Optional[np.ndarray] = None

    for _ in range(num_runs):
        t0 = time.perf_counter()
        last_dists, last_indices = index.search(queries, k=k, **kwargs)
        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)

    median_time = float(np.median(run_times))
    qps = n_queries / median_time if median_time > 0 else 0.0
    latency_ms = (median_time / n_queries) * 1000.0

    result = TimingResult(
        query_mode="batch",
        threads=threads,
        num_queries=n_queries,
        total_time_s=median_time,
        qps=qps,
        latency_ms_per_query=latency_ms,
        run_times=run_times,
    )
    return result, last_dists, last_indices


def time_single_query_search(
    index: BaseIndex,
    queries: np.ndarray,
    k: int = 10,
    num_runs: int = 3,
    threads: int = 1,
    max_eval_queries: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[TimingResult, np.ndarray, np.ndarray]:
    """
    Measures search time when executing queries one by one.
    Repeats num_runs times and selects the median time.
    """
    faiss.omp_set_num_threads(threads)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    total_queries = queries.shape[0]

    eval_count = total_queries
    if max_eval_queries is not None and max_eval_queries < total_queries:
        eval_count = max_eval_queries
        eval_queries = queries[:eval_count]
    else:
        eval_queries = queries

    run_times: List[float] = []
    all_indices: Optional[np.ndarray] = None
    all_dists: Optional[np.ndarray] = None

    for _ in range(num_runs):
        cur_indices = np.empty((eval_count, k), dtype=np.int64)
        cur_dists = np.empty((eval_count, k), dtype=np.float32)

        t0 = time.perf_counter()
        for i in range(eval_count):
            d, idx = index.search(eval_queries[i : i + 1], k=k, **kwargs)
            cur_dists[i] = d[0]
            cur_indices[i] = idx[0]
        elapsed = time.perf_counter() - t0

        # Scale elapsed time if a subset was sampled
        if eval_count < total_queries:
            elapsed = elapsed * (total_queries / float(eval_count))

        run_times.append(elapsed)
        all_indices = cur_indices
        all_dists = cur_dists

    median_time = float(np.median(run_times))
    qps = total_queries / median_time if median_time > 0 else 0.0
    latency_ms = (median_time / total_queries) * 1000.0

    result = TimingResult(
        query_mode="single",
        threads=threads,
        num_queries=total_queries,
        total_time_s=median_time,
        qps=qps,
        latency_ms_per_query=latency_ms,
        run_times=run_times,
    )
    return result, all_dists, all_indices
