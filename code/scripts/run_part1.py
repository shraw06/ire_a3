"""CLI runner to execute Part 1: Exact Baseline benchmarks and ground-truth agreement."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.benchmark.reporter import log_result, print_summary_table
from src.benchmark.timing import time_batch_search, time_single_query_search
from src.config import DEFAULT_EPSILON, DEFAULT_K
from src.data import load_dataset
from src.index.exact import ExactFlatIndex
from src.metrics import recall_at_k


def run_part1_for_dataset(
    dataset_name: str,
    max_train: int = None,
    threads_list: List[int] = [1],
    num_runs: int = 3,
    verify_only: bool = False,
    single_query_sample: int = 500,
) -> List[Dict[str, Any]]:
    """
    Executes Part 1 exact baseline for a given dataset.
    """
    print(f"\n{'='*70}")
    print(f"Running Part 1 Exact Baseline for dataset: {dataset_name.upper()}")
    print(f"{'='*70}")

    # 1. Load data
    print(f"Loading {dataset_name} data...")
    splits = load_dataset(dataset_name, max_train=max_train, normalize_glove=True)
    n_train = splits.train.shape[0]
    n_eval = splits.test_eval.shape[0]
    dim = splits.dim
    print(f"Loaded train: {n_train} vectors ({dim}-d), eval queries: {n_eval}")

    # 2. Build Exact Index
    print("Building FAISS Exact Flat Index (IndexFlatL2)...")
    index = ExactFlatIndex(dim=dim)
    index.build(splits.train)
    index_size_mb = index.get_size_bytes() / (1024 * 1024)
    print(f"Index built in {index.build_time:.2f}s | Size: {index_size_mb:.1f} MB ({index.get_bytes_per_vector():.1f} bytes/vec)")

    # 3. Ground Truth Verification on Evaluation Split (queries 5,000–10,000)
    print("\n[Step 1/3] Verifying exact search agreement against dataset ground truth...")
    distances, indices = index.search(splits.test_eval, k=DEFAULT_K)

    # Analyze ties
    d10 = splits.gt_distances_eval[:, DEFAULT_K - 1]
    tie_mask = (splits.gt_distances_eval <= (d10[:, None] + DEFAULT_EPSILON)).sum(axis=1) > DEFAULT_K
    num_tie_queries = int(np.sum(tie_mask))

    # Calculate distance-threshold recall using FAISS distances aligned to dataset metric
    metric = "euclidean" if dataset_name == "sift" else "angular"
    mean_recall, per_query_recalls = recall_at_k(
        retrieved_distances=distances,
        gt_distances=splits.gt_distances_eval,
        retrieved_indices=indices,
        metric=metric,
        k=DEFAULT_K,
        eps=DEFAULT_EPSILON,
        return_per_query=True,
    )

    min_recall = float(np.min(per_query_recalls))
    print(f"  Total evaluation queries : {n_eval}")
    print(f"  Queries with boundary tie: {num_tie_queries} ({num_tie_queries / n_eval * 100:.1f}%)")
    print(f"  Mean Recall@{DEFAULT_K}          : {mean_recall:.6f}")
    print(f"  Min Recall@{DEFAULT_K}           : {min_recall:.6f}")

    if mean_recall == 1.0:
        print("  --> SUCCESS: Exact search results perfectly agree with dataset ground truth!")
    else:
        print("  --> WARNING: Mean recall is below 1.0. Check distance metric or tie tolerances.")

    if verify_only:
        return []

    # 4. Timing Benchmarks: Single-query vs. Batch-query
    print("\n[Step 2/3] Benchmarking exact search timing (Single-Query vs. Batch-Query)...")
    results: List[Dict[str, Any]] = []

    for threads in threads_list:
        print(f"\n--- Testing with threads = {threads} ---")

        # A. Batch query search
        print(f"  [A] Batch search ({n_eval} queries, median of {num_runs} runs)...")
        t_batch, _, _ = time_batch_search(
            index=index,
            queries=splits.test_eval,
            k=DEFAULT_K,
            num_runs=num_runs,
            threads=threads,
        )
        print(f"      Batch:  {t_batch.qps:.2f} QPS | Latency: {t_batch.latency_ms_per_query:.2f} ms/query | Total: {t_batch.total_time_s:.2f}s")

        batch_row = {
            "dataset": dataset_name,
            "index_type": index.name,
            "threads": threads,
            "query_mode": "batch",
            "build_params": "{}",
            "search_params": "{}",
            "recall_at_10": mean_recall,
            "qps": round(t_batch.qps, 2),
            "latency_ms": round(t_batch.latency_ms_per_query, 2),
            "build_time_s": round(index.build_time, 2),
            "index_size_bytes": index.get_size_bytes(),
            "bytes_per_vector": round(index.get_bytes_per_vector(), 2),
        }
        log_result(batch_row)
        results.append(batch_row)

        # B. Single query search
        sample_msg = f"(sampled {single_query_sample}/{n_eval} queries)" if single_query_sample and single_query_sample < n_eval else f"({n_eval} queries)"
        print(f"  [B] Single query search {sample_msg}, median of {num_runs} runs...")
        t_single, _, _ = time_single_query_search(
            index=index,
            queries=splits.test_eval,
            k=DEFAULT_K,
            num_runs=num_runs,
            threads=threads,
            max_eval_queries=single_query_sample,
        )
        print(f"      Single: {t_single.qps:.2f} QPS | Latency: {t_single.latency_ms_per_query:.2f} ms/query | Extrapolated Total: {t_single.total_time_s:.2f}s")

        single_row = {
            "dataset": dataset_name,
            "index_type": index.name,
            "threads": threads,
            "query_mode": "single",
            "build_params": "{}",
            "search_params": "{}",
            "recall_at_10": mean_recall,
            "qps": round(t_single.qps, 2),
            "latency_ms": round(t_single.latency_ms_per_query, 2),
            "build_time_s": round(index.build_time, 2),
            "index_size_bytes": index.get_size_bytes(),
            "bytes_per_vector": round(index.get_bytes_per_vector(), 2),
        }
        log_result(single_row)
        results.append(single_row)

        speedup = t_batch.qps / t_single.qps if t_single.qps > 0 else 0.0
        print(f"      >>> Batch vs. Single Speedup: {speedup:.2f}x")

    return results


def print_architectural_explanation():
    """Prints the theoretical explanation for single-query vs batch search difference."""
    explanation = """
====================================================================================================
ARCHITECTURAL EXPLANATION: WHY BATCH SEARCH AND SINGLE-QUERY SEARCH DIFFER SO MUCH
====================================================================================================
1. Arithmetic Intensity & BLAS Level (GEMM vs. GEMV):
   - Single-query search: Querying one vector against N database vectors (1 x d vs N x d) is a
     BLAS Level 2 matrix-vector multiplication (GEMV). For every query, the entire dataset
     (~500 MB) must be streamed across the system memory bus from RAM into CPU caches. The arithmetic
     intensity (FLOPs per byte transferred) is minimal (O(1)).
   - Batch search: Querying B vectors simultaneously (B x d vs N x d) is a BLAS Level 3 matrix-matrix
     multiplication (GEMM). The database vectors reside in CPU cache and are reused across all B
     queries in cache tiles, increasing arithmetic intensity by a factor of B and saturating SIMD
     floating-point execution units (AVX2/FMA) rather than waiting on memory bandwidth.

2. Memory Bandwidth Saturation:
   - Executing 5,000 queries individually forces the memory controller to transfer:
     5,000 queries * 512 MB/query = ~2.56 Terabytes of data across the memory bus!
     At a typical DDR4/DDR5 bandwidth of 40-50 GB/s, memory transfer alone requires ~50-60 seconds.
   - Batching computes matrix blocks in-cache, eliminating redundant RAM streaming transfers.

3. Call & Thread Dispatch Amortization:
   - Single-query search repeats Python-to-C++ wrapper invocations, index dispatch, and result heap
     allocations 5,000 times. Batch search incurs this overhead only once for the entire batch.
====================================================================================================
"""
    print(explanation)


def main():
    parser = argparse.ArgumentParser(description="Part 1: Exact Baseline Benchmarking")
    parser.add_argument(
        "--dataset",
        choices=["sift", "glove", "all"],
        default="all",
        help="Dataset to benchmark (default: all)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1],
        help="Thread counts to benchmark (e.g. 1 or 1 16)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with 100K train vectors and 200 eval queries",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify ground-truth agreement, skip timing",
    )
    parser.add_argument(
        "--single-query-sample",
        type=int,
        default=500,
        help="Number of queries to sample for single-query exact search (default: 500)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs to compute median timing (default: 3)",
    )

    args = parser.parse_args()

    datasets = ["sift", "glove"] if args.dataset == "all" else [args.dataset]
    max_train = 100_000 if args.quick else None
    single_query_sample = 200 if args.quick else args.single_query_sample

    all_results = []
    for d in datasets:
        res = run_part1_for_dataset(
            dataset_name=d,
            max_train=max_train,
            threads_list=args.threads,
            num_runs=args.num_runs,
            verify_only=args.verify_only,
            single_query_sample=single_query_sample,
        )
        all_results.extend(res)

    if all_results:
        print("\n" + "="*70)
        print("SUMMARY OF BENCHMARK RUNS (saved to results/results.csv)")
        print("="*70)
        print_summary_table(all_results)
        print_architectural_explanation()


if __name__ == "__main__":
    main()
