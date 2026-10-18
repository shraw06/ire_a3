"""CLI runner to execute Part 2: Benchmarking three indexes (HNSW, IVF-Flat, IVF-PQ)."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.plotting import (
    compute_pareto_frontier,
    interpolate_qps_at_recall,
    plot_individual_sweep,
    plot_overall_pareto,
)
from src.benchmark.reporter import log_result
from src.benchmark.timing import time_single_query_search
from src.config import (
    DEFAULT_EPSILON,
    DEFAULT_K,
    FIGURES_DIR,
    GLOVE_PQ_M_LIST,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH_LIST,
    HNSW_M_LIST,
    IVF_FLAT_NLIST_LIST,
    IVF_FLAT_NPROBE_LIST,
    IVF_PQ_NBITS,
    IVF_PQ_NLIST,
    IVF_PQ_NPROBE_LIST,
    QUICK_GLOVE_PQ_M,
    QUICK_HNSW_EF_SEARCH,
    QUICK_HNSW_M_LIST,
    QUICK_IVF_NLIST_LIST,
    QUICK_IVF_NPROBE,
    QUICK_SIFT_PQ_M,
    RESULTS_CSV,
    SIFT_PQ_M_LIST,
    SWEEPS_DIR,
)
from src.data import load_dataset
from src.index.exact import ExactFlatIndex
from src.index.hnsw import HNSWIndex
from src.index.ivf_flat import IVFFlatIndex
from src.index.ivf_pq import IVFPQIndex
from src.metrics import recall_at_k


def get_exact_baseline_qps(dataset_name: str, splits, num_runs: int = 2) -> float:
    """Reads exact baseline single-query QPS from results.csv if present, or measures it."""
    if RESULTS_CSV.exists():
        import csv
        with open(RESULTS_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (
                    row.get("dataset") == dataset_name
                    and row.get("index_type") == "Exact-Flat"
                    and row.get("query_mode") == "single"
                    and row.get("threads") == "1"
                ):
                    try:
                        return float(row["qps"])
                    except (ValueError, KeyError):
                        pass

    # Fallback: measure on a sample
    print(f"  Measuring Exact-Flat baseline single-query QPS for {dataset_name}...")
    exact_idx = ExactFlatIndex(dim=splits.dim)
    exact_idx.build(splits.train)
    t_res, _, _ = time_single_query_search(
        exact_idx,
        splits.test_eval,
        k=DEFAULT_K,
        num_runs=num_runs,
        threads=1,
        max_eval_queries=500,
    )
    return t_res.qps


def benchmark_hnsw_sweeps(
    dataset_name: str,
    splits,
    m_list: List[int],
    ef_search_list: List[int],
    ef_construction: int,
    metric: str,
    num_runs: int = 3,
    max_eval_queries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Runs build and search sweeps for HNSW."""
    print(f"\n[{dataset_name.upper()}] Benchmarking HNSW sweeps across M={m_list}...")
    results: List[Dict[str, Any]] = []

    for M in m_list:
        print(f"  Building HNSW (M={M}, efConstruction={ef_construction})...")
        index = HNSWIndex(dim=splits.dim, M=M, efConstruction=ef_construction, metric=metric)
        index.build(splits.train, num_threads=16)
        size_mb = index.get_size_bytes() / (1024 * 1024)
        print(f"    Built in {index.build_time:.2f}s | Size: {size_mb:.1f} MB ({index.get_bytes_per_vector():.1f} bytes/vec)")

        for ef_search in ef_search_list:
            t_res, cand_dists, cand_idx = time_single_query_search(
                index=index,
                queries=splits.test_eval,
                k=DEFAULT_K,
                num_runs=num_runs,
                threads=1,
                max_eval_queries=max_eval_queries,
                efSearch=ef_search,
            )

            rec = recall_at_k(
                retrieved_distances=cand_dists,
                gt_distances=splits.gt_distances_eval[: len(cand_dists)],
                retrieved_indices=cand_idx,
                metric=metric,
                k=DEFAULT_K,
                eps=DEFAULT_EPSILON,
            )

            row = {
                "dataset": dataset_name,
                "index_type": "HNSW",
                "threads": 1,
                "query_mode": "single",
                "build_params": f"M={M},efConst={ef_construction}",
                "search_params": f"efSearch={ef_search}",
                "recall_at_10": round(rec, 4),
                "qps": round(t_res.qps, 2),
                "latency_ms": round(t_res.latency_ms_per_query, 2),
                "build_time_s": round(index.build_time, 2),
                "index_size_bytes": index.get_size_bytes(),
                "bytes_per_vector": round(index.get_bytes_per_vector(), 2),
            }
            log_result(row)
            results.append(row)
            print(f"    efSearch={ef_search:<4} -> Recall@10={rec:.4f} | QPS={t_res.qps:<8.1f} | Latency={t_res.latency_ms_per_query:.2f} ms")

    plot_individual_sweep(
        dataset_name=dataset_name,
        index_type="HNSW",
        results=results,
        output_path=SWEEPS_DIR / f"{dataset_name}_hnsw_sweep.png",
    )
    return results


def benchmark_ivf_flat_sweeps(
    dataset_name: str,
    splits,
    nlist_list: List[int],
    nprobe_list: List[int],
    metric: str,
    num_runs: int = 3,
    max_eval_queries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Runs build and search sweeps for IVF-Flat."""
    print(f"\n[{dataset_name.upper()}] Benchmarking IVF-Flat sweeps across nlist={nlist_list}...")
    results: List[Dict[str, Any]] = []

    for nlist in nlist_list:
        print(f"  Building IVF-Flat (nlist={nlist})...")
        index = IVFFlatIndex(dim=splits.dim, nlist=nlist, metric=metric)
        index.build(splits.train, num_threads=16)
        size_mb = index.get_size_bytes() / (1024 * 1024)
        print(f"    Built in {index.build_time:.2f}s | Size: {size_mb:.1f} MB ({index.get_bytes_per_vector():.1f} bytes/vec)")

        for nprobe in nprobe_list:
            if nprobe > nlist:
                continue
            t_res, cand_dists, cand_idx = time_single_query_search(
                index=index,
                queries=splits.test_eval,
                k=DEFAULT_K,
                num_runs=num_runs,
                threads=1,
                max_eval_queries=max_eval_queries,
                nprobe=nprobe,
            )

            rec = recall_at_k(
                retrieved_distances=cand_dists,
                gt_distances=splits.gt_distances_eval[: len(cand_dists)],
                retrieved_indices=cand_idx,
                metric=metric,
                k=DEFAULT_K,
                eps=DEFAULT_EPSILON,
            )

            row = {
                "dataset": dataset_name,
                "index_type": "IVF-Flat",
                "threads": 1,
                "query_mode": "single",
                "build_params": f"nlist={nlist}",
                "search_params": f"nprobe={nprobe}",
                "recall_at_10": round(rec, 4),
                "qps": round(t_res.qps, 2),
                "latency_ms": round(t_res.latency_ms_per_query, 2),
                "build_time_s": round(index.build_time, 2),
                "index_size_bytes": index.get_size_bytes(),
                "bytes_per_vector": round(index.get_bytes_per_vector(), 2),
            }
            log_result(row)
            results.append(row)
            print(f"    nprobe={nprobe:<4} -> Recall@10={rec:.4f} | QPS={t_res.qps:<8.1f} | Latency={t_res.latency_ms_per_query:.2f} ms")

    plot_individual_sweep(
        dataset_name=dataset_name,
        index_type="IVF-Flat",
        results=results,
        output_path=SWEEPS_DIR / f"{dataset_name}_ivf_flat_sweep.png",
    )
    return results


def benchmark_ivf_pq_sweeps(
    dataset_name: str,
    splits,
    nlist: int,
    m_pq_list: List[int],
    nprobe_list: List[int],
    nbits: int,
    metric: str,
    num_runs: int = 3,
    max_eval_queries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Runs build and search sweeps for IVF-PQ."""
    print(f"\n[{dataset_name.upper()}] Benchmarking IVF-PQ sweeps across M_pq={m_pq_list} (nlist={nlist})...")
    results: List[Dict[str, Any]] = []

    for M_pq in m_pq_list:
        print(f"  Building IVF-PQ (nlist={nlist}, M_pq={M_pq}, nbits={nbits})...")
        index = IVFPQIndex(dim=splits.dim, nlist=nlist, M_pq=M_pq, nbits=nbits, metric=metric)
        index.build(splits.train, num_threads=16)
        size_mb = index.get_size_bytes() / (1024 * 1024)
        print(f"    Built in {index.build_time:.2f}s | Size: {size_mb:.1f} MB ({index.get_bytes_per_vector():.1f} bytes/vec)")

        for nprobe in nprobe_list:
            if nprobe > nlist:
                continue
            t_res, cand_dists, cand_idx = time_single_query_search(
                index=index,
                queries=splits.test_eval,
                k=DEFAULT_K,
                num_runs=num_runs,
                threads=1,
                max_eval_queries=max_eval_queries,
                nprobe=nprobe,
            )

            rec = recall_at_k(
                retrieved_distances=cand_dists,
                gt_distances=splits.gt_distances_eval[: len(cand_dists)],
                retrieved_indices=cand_idx,
                metric=metric,
                k=DEFAULT_K,
                eps=DEFAULT_EPSILON,
            )

            row = {
                "dataset": dataset_name,
                "index_type": "IVF-PQ",
                "threads": 1,
                "query_mode": "single",
                "build_params": f"nlist={nlist},M_pq={M_pq}",
                "search_params": f"nprobe={nprobe}",
                "recall_at_10": round(rec, 4),
                "qps": round(t_res.qps, 2),
                "latency_ms": round(t_res.latency_ms_per_query, 2),
                "build_time_s": round(index.build_time, 2),
                "index_size_bytes": index.get_size_bytes(),
                "bytes_per_vector": round(index.get_bytes_per_vector(), 2),
            }
            log_result(row)
            results.append(row)
            print(f"    nprobe={nprobe:<4} -> Recall@10={rec:.4f} | QPS={t_res.qps:<8.1f} | Latency={t_res.latency_ms_per_query:.2f} ms")

    plot_individual_sweep(
        dataset_name=dataset_name,
        index_type="IVF-PQ",
        results=results,
        output_path=SWEEPS_DIR / f"{dataset_name}_ivf_pq_sweep.png",
    )
    return results


def print_qps_summary_table(dataset_name: str, pareto_by_index: Dict[str, List[Dict[str, Any]]]) -> None:
    """Prints the required table of QPS at recall 0.90 and 0.95."""
    print(f"\n{'='*75}")
    print(f"TABLE 1: {dataset_name.upper()} QUERIES PER SECOND AT RECALL 0.90 AND 0.95")
    print(f"{'='*75}")
    print(f"{'Index Type':<16} | {'QPS @ Recall 0.90':<20} | {'QPS @ Recall 0.95':<20} | {'Max Recall Achieved':<20}")
    print(f"{'-'*16}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")

    for idx_name, pts in pareto_by_index.items():
        if not pts:
            continue
        max_rec = max(p["recall_at_10"] for p in pts)
        q_090 = interpolate_qps_at_recall(pts, 0.90)
        q_095 = interpolate_qps_at_recall(pts, 0.95)

        q_090_str = f"{q_090:.1f} QPS" if q_090 is not None else "N/A (< 0.90)"
        q_095_str = f"{q_095:.1f} QPS" if q_095 is not None else "N/A (< 0.95)"
        print(f"{idx_name:<16} | {q_090_str:<20} | {q_095_str:<20} | {max_rec:<20.4f}")
    print(f"{'='*75}")


def print_build_cost_summary_table(
    dataset_name: str,
    all_results_by_index: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Prints the required table of build time, index size, and bytes/vector at recall 0.95."""
    print(f"\n{'='*85}")
    print(f"TABLE 2: {dataset_name.upper()} BUILD TIME, INDEX SIZE, AND BYTES/VECTOR AT RECALL 0.95")
    print(f"{'='*85}")
    print(f"{'Index Type':<12} | {'Config @ Recall >= 0.95':<25} | {'Build Time (s)':<15} | {'Size (MB)':<12} | {'Bytes/Vec':<12}")
    print(f"{'-'*12}-+-{'-'*25}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}")

    for idx_name, pts in all_results_by_index.items():
        # Find points reaching recall >= 0.95
        eligible = [p for p in pts if p["recall_at_10"] >= 0.95]
        if eligible:
            # Pick the fastest configuration (highest QPS) that achieved recall >= 0.95
            best = max(eligible, key=lambda x: x["qps"])
            config_str = f"{best['build_params']} ({best['search_params']})"
            size_mb = best["index_size_bytes"] / (1024 * 1024)
            print(
                f"{idx_name:<12} | {config_str:<25} | {best['build_time_s']:<15.2f} | {size_mb:<12.1f} | {best['bytes_per_vector']:<12.1f}"
            )
        else:
            max_pt = max(pts, key=lambda x: x["recall_at_10"]) if pts else None
            max_rec_str = f"Max recall: {max_pt['recall_at_10']:.4f}" if max_pt else "No runs"
            print(f"{idx_name:<12} | {'Did not reach 0.95':<25} | {'N/A (' + max_rec_str + ')':<15} | {'N/A':<12} | {'N/A':<12}")
    print(f"{'='*85}")


def print_recall_ceiling_explanation():
    """Prints the mechanism explanation for why IVF-PQ fails to reach high recall."""
    explanation = """
====================================================================================================
ARCHITECTURAL ANALYSIS: WHY IVF-PQ EXHIBITS A RECALL CEILING
====================================================================================================
1. Asymmetric Distance Quantization Distortion (ADC):
   - HNSW and IVF-Flat store raw, full-precision 32-bit floating-point vectors. As search parameters
     expand (efSearch -> large or nprobe -> nlist), their distance evaluations are exact, permitting
     recall to monotonically approach 1.000.
   - In contrast, IVF-PQ maps high-dimensional vectors to M_pq discrete sub-vector centroid codes
     (e.g., 8-bit integers). During search, distances are estimated using Asymmetric Distance Computation
     (ADC), which measures query distance to the nearest sub-space centroids rather than the true vector.

2. Irreversible Loss of High-Frequency Variance:
   - Quantizing 128 dimensions into 16 or 32 bytes compresses vectors by 4x to 8x, but introduces
     irreversible quantization distortion. At the boundary of the top-10 nearest neighbours, true
     pairwise distance differences between candidates are minute (often < 1% relative distance).
   - The quantization noise exceeds these fine distance differences, corrupting the candidate ranking.
     Consequently, even if all inverted lists are probed (nprobe = nlist), IVF-PQ hits an asymptotic
     recall plateau (< 0.95) unless M_pq is set very high or exact candidate re-ranking is performed.
====================================================================================================
"""
    print(explanation)


def run_part2_for_dataset(
    dataset_name: str,
    quick: bool = False,
    num_runs: int = 3,
) -> None:
    """Executes Part 2 for a single dataset."""
    print(f"\n{'='*75}")
    print(f"RUNNING PART 2 BENCHMARKS FOR DATASET: {dataset_name.upper()}")
    print(f"{'='*75}")

    max_train = 100_000 if quick else None
    max_eval_queries = 500 if quick else None
    num_runs = 2 if quick else num_runs

    # Parameter selection
    if quick:
        hnsw_m = QUICK_HNSW_M_LIST
        hnsw_ef = QUICK_HNSW_EF_SEARCH
        ivf_nlist = QUICK_IVF_NLIST_LIST
        ivf_nprobe = QUICK_IVF_NPROBE
        pq_m = QUICK_SIFT_PQ_M if dataset_name == "sift" else QUICK_GLOVE_PQ_M
        pq_nprobe = QUICK_IVF_NPROBE
    else:
        hnsw_m = HNSW_M_LIST
        hnsw_ef = HNSW_EF_SEARCH_LIST
        ivf_nlist = IVF_FLAT_NLIST_LIST
        ivf_nprobe = IVF_FLAT_NPROBE_LIST
        pq_m = SIFT_PQ_M_LIST if dataset_name == "sift" else GLOVE_PQ_M_LIST
        pq_nprobe = IVF_PQ_NPROBE_LIST

    # 1. Load data
    print(f"Loading {dataset_name} dataset...")
    splits = load_dataset(dataset_name, max_train=max_train, normalize_glove=True)
    metric = "euclidean" if dataset_name == "sift" else "angular"
    print(f"Loaded train: {splits.train.shape[0]} vectors ({splits.dim}-d), eval: {splits.test_eval.shape[0]} queries")

    # 2. Get exact baseline QPS for reference
    exact_qps = get_exact_baseline_qps(dataset_name, splits, num_runs=num_runs)
    print(f"Exact baseline single-query throughput: {exact_qps:.2f} QPS")

    # 3. Benchmark HNSW
    hnsw_results = benchmark_hnsw_sweeps(
        dataset_name=dataset_name,
        splits=splits,
        m_list=hnsw_m,
        ef_search_list=hnsw_ef,
        ef_construction=HNSW_EF_CONSTRUCTION,
        metric=metric,
        num_runs=num_runs,
        max_eval_queries=max_eval_queries,
    )

    # 4. Benchmark IVF-Flat
    ivf_flat_results = benchmark_ivf_flat_sweeps(
        dataset_name=dataset_name,
        splits=splits,
        nlist_list=ivf_nlist,
        nprobe_list=ivf_nprobe,
        metric=metric,
        num_runs=num_runs,
        max_eval_queries=max_eval_queries,
    )

    # 5. Benchmark IVF-PQ
    ivf_pq_results = benchmark_ivf_pq_sweeps(
        dataset_name=dataset_name,
        splits=splits,
        nlist=IVF_PQ_NLIST if not quick else 512,
        m_pq_list=pq_m,
        nprobe_list=pq_nprobe,
        nbits=IVF_PQ_NBITS,
        metric=metric,
        num_runs=num_runs,
        max_eval_queries=max_eval_queries,
    )

    # 6. Compute Pareto Frontiers
    pareto_hnsw = compute_pareto_frontier(hnsw_results)
    pareto_ivf_flat = compute_pareto_frontier(ivf_flat_results)
    pareto_ivf_pq = compute_pareto_frontier(ivf_pq_results)

    pareto_by_index = {
        "HNSW": pareto_hnsw,
        "IVF-Flat": pareto_ivf_flat,
        "IVF-PQ": pareto_ivf_pq,
    }

    # 7. Generate Consolidated Pareto Plot
    plot_overall_pareto(
        dataset_name=dataset_name,
        pareto_by_index=pareto_by_index,
        exact_baseline_qps=exact_qps,
        output_path=FIGURES_DIR / f"{dataset_name}_recall_vs_qps.png",
    )

    # 8. Print Tables and Analysis
    print_qps_summary_table(dataset_name, pareto_by_index)
    all_results = {
        "HNSW": hnsw_results,
        "IVF-Flat": ivf_flat_results,
        "IVF-PQ": ivf_pq_results,
    }
    print_build_cost_summary_table(dataset_name, all_results)
    print_recall_ceiling_explanation()


def main():
    parser = argparse.ArgumentParser(description="Part 2: Benchmark HNSW, IVF-Flat, and IVF-PQ")
    parser.add_argument(
        "--dataset",
        choices=["sift", "glove", "all"],
        default="all",
        help="Dataset to benchmark (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick verification on 100K vectors and smaller sweep grids",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs to take median timing (default: 3)",
    )

    args = parser.parse_args()
    datasets = ["sift", "glove"] if args.dataset == "all" else [args.dataset]

    for d in datasets:
        run_part2_for_dataset(
            dataset_name=d,
            quick=args.quick,
            num_runs=args.num_runs,
        )


if __name__ == "__main__":
    main()
