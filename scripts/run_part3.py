"""CLI runner to execute Part 3: Scaling Experiment on SIFT (100K, 300K, 1M)."""

import argparse
import csv
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
    plot_overall_pareto,
    plot_scaling_curves,
)
from src.benchmark.reporter import log_result
from src.benchmark.timing import time_single_query_search
from src.config import (
    DEFAULT_EPSILON,
    DEFAULT_K,
    PART3_FIGURES_DIR,
    PART3_HNSW_EF_CONSTRUCTION,
    PART3_HNSW_EF_SEARCH_LIST,
    PART3_HNSW_M_LIST,
    PART3_IVF_NLIST_MAP,
    PART3_IVF_NPROBE_LIST,
    QUICK_PART3_HNSW_EF_SEARCH,
    QUICK_PART3_HNSW_M,
    QUICK_PART3_IVF_NLIST,
    QUICK_PART3_IVF_NPROBE,
    QUICK_PART3_SIZES,
    RESULTS_CSV,
    SCALING_DATASET_SIZES,
)
from src.data import load_dataset
from src.index.exact import ExactFlatIndex
from src.index.hnsw import HNSWIndex
from src.index.ivf_flat import IVFFlatIndex
from src.metrics import recall_at_k


def load_1m_sift_results_from_csv() -> Optional[Dict[str, Any]]:
    """Loads existing 1M SIFT benchmark rows from results.csv if available."""
    if not RESULTS_CSV.exists():
        return None

    exact_row = None
    hnsw_rows = []
    ivf_rows = []

    with open(RESULTS_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") == "sift":
                row_dict = {
                    "dataset": "sift",
                    "index_type": row.get("index_type"),
                    "threads": int(row.get("threads", 1)),
                    "query_mode": row.get("query_mode"),
                    "build_params": row.get("build_params"),
                    "search_params": row.get("search_params"),
                    "recall_at_10": float(row.get("recall_at_10", 0.0)),
                    "qps": float(row.get("qps", 0.0)),
                    "latency_ms": float(row.get("latency_ms", 0.0)),
                    "build_time_s": float(row.get("build_time_s", 0.0)),
                    "index_size_bytes": int(float(row.get("index_size_bytes", 0))),
                    "bytes_per_vector": float(row.get("bytes_per_vector", 0.0)),
                }
                if row.get("index_type") == "Exact-Flat" and row.get("query_mode") == "single":
                    exact_row = row_dict
                elif row.get("index_type") == "HNSW":
                    hnsw_rows.append(row_dict)
                elif row.get("index_type") == "IVF-Flat":
                    ivf_rows.append(row_dict)

    if exact_row and hnsw_rows and ivf_rows:
        return {
            "exact": exact_row,
            "hnsw": hnsw_rows,
            "ivf": ivf_rows,
        }
    return None


def run_benchmark_for_size(
    size: int,
    num_runs: int = 2,
    quick: bool = False,
    max_eval_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs Exact, HNSW, and IVF-Flat benchmarks for a specific subset size N."""
    size_label = f"{int(size/1000)}k" if size < 1_000_000 else "1m"
    dataset_name = f"sift_{size_label}"
    print(f"\n=======================================================")
    print(f"  BENCHMARKING SIFT SCALING: N = {size:,} ({size_label.upper()})")
    print(f"=======================================================")

    print(f"Loading SIFT dataset (N = {size:,})...")
    splits = load_dataset("sift", max_train=size)
    eval_queries = splits.test_eval if max_eval_queries is None else splits.test_eval[:max_eval_queries]
    gt_dists_eval = splits.gt_distances_eval if max_eval_queries is None else splits.gt_distances_eval[:max_eval_queries]

    # 1. Exact-Flat Baseline
    print(f"\n[{size_label.upper()}] Benchmarking Exact-Flat Baseline...")
    exact_idx = ExactFlatIndex(dim=splits.dim)
    exact_idx.build(splits.train)

    # In quick mode, evaluate exact on a smaller sample to save time
    exact_eval_queries = eval_queries if not quick else eval_queries[:500]
    exact_gt_dists = gt_dists_eval if not quick else gt_dists_eval[:500]

    t_res, exact_dists, exact_ids = time_single_query_search(
        exact_idx,
        exact_eval_queries,
        k=DEFAULT_K,
        num_runs=num_runs,
        threads=1,
    )
    recalls = recall_at_k(
        exact_dists,
        exact_gt_dists,
        exact_ids,
        metric="euclidean",
        k=DEFAULT_K,
        eps=DEFAULT_EPSILON,
    )
    exact_recall = float(np.mean(recalls))

    exact_record = {
        "dataset": dataset_name,
        "index_type": "Exact-Flat",
        "threads": 1,
        "query_mode": "single",
        "build_params": "{}",
        "search_params": "{}",
        "recall_at_10": round(exact_recall, 4),
        "qps": round(t_res.qps, 2),
        "latency_ms": round(t_res.latency_ms_per_query, 2),
        "build_time_s": round(exact_idx.build_time, 2),
        "index_size_bytes": exact_idx.get_size_bytes(),
        "bytes_per_vector": round(exact_idx.get_bytes_per_vector(), 2),
    }
    log_result(exact_record)
    print(f"  Exact-Flat: Recall={exact_recall:.4f}, QPS={t_res.qps:.2f}, Latency={t_res.latency_ms_per_query:.2f}ms")

    # 2. HNSW Sweeps
    m_list = QUICK_PART3_HNSW_M if quick else PART3_HNSW_M_LIST
    ef_search_list = QUICK_PART3_HNSW_EF_SEARCH if quick else PART3_HNSW_EF_SEARCH_LIST
    print(f"\n[{size_label.upper()}] Benchmarking HNSW sweeps across M={m_list}...")
    hnsw_records: List[Dict[str, Any]] = []

    for M in m_list:
        hnsw_idx = HNSWIndex(dim=splits.dim, M=M, efConstruction=PART3_HNSW_EF_CONSTRUCTION)
        hnsw_idx.build(splits.train)
        print(f"  Built HNSW (M={M}) in {hnsw_idx.build_time:.2f}s, Size={hnsw_idx.get_size_bytes() / (1024**2):.1f}MB")

        for efSearch in ef_search_list:
            t_res, dists, ids = time_single_query_search(
                hnsw_idx,
                eval_queries,
                k=DEFAULT_K,
                num_runs=num_runs,
                threads=1,
                efSearch=efSearch,
            )
            rec = float(np.mean(recall_at_k(dists, gt_dists_eval, ids, metric="euclidean", k=DEFAULT_K, eps=DEFAULT_EPSILON)))
            rec_dict = {
                "dataset": dataset_name,
                "index_type": "HNSW",
                "threads": 1,
                "query_mode": "single",
                "build_params": f"M={M},efConst={PART3_HNSW_EF_CONSTRUCTION}",
                "search_params": f"efSearch={efSearch}",
                "recall_at_10": round(rec, 4),
                "qps": round(t_res.qps, 2),
                "latency_ms": round(t_res.latency_ms_per_query, 2),
                "build_time_s": round(hnsw_idx.build_time, 2),
                "index_size_bytes": hnsw_idx.get_size_bytes(),
                "bytes_per_vector": round(hnsw_idx.get_bytes_per_vector(), 2),
            }
            log_result(rec_dict)
            hnsw_records.append(rec_dict)
            print(f"    M={M:2d}, efSearch={efSearch:3d} -> Recall={rec:.4f}, QPS={t_res.qps:8.1f}")

    # 3. IVF-Flat Sweeps
    nlist_list = QUICK_PART3_IVF_NLIST if quick else PART3_IVF_NLIST_MAP.get(size, [512, 1024, 2048])
    nprobe_list = QUICK_PART3_IVF_NPROBE if quick else PART3_IVF_NPROBE_LIST
    print(f"\n[{size_label.upper()}] Benchmarking IVF-Flat sweeps across nlist={nlist_list}...")
    ivf_records: List[Dict[str, Any]] = []

    for nlist in nlist_list:
        ivf_idx = IVFFlatIndex(dim=splits.dim, nlist=nlist)
        ivf_idx.build(splits.train)
        print(f"  Built IVF-Flat (nlist={nlist}) in {ivf_idx.build_time:.2f}s, Size={ivf_idx.get_size_bytes() / (1024**2):.1f}MB")

        for nprobe in nprobe_list:
            if nprobe > nlist:
                continue
            t_res, dists, ids = time_single_query_search(
                ivf_idx,
                eval_queries,
                k=DEFAULT_K,
                num_runs=num_runs,
                threads=1,
                nprobe=nprobe,
            )
            rec = float(np.mean(recall_at_k(dists, gt_dists_eval, ids, metric="euclidean", k=DEFAULT_K, eps=DEFAULT_EPSILON)))
            rec_dict = {
                "dataset": dataset_name,
                "index_type": "IVF-Flat",
                "threads": 1,
                "query_mode": "single",
                "build_params": f"nlist={nlist}",
                "search_params": f"nprobe={nprobe}",
                "recall_at_10": round(rec, 4),
                "qps": round(t_res.qps, 2),
                "latency_ms": round(t_res.latency_ms_per_query, 2),
                "build_time_s": round(ivf_idx.build_time, 2),
                "index_size_bytes": ivf_idx.get_size_bytes(),
                "bytes_per_vector": round(ivf_idx.get_bytes_per_vector(), 2),
            }
            log_result(rec_dict)
            ivf_records.append(rec_dict)
            print(f"    nlist={nlist:4d}, nprobe={nprobe:3d} -> Recall={rec:.4f}, QPS={t_res.qps:8.1f}")

    return {
        "exact": exact_record,
        "hnsw": hnsw_records,
        "ivf": ivf_records,
    }


def analyze_scaling_results(
    all_scale_data: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extracts QPS, Latency, Build Time, and Memory at Recall 0.95 across all sizes N.
    """
    scaling_summary: List[Dict[str, Any]] = []

    for size, data in all_scale_data.items():
        exact_rec = data["exact"]
        hnsw_recs = data["hnsw"]
        ivf_recs = data["ivf"]

        # Exact: recall = 1.000
        scaling_summary.append({
            "dataset_size": size,
            "index_type": "Exact-Flat",
            "qps": exact_rec["qps"],
            "latency_ms": exact_rec["latency_ms"],
            "build_time_s": exact_rec["build_time_s"],
            "index_size_mb": exact_rec["index_size_bytes"] / (1024**2),
            "bytes_per_vector": exact_rec["bytes_per_vector"],
            "operating_recall": 1.0000,
            "config": "Exact Brute-Force",
        })

        # HNSW: interpolate at recall 0.95
        p_hnsw = compute_pareto_frontier(hnsw_recs)
        qps_hnsw_95 = interpolate_qps_at_recall(p_hnsw, 0.95)
        # Find best config reaching recall >= 0.95
        candidates = [r for r in hnsw_recs if r["recall_at_10"] >= 0.95]
        if candidates:
            best_hnsw = sorted(candidates, key=lambda x: -x["qps"])[0]
            scaling_summary.append({
                "dataset_size": size,
                "index_type": "HNSW",
                "qps": qps_hnsw_95 if qps_hnsw_95 is not None else best_hnsw["qps"],
                "latency_ms": 1000.0 / (qps_hnsw_95 if qps_hnsw_95 is not None else best_hnsw["qps"]),
                "build_time_s": best_hnsw["build_time_s"],
                "index_size_mb": best_hnsw["index_size_bytes"] / (1024**2),
                "bytes_per_vector": best_hnsw["bytes_per_vector"],
                "operating_recall": best_hnsw["recall_at_10"],
                "config": f"{best_hnsw['build_params']}, {best_hnsw['search_params']}",
            })

        # IVF-Flat: interpolate at recall 0.95
        p_ivf = compute_pareto_frontier(ivf_recs)
        qps_ivf_95 = interpolate_qps_at_recall(p_ivf, 0.95)
        candidates = [r for r in ivf_recs if r["recall_at_10"] >= 0.95]
        if candidates:
            best_ivf = sorted(candidates, key=lambda x: -x["qps"])[0]
            scaling_summary.append({
                "dataset_size": size,
                "index_type": "IVF-Flat",
                "qps": qps_ivf_95 if qps_ivf_95 is not None else best_ivf["qps"],
                "latency_ms": 1000.0 / (qps_ivf_95 if qps_ivf_95 is not None else best_ivf["qps"]),
                "build_time_s": best_ivf["build_time_s"],
                "index_size_mb": best_ivf["index_size_bytes"] / (1024**2),
                "bytes_per_vector": best_ivf["bytes_per_vector"],
                "operating_recall": best_ivf["recall_at_10"],
                "config": f"{best_ivf['build_params']}, {best_ivf['search_params']}",
            })

    return scaling_summary


def print_scaling_table(scaling_summary: List[Dict[str, Any]]) -> None:
    """Prints a formatted markdown table comparing performance at recall 0.95."""
    print("\n" + "=" * 90)
    print("  PART 3 SCALING EXPERIMENT SUMMARY: PERFORMANCE AT RECALL ≥ 0.95")
    print("=" * 90)
    print(
        f"{'Size (N)':<10} | {'Index Type':<12} | {'Recall':<8} | {'QPS @ 0.95':<12} | {'Latency (ms)':<14} | {'Build (s)':<10} | {'Size (MB)':<10} | {'Bytes/Vec':<10}"
    )
    print("-" * 90)
    for row in scaling_summary:
        size_lbl = f"{int(row['dataset_size']/1000)}K" if row["dataset_size"] < 1_000_000 else "1M"
        print(
            f"{size_lbl:<10} | {row['index_type']:<12} | {row['operating_recall']:<8.4f} | {row['qps']:<12.1f} | {row['latency_ms']:<14.2f} | {row['build_time_s']:<10.2f} | {row['index_size_mb']:<10.1f} | {row['bytes_per_vector']:<10.1f}"
        )
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Part 3: SIFT Scaling Experiment CLI Runner")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=SCALING_DATASET_SIZES,
        help="Dataset sizes to benchmark (default: 100000 300000 1000000)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Number of timing repetitions to take median (default: 2)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: reduced sweep parameters for rapid verification",
    )
    parser.add_argument(
        "--rerun-1m",
        action="store_true",
        help="Force re-running 1M SIFT benchmark even if present in results.csv",
    )
    args = parser.parse_args()

    sizes = QUICK_PART3_SIZES if args.quick else args.sizes
    all_scale_data: Dict[int, Dict[str, Any]] = {}

    for size in sizes:
        if size == 1_000_000 and not args.rerun_1m:
            cached_1m = load_1m_sift_results_from_csv()
            if cached_1m is not None:
                print(f"\n[1M SIFT] Reusing existing benchmark numbers from results.csv (use --rerun-1m to recompute).")
                all_scale_data[size] = cached_1m
                continue

        data = run_benchmark_for_size(
            size=size,
            num_runs=args.num_runs,
            quick=args.quick,
            max_eval_queries=500 if args.quick else None,
        )
        all_scale_data[size] = data

        # Plot Pareto curve for this specific subset
        p_hnsw = compute_pareto_frontier(data["hnsw"])
        p_ivf = compute_pareto_frontier(data["ivf"])
        size_lbl = f"{int(size/1000)}k" if size < 1_000_000 else "1m"
        pareto_map = {"HNSW": p_hnsw, "IVF-Flat": p_ivf}
        tradeoff_path = PART3_FIGURES_DIR / f"sift_{size_lbl}_recall_vs_qps.png"
        plot_overall_pareto(
            dataset_name=f"sift-{size_lbl}",
            pareto_by_index=pareto_map,
            exact_baseline_qps=data["exact"]["qps"],
            output_path=tradeoff_path,
        )

    # Scaling analysis across sizes
    scaling_summary = analyze_scaling_results(all_scale_data)
    print_scaling_table(scaling_summary)

    # Generate Scaling Plots if at least 2 sizes were evaluated
    if len(all_scale_data) >= 2:
        print("Generating Part 3 Scaling Curves...")
        plot_scaling_curves(scaling_summary, output_dir=PART3_FIGURES_DIR)


if __name__ == "__main__":
    main()
