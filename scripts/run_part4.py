"""CLI runner to execute Part 4: Hard Geometry (Cluster Separation) Experiment."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.plotting import (
    compute_pareto_frontier,
    interpolate_qps_at_recall,
    plot_part4_hard_geometry,
)
from src.benchmark.reporter import log_result
from src.benchmark.timing import time_single_query_search
from src.config import (
    DEFAULT_EPSILON,
    DEFAULT_K,
    PART4_CONTROL_SEPARATION,
    PART4_DIM,
    PART4_FIGURES_DIR,
    PART4_HNSW_EF_CONSTRUCTION,
    PART4_HNSW_EF_SEARCH_LIST,
    PART4_HNSW_M,
    PART4_IVF_NLIST,
    PART4_IVF_NPROBE_LIST,
    PART4_N_CLUSTERS,
    PART4_N_QUERIES,
    PART4_N_VECTORS,
    PART4_SEED,
    PART4_SEPARATIONS,
    QUICK_PART4_HNSW_EF_SEARCH,
    QUICK_PART4_IVF_NLIST,
    QUICK_PART4_IVF_NPROBE,
    QUICK_PART4_N_QUERIES,
    QUICK_PART4_N_VECTORS,
    QUICK_PART4_SEPARATIONS,
    RESULTS_CSV,
)
from src.data_generator import generate_clustered_dataset
from src.index.exact import ExactFlatIndex
from src.index.hnsw import HNSWIndex
from src.index.ivf_flat import IVFFlatIndex
from src.metrics import recall_at_k


def run_hard_geometry_benchmark(
    separations: List[float],
    n_vectors: int,
    n_queries: int,
    dim: int,
    n_clusters: int,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search_list: List[int],
    ivf_nlist: int,
    ivf_nprobe_list: List[int],
    num_runs: int = 2,
    seed: int = PART4_SEED,
    force_recompute: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Runs the complete Part 4 benchmark across the specified separation multipliers.

    Returns:
        Tuple of (all_runs, summary_records).
    """
    print("\n" + "=" * 80)
    print("  PART 4: HARD GEOMETRY EXPERIMENT (CLUSTER SEPARATION)")
    print(f"  Settings: N={n_vectors:,} | D={dim} | K={n_clusters} | Separations={separations}")
    print("=" * 80)

    print("\nGenerating / Loading Synthetic Datasets with shared centroids, assignments, and noise...")
    datasets = generate_clustered_dataset(
        n_vectors=n_vectors,
        n_queries=n_queries,
        dim=dim,
        n_clusters=n_clusters,
        separations=separations,
        seed=seed,
        force_recompute=force_recompute,
    )
    print_part4_diagnostics_table(datasets)

    all_runs: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    for s in separations:
        splits = datasets[s]
        tag = f"synthetic_sep_{s}"
        ctrl_mark = " (CONTROL)" if s == PART4_CONTROL_SEPARATION else ""
        print(f"\n>>> Benchmarking Separation Multiplier: s = {s}{ctrl_mark} ({splits.name})")

        # 1. Exact-Flat Reference
        print(f"  [Exact-Flat] Measuring single-query baseline...")
        exact_idx = ExactFlatIndex(dim=splits.dim)
        exact_idx.build(splits.train)

        t_res, exact_dists, exact_ids = time_single_query_search(
            exact_idx,
            splits.test_eval,
            k=DEFAULT_K,
            num_runs=num_runs,
            threads=1,
        )
        recalls = recall_at_k(
            exact_dists,
            splits.gt_distances_eval,
            exact_ids,
            metric="euclidean",
            k=DEFAULT_K,
            eps=DEFAULT_EPSILON,
        )
        exact_recall = float(np.mean(recalls))
        print(f"    Exact-Flat: Recall={exact_recall:.4f}, QPS={t_res.qps:.1f}, Latency={t_res.latency_ms_per_query:.2f}ms")

        exact_row = {
            "dataset": tag,
            "separation": s,
            "index_type": "Exact-Flat",
            "threads": 1,
            "query_mode": "single",
            "build_params": "{}",
            "search_params": "{}",
            "search_param_val": 0,
            "recall_at_10": round(exact_recall, 4),
            "qps": round(t_res.qps, 2),
            "latency_ms": round(t_res.latency_ms_per_query, 2),
            "build_time_s": round(exact_idx.build_time, 2),
            "index_size_bytes": exact_idx.get_size_bytes(),
            "bytes_per_vector": round(exact_idx.get_bytes_per_vector(), 2),
        }
        log_result(exact_row)
        all_runs.append(exact_row)

        summary_records.append({
            "separation": s,
            "index_type": "Exact-Flat",
            "qps_at_95": t_res.qps,
            "latency_ms": t_res.latency_ms_per_query,
            "param_at_95": 0,
            "operating_recall": exact_recall,
            "qps_at_param": t_res.qps,
            "build_time_s": exact_idx.build_time,
            "index_size_mb": exact_idx.get_size_bytes() / (1024**2),
        })

        # 2. HNSW Benchmark (Fixed M, efConstruction)
        print(f"  [HNSW] Building index (M={hnsw_m}, efConstruction={hnsw_ef_construction})...")
        hnsw_idx = HNSWIndex(dim=splits.dim, M=hnsw_m, efConstruction=hnsw_ef_construction)
        hnsw_idx.build(splits.train)
        print(f"    Built HNSW in {hnsw_idx.build_time:.2f}s | Size: {hnsw_idx.get_size_bytes() / (1024**2):.1f} MB")

        hnsw_rows_s: List[Dict[str, Any]] = []
        for ef_search in hnsw_ef_search_list:
            t_res, dists, ids = time_single_query_search(
                hnsw_idx,
                splits.test_eval,
                k=DEFAULT_K,
                num_runs=num_runs,
                threads=1,
                efSearch=ef_search,
            )
            rec = float(np.mean(recall_at_k(dists, splits.gt_distances_eval, ids, metric="euclidean", k=DEFAULT_K, eps=DEFAULT_EPSILON)))
            row = {
                "dataset": tag,
                "separation": s,
                "index_type": "HNSW",
                "threads": 1,
                "query_mode": "single",
                "build_params": f"M={hnsw_m},efConst={hnsw_ef_construction}",
                "search_params": f"efSearch={ef_search}",
                "search_param_val": ef_search,
                "recall_at_10": round(rec, 4),
                "qps": round(t_res.qps, 2),
                "latency_ms": round(t_res.latency_ms_per_query, 2),
                "build_time_s": round(hnsw_idx.build_time, 2),
                "index_size_bytes": hnsw_idx.get_size_bytes(),
                "bytes_per_vector": round(hnsw_idx.get_bytes_per_vector(), 2),
            }
            log_result(row)
            all_runs.append(row)
            hnsw_rows_s.append(row)
            print(f"    efSearch={ef_search:3d} -> Recall={rec:.4f}, QPS={t_res.qps:8.1f}")

        # HNSW Analysis at Recall >= 0.95
        p_hnsw = compute_pareto_frontier(hnsw_rows_s)
        qps_hnsw_95 = interpolate_qps_at_recall(p_hnsw, 0.95)
        # Find first efSearch achieving recall >= 0.95
        cand_hnsw = [r for r in hnsw_rows_s if r["recall_at_10"] >= 0.95]
        if cand_hnsw:
            first_ef_95 = min(cand_hnsw, key=lambda x: x["search_param_val"])
            summary_records.append({
                "separation": s,
                "index_type": "HNSW",
                "param_at_95": first_ef_95["search_param_val"],
                "operating_recall": first_ef_95["recall_at_10"],
                "qps_at_param": first_ef_95["qps"],
                "latency_ms": first_ef_95["latency_ms"],
                "qps_at_95": qps_hnsw_95 if qps_hnsw_95 is not None else first_ef_95["qps"],
                "build_time_s": hnsw_idx.build_time,
                "index_size_mb": hnsw_idx.get_size_bytes() / (1024**2),
            })
        else:
            best_r = max(hnsw_rows_s, key=lambda x: x["recall_at_10"])
            summary_records.append({
                "separation": s,
                "index_type": "HNSW",
                "param_at_95": best_r["search_param_val"],
                "operating_recall": best_r["recall_at_10"],
                "qps_at_param": best_r["qps"],
                "latency_ms": best_r["latency_ms"],
                "qps_at_95": best_r["qps"],
                "build_time_s": hnsw_idx.build_time,
                "index_size_mb": hnsw_idx.get_size_bytes() / (1024**2),
            })

        # 3. IVF-Flat Benchmark (Fixed nlist)
        print(f"  [IVF-Flat] Building index (nlist={ivf_nlist})...")
        ivf_idx = IVFFlatIndex(dim=splits.dim, nlist=ivf_nlist)
        ivf_idx.build(splits.train)
        print(f"    Built IVF-Flat in {ivf_idx.build_time:.2f}s | Size: {ivf_idx.get_size_bytes() / (1024**2):.1f} MB")

        ivf_rows_s: List[Dict[str, Any]] = []
        for nprobe in ivf_nprobe_list:
            if nprobe > ivf_nlist:
                continue
            t_res, dists, ids = time_single_query_search(
                ivf_idx,
                splits.test_eval,
                k=DEFAULT_K,
                num_runs=num_runs,
                threads=1,
                nprobe=nprobe,
            )
            rec = float(np.mean(recall_at_k(dists, splits.gt_distances_eval, ids, metric="euclidean", k=DEFAULT_K, eps=DEFAULT_EPSILON)))
            row = {
                "dataset": tag,
                "separation": s,
                "index_type": "IVF-Flat",
                "threads": 1,
                "query_mode": "single",
                "build_params": f"nlist={ivf_nlist}",
                "search_params": f"nprobe={nprobe}",
                "search_param_val": nprobe,
                "recall_at_10": round(rec, 4),
                "qps": round(t_res.qps, 2),
                "latency_ms": round(t_res.latency_ms_per_query, 2),
                "build_time_s": round(ivf_idx.build_time, 2),
                "index_size_bytes": ivf_idx.get_size_bytes(),
                "bytes_per_vector": round(ivf_idx.get_bytes_per_vector(), 2),
            }
            log_result(row)
            all_runs.append(row)
            ivf_rows_s.append(row)
            print(f"    nprobe={nprobe:3d} -> Recall={rec:.4f}, QPS={t_res.qps:8.1f}")

        # IVF-Flat Analysis at Recall >= 0.95
        p_ivf = compute_pareto_frontier(ivf_rows_s)
        qps_ivf_95 = interpolate_qps_at_recall(p_ivf, 0.95)
        # Find first nprobe achieving recall >= 0.95
        cand_ivf = [r for r in ivf_rows_s if r["recall_at_10"] >= 0.95]
        if cand_ivf:
            first_nprobe_95 = min(cand_ivf, key=lambda x: x["search_param_val"])
            summary_records.append({
                "separation": s,
                "index_type": "IVF-Flat",
                "param_at_95": first_nprobe_95["search_param_val"],
                "operating_recall": first_nprobe_95["recall_at_10"],
                "qps_at_param": first_nprobe_95["qps"],
                "latency_ms": first_nprobe_95["latency_ms"],
                "qps_at_95": qps_ivf_95 if qps_ivf_95 is not None else first_nprobe_95["qps"],
                "build_time_s": ivf_idx.build_time,
                "index_size_mb": ivf_idx.get_size_bytes() / (1024**2),
            })
        else:
            best_r = max(ivf_rows_s, key=lambda x: x["recall_at_10"])
            summary_records.append({
                "separation": s,
                "index_type": "IVF-Flat",
                "param_at_95": best_r["search_param_val"],
                "operating_recall": best_r["recall_at_10"],
                "qps_at_param": best_r["qps"],
                "latency_ms": best_r["latency_ms"],
                "qps_at_95": best_r["qps"],
                "build_time_s": ivf_idx.build_time,
                "index_size_mb": ivf_idx.get_size_bytes() / (1024**2),
            })

    return all_runs, summary_records


def print_part4_diagnostics_table(datasets: Dict[float, Any]) -> None:
    """Prints a clear diagnostics table summarizing geometry, cluster sizes, and GT verification."""
    print("\n" + "=" * 125)
    print("  PART 4: CLUSTER GEOMETRY DIAGNOSTICS & REPRODUCIBILITY SANITY CHECKS")
    print("=" * 125)
    print(
        f"{'Separation (s)':<16} | {'Inter-Dist (mean)':<18} | {'Within-Spread':<14} | {'Rel Ratio':<10} | {'Cluster Sizes':<18} | {'GT Sanity Check':<18} | {'GT Max Diff':<12}"
    )
    print("-" * 125)
    for s, ds in datasets.items():
        diag = ds.diagnostics
        if diag is None:
            continue
        ctrl_mark = " (CONTROL)" if s == PART4_CONTROL_SEPARATION else ""
        s_str = f"s = {s}{ctrl_mark}"
        inter_str = f"{diag.mean_inter_cluster_dist:.2f} [{diag.min_inter_cluster_dist:.1f}-{diag.max_inter_cluster_dist:.1f}]"
        within_str = f"{diag.mean_within_cluster_spread:.2f}±{diag.std_within_cluster_spread:.2f}"
        sizes_str = f"{diag.cluster_size_min}-{diag.cluster_size_max} (μ={diag.cluster_size_mean:.0f})"
        gt_status = "PASSED (0 errors)" if diag.gt_sanity_check_passed else "FAILED"
        print(
            f"{s_str:<16} | {inter_str:<18} | {within_str:<14} | {diag.relative_separation_ratio:<10.2f} | {sizes_str:<18} | {gt_status:<18} | {diag.gt_max_discrepancy:<12.2e}"
        )
    print("=" * 125 + "\n")


def print_part4_summary_table(summary_records: List[Dict[str, Any]]) -> None:
    """Prints a clear markdown summary table comparing search effort across separation settings."""
    print("\n" + "=" * 105)
    print("  PART 4: HARD GEOMETRY EXPERIMENT SUMMARY — SEARCH EFFORT & PERFORMANCE AT RECALL ≥ 0.95")
    print("=" * 105)
    print(
        f"{'Separation (s)':<16} | {'Index Type':<12} | {'Search Param @ 0.95':<22} | {'Recall':<8} | {'QPS @ Param':<12} | {'Latency (ms)':<12}"
    )
    print("-" * 105)
    for row in summary_records:
        s_val = row["separation"]
        ctrl_mark = " (CONTROL)" if s_val == PART4_CONTROL_SEPARATION else ""
        s_str = f"s = {s_val}{ctrl_mark}"

        if row["index_type"] == "HNSW":
            param_str = f"efSearch = {row['param_at_95']}"
        elif row["index_type"] == "IVF-Flat":
            param_str = f"nprobe = {row['param_at_95']}"
        else:
            param_str = "Brute-force (exact)"

        qps_val = row.get("qps_at_param", row.get("qps_at_95", 0.0))
        print(
            f"{s_str:<16} | {row['index_type']:<12} | {param_str:<22} | {row['operating_recall']:<8.4f} | {qps_val:<12.1f} | {row['latency_ms']:<12.2f}"
        )
    print("=" * 105 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Part 4: Hard Geometry (Cluster Separation) Benchmark CLI")
    parser.add_argument(
        "--separations",
        nargs="+",
        type=float,
        default=PART4_SEPARATIONS,
        help="Cluster separation multipliers to evaluate (default: 0.5 1.0 2.0 4.0)",
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
        help="Quick mode: 10K vectors, reduced sweeps for fast verification",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force regeneration of synthetic data and ground truth even if cached",
    )
    args = parser.parse_args()

    if args.quick:
        n_vectors = QUICK_PART4_N_VECTORS
        n_queries = QUICK_PART4_N_QUERIES
        separations = QUICK_PART4_SEPARATIONS
        ivf_nlist = QUICK_PART4_IVF_NLIST
        hnsw_ef_search = QUICK_PART4_HNSW_EF_SEARCH
        ivf_nprobe = QUICK_PART4_IVF_NPROBE
    else:
        n_vectors = PART4_N_VECTORS
        n_queries = PART4_N_QUERIES
        separations = args.separations
        ivf_nlist = PART4_IVF_NLIST
        hnsw_ef_search = PART4_HNSW_EF_SEARCH_LIST
        ivf_nprobe = PART4_IVF_NPROBE_LIST

    all_runs, summary_records = run_hard_geometry_benchmark(
        separations=separations,
        n_vectors=n_vectors,
        n_queries=n_queries,
        dim=PART4_DIM,
        n_clusters=PART4_N_CLUSTERS,
        hnsw_m=PART4_HNSW_M,
        hnsw_ef_construction=PART4_HNSW_EF_CONSTRUCTION,
        hnsw_ef_search_list=hnsw_ef_search,
        ivf_nlist=ivf_nlist,
        ivf_nprobe_list=ivf_nprobe,
        num_runs=args.num_runs,
        seed=PART4_SEED,
        force_recompute=args.force_recompute,
    )

    print_part4_summary_table(summary_records)

    print("Generating Part 4 Hard Geometry Figures...")
    plot_part4_hard_geometry(
        all_runs=all_runs,
        summary_records=summary_records,
        output_dir=PART4_FIGURES_DIR,
    )
    print("Part 4 Benchmark and Figures successfully completed!")


if __name__ == "__main__":
    main()
