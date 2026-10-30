"""Plotting and Pareto trade-off curve generation for ANN benchmarking."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


def compute_pareto_frontier(
    points: List[Dict[str, Any]],
    recall_key: str = "recall_at_10",
    qps_key: str = "qps",
) -> List[Dict[str, Any]]:
    """
    Computes the non-dominated Pareto frontier (maximizing both recall and QPS).

    Args:
        points: List of benchmark result dictionaries.

    Returns:
        Sorted list of non-dominated Pareto points (recall ascending, QPS descending).
    """
    if not points:
        return []

    # Filter valid points
    valid = [p for p in points if p.get(recall_key) is not None and p.get(qps_key) is not None]
    if not valid:
        return []

    # Sort primarily by recall ascending, secondarily by QPS descending
    sorted_pts = sorted(valid, key=lambda x: (x[recall_key], -x[qps_key]))

    pareto: List[Dict[str, Any]] = []
    # Work backwards from highest recall: track maximum QPS seen so far
    max_qps_so_far = -1.0
    for p in reversed(sorted_pts):
        if p[qps_key] > max_qps_so_far:
            pareto.append(p)
            max_qps_so_far = p[qps_key]

    pareto.reverse()
    return pareto


def interpolate_qps_at_recall(
    pareto_points: List[Dict[str, Any]],
    target_recall: float,
    recall_key: str = "recall_at_10",
    qps_key: str = "qps",
) -> Optional[float]:
    """
    Interpolates QPS achieved on the Pareto curve at a target recall value (e.g. 0.90 or 0.95).
    Returns None if the index never reaches the target recall.
    """
    if not pareto_points:
        return None

    recalls = [p[recall_key] for p in pareto_points]
    qps_vals = [p[qps_key] for p in pareto_points]

    if max(recalls) < target_recall:
        return None  # Index never reached this recall

    if min(recalls) >= target_recall:
        return float(qps_vals[0])

    for i in range(len(recalls) - 1):
        r1, r2 = recalls[i], recalls[i + 1]
        q1, q2 = qps_vals[i], qps_vals[i + 1]
        if r1 <= target_recall <= r2:
            if r1 == r2:
                return float(max(q1, q2))
            alpha = (target_recall - r1) / (r2 - r1)
            # Log-linear interpolation on QPS
            log_q = np.log(q1) + alpha * (np.log(q2) - np.log(q1))
            return float(np.exp(log_q))

    return None


def plot_individual_sweep(
    dataset_name: str,
    index_type: str,
    results: List[Dict[str, Any]],
    output_path: Path,
    build_param_key: str = "build_params",
) -> None:
    """
    Plots Recall@10 vs. QPS for each build setting across search parameter sweeps.
    Saves plot to results/figures/sweeps/.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group results by build_params
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        b_param = r.get(build_param_key, "Default")
        groups.setdefault(b_param, []).append(r)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, (b_param, pts) in enumerate(groups.items()):
        pts_sorted = sorted(pts, key=lambda x: x["recall_at_10"])
        recalls = [p["recall_at_10"] for p in pts_sorted]
        qps_vals = [p["qps"] for p in pts_sorted]
        marker = markers[i % len(markers)]
        ax.plot(
            recalls,
            qps_vals,
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=f"Build: {b_param}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10", fontsize=12, fontweight="bold")
    ax.set_ylabel("Queries Per Second (QPS, log scale)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{dataset_name.upper()} — {index_type} Parameter Sweeps",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"  [Figure] Saved individual sweep plot: {output_path}")


def plot_overall_pareto(
    dataset_name: str,
    pareto_by_index: Dict[str, List[Dict[str, Any]]],
    exact_baseline_qps: float,
    output_path: Path,
) -> None:
    """
    Plots the consolidated Recall@10 vs. QPS trade-off curves for HNSW, IVF-Flat,
    IVF-PQ, and the Exact-Flat baseline. Saves to results/figures/.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

    colors = {
        "HNSW": "#1f77b4",       # Blue
        "IVF-Flat": "#2ca02c",   # Green
        "IVF-PQ": "#d62728",     # Red
    }
    markers = {
        "HNSW": "o",
        "IVF-Flat": "s",
        "IVF-PQ": "^",
    }

    for idx_name, pts in pareto_by_index.items():
        if not pts:
            continue
        recalls = [p["recall_at_10"] for p in pts]
        qps_vals = [p["qps"] for p in pts]
        color = colors.get(idx_name, "black")
        marker = markers.get(idx_name, "o")

        ax.plot(
            recalls,
            qps_vals,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            color=color,
            label=f"{idx_name} (Pareto)",
        )

    # Plot exact baseline reference point at recall 1.0
    ax.scatter(
        [1.0],
        [exact_baseline_qps],
        color="#9467bd",
        marker="*",
        s=200,
        zorder=5,
        label=f"Exact Baseline ({exact_baseline_qps:.1f} QPS)",
    )

    # Reference lines for recall 0.90 and 0.95
    ax.axvline(0.90, color="gray", linestyle=":", alpha=0.6, label="Recall 0.90")
    ax.axvline(0.95, color="black", linestyle="--", alpha=0.6, label="Recall 0.95")

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10", fontsize=12, fontweight="bold")
    ax.set_ylabel("Queries Per Second (QPS, log scale)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{dataset_name.upper()} — Recall@10 vs. QPS Trade-Off (Single Thread)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"  [Figure] Saved overall Pareto plot: {output_path}")


def plot_scaling_curves(
    scaling_records: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Generates scaling plots showing how QPS, Latency, Build Time, and Memory scale
    with database size N (100K, 300K, 1M) for Exact-Flat, HNSW, and IVF-Flat.

    Args:
        scaling_records: List of dicts, each with keys:
            ['dataset_size', 'index_type', 'qps', 'latency_ms', 'build_time_s', 'index_size_mb']
        output_dir: Directory to save the generated plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "Exact-Flat": "#9467bd",  # Purple
        "HNSW": "#1f77b4",        # Blue
        "IVF-Flat": "#2ca02c",    # Green
    }
    markers = {
        "Exact-Flat": "*",
        "HNSW": "o",
        "IVF-Flat": "s",
    }

    metrics_to_plot = [
        ("qps", "Queries Per Second (QPS) @ Recall 0.95", "scaling_qps_vs_size.png", True),
        ("latency_ms", "Single-Query Latency (ms) @ Recall 0.95", "scaling_latency_vs_size.png", True),
        ("build_time_s", "Index Build Time (s) @ Recall ≥ 0.95", "scaling_build_time_vs_size.png", True),
        ("index_size_mb", "Index Size (MB) @ Recall ≥ 0.95", "scaling_memory_vs_size.png", False),
    ]

    import pandas as pd
    df = pd.DataFrame(scaling_records)

    sizes = sorted(df["dataset_size"].unique())
    size_labels = [f"{int(s/1000)}K" if s < 1_000_000 else "1M" for s in sizes]

    for metric_col, ylabel, filename, log_y in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

        for idx_type in ["Exact-Flat", "HNSW", "IVF-Flat"]:
            sub = df[df["index_type"] == idx_type].sort_values("dataset_size")
            if sub.empty:
                continue

            x_vals = sub["dataset_size"].tolist()
            y_vals = sub[metric_col].tolist()
            c = colors.get(idx_type, "black")
            m = markers.get(idx_type, "o")

            ax.plot(
                x_vals,
                y_vals,
                marker=m,
                color=c,
                linewidth=2.2,
                markersize=8,
                label=idx_type,
            )

        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.set_xticks(sizes)
        ax.set_xticklabels(size_labels, fontsize=11, fontweight="bold")
        ax.set_xlabel("Database Size (Vectors)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(f"SIFT Scaling Analysis: {ylabel.split('(')[0].strip()}", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=11)

        plt.tight_layout()
        save_path = output_dir / filename
        plt.savefig(save_path)
        plt.close(fig)
        print(f"  [Figure] Saved scaling plot: {save_path}")


def plot_part4_hard_geometry(
    all_runs: List[Dict[str, Any]],
    summary_records: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Generates the publication-quality 4-panel main figure and standalone figures
    for Part 4 (Hard Geometry / Cluster Separation experiment).

    Panels:
    - (A) IVF-Flat Recall@10 vs. QPS trade-off across 4 separation settings
    - (B) HNSW Recall@10 vs. QPS trade-off across 4 separation settings
    - (C) Search effort (nprobe / efSearch) required to achieve Recall@10 >= 0.95
    - (D) QPS achieved at Recall@10 >= 0.95 vs. Cluster Separation Multiplier
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sep_colors = {
        0.5: "#d62728",  # Red (Heavy Overlap)
        1.0: "#ff7f0e",  # Orange (Control)
        2.0: "#2ca02c",  # Green (Separated)
        4.0: "#1f77b4",  # Blue (Isolated)
    }
    sep_markers = {
        0.5: "x",
        1.0: "o",
        2.0: "^",
        4.0: "s",
    }
    sep_labels = {
        0.5: "s = 0.5 (Heavy Overlap)",
        1.0: "s = 1.0 (Control)",
        2.0: "s = 2.0 (Separated)",
        4.0: "s = 4.0 (Isolated)",
    }

    import pandas as pd
    df_runs = pd.DataFrame(all_runs)
    df_sum = pd.DataFrame(summary_records)

    separations = sorted(df_sum["separation"].unique())

    # Create 2x2 Master Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)

    # -------------------------------------------------------------
    # Panel A: IVF-Flat Recall vs QPS Trade-off
    # -------------------------------------------------------------
    ax_ivf = axes[0, 0]
    for s in separations:
        sub = df_runs[(df_runs["index_type"] == "IVF-Flat") & (df_runs["separation"] == s)]
        if sub.empty:
            continue
        p_pts = compute_pareto_frontier(sub.to_dict("records"))
        recalls = [p["recall_at_10"] for p in p_pts]
        qps_vals = [p["qps"] for p in p_pts]
        c = sep_colors.get(s, "black")
        m = sep_markers.get(s, "o")
        lbl = sep_labels.get(s, f"s = {s}")
        ax_ivf.plot(recalls, qps_vals, marker=m, color=c, linewidth=2.0, markersize=7, label=lbl)

    ax_ivf.axvline(0.95, color="black", linestyle="--", alpha=0.6, label="Recall 0.95")
    ax_ivf.set_yscale("log")
    ax_ivf.set_xlabel("Recall@10", fontsize=11, fontweight="bold")
    ax_ivf.set_ylabel("Queries Per Second (QPS, log scale)", fontsize=11, fontweight="bold")
    ax_ivf.set_title("(A) IVF-Flat (nlist=512): Recall vs. QPS", fontsize=12, fontweight="bold")
    ax_ivf.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_ivf.legend(fontsize=9, loc="lower left")

    # -------------------------------------------------------------
    # Panel B: HNSW Recall vs QPS Trade-off
    # -------------------------------------------------------------
    ax_hnsw = axes[0, 1]
    for s in separations:
        sub = df_runs[(df_runs["index_type"] == "HNSW") & (df_runs["separation"] == s)]
        if sub.empty:
            continue
        p_pts = compute_pareto_frontier(sub.to_dict("records"))
        recalls = [p["recall_at_10"] for p in p_pts]
        qps_vals = [p["qps"] for p in p_pts]
        c = sep_colors.get(s, "black")
        m = sep_markers.get(s, "o")
        lbl = sep_labels.get(s, f"s = {s}")
        ax_hnsw.plot(recalls, qps_vals, marker=m, color=c, linewidth=2.0, markersize=7, label=lbl)

    ax_hnsw.axvline(0.95, color="black", linestyle="--", alpha=0.6, label="Recall 0.95")
    ax_hnsw.set_yscale("log")
    ax_hnsw.set_xlabel("Recall@10", fontsize=11, fontweight="bold")
    ax_hnsw.set_ylabel("Queries Per Second (QPS, log scale)", fontsize=11, fontweight="bold")
    ax_hnsw.set_title("(B) HNSW (M=16, efConst=64): Recall vs. QPS", fontsize=12, fontweight="bold")
    ax_hnsw.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_hnsw.legend(fontsize=9, loc="lower left")

    # -------------------------------------------------------------
    # Panel C: Search Effort Required to Reach Recall >= 0.95
    # -------------------------------------------------------------
    ax_effort = axes[1, 0]
    ivf_sum = df_sum[df_sum["index_type"] == "IVF-Flat"].sort_values("separation")
    hnsw_sum = df_sum[df_sum["index_type"] == "HNSW"].sort_values("separation")

    if not ivf_sum.empty:
        ax_effort.plot(
            ivf_sum["separation"],
            ivf_sum["param_at_95"],
            marker="s",
            color="#2ca02c",
            linewidth=2.2,
            markersize=8,
            label="IVF-Flat: nprobe required",
        )
    if not hnsw_sum.empty:
        ax_effort.plot(
            hnsw_sum["separation"],
            hnsw_sum["param_at_95"],
            marker="o",
            color="#1f77b4",
            linewidth=2.2,
            markersize=8,
            label="HNSW: efSearch required",
        )

    ax_effort.set_xscale("log", base=2)
    ax_effort.set_yscale("log", base=2)
    ax_effort.set_xticks(separations)
    ax_effort.set_xticklabels([str(s) for s in separations], fontsize=10, fontweight="bold")
    ax_effort.set_xlabel("Cluster Separation Multiplier (log2 scale)", fontsize=11, fontweight="bold")
    ax_effort.set_ylabel("Search Effort Value (log2 scale)", fontsize=11, fontweight="bold")
    ax_effort.set_title("(C) Search Effort Required for Recall@10 ≥ 0.95", fontsize=12, fontweight="bold")
    ax_effort.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_effort.legend(fontsize=10)

    # -------------------------------------------------------------
    # Panel D: QPS at Recall >= 0.95 vs Separation
    # -------------------------------------------------------------
    ax_qps = axes[1, 1]
    exact_sum = df_sum[df_sum["index_type"] == "Exact-Flat"].sort_values("separation")
    if not exact_sum.empty:
        ax_qps.plot(
            exact_sum["separation"],
            exact_sum["qps_at_95"],
            marker="*",
            color="#9467bd",
            linewidth=2.0,
            markersize=10,
            label="Exact-Flat (Baseline)",
        )
    if not hnsw_sum.empty:
        hnsw_qps = hnsw_sum["qps_at_param"] if "qps_at_param" in hnsw_sum.columns else hnsw_sum["qps_at_95"]
        ax_qps.plot(
            hnsw_sum["separation"],
            hnsw_qps,
            marker="o",
            color="#1f77b4",
            linewidth=2.2,
            markersize=8,
            label="HNSW @ Min Param",
        )
    if not ivf_sum.empty:
        ivf_qps = ivf_sum["qps_at_param"] if "qps_at_param" in ivf_sum.columns else ivf_sum["qps_at_95"]
        ax_qps.plot(
            ivf_sum["separation"],
            ivf_qps,
            marker="s",
            color="#2ca02c",
            linewidth=2.2,
            markersize=8,
            label="IVF-Flat @ Min Param",
        )

    ax_qps.set_xscale("log", base=2)
    ax_qps.set_yscale("log")
    ax_qps.set_xticks(separations)
    ax_qps.set_xticklabels([str(s) for s in separations], fontsize=10, fontweight="bold")
    ax_qps.set_xlabel("Cluster Separation Multiplier (log2 scale)", fontsize=11, fontweight="bold")
    ax_qps.set_ylabel("Throughput (QPS @ Recall 0.95, log scale)", fontsize=11, fontweight="bold")
    ax_qps.set_title("(D) Throughput at Recall@10 ≥ 0.95 vs. Separation", fontsize=12, fontweight="bold")
    ax_qps.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_qps.legend(fontsize=10)

    plt.suptitle("Part 4: Hard Geometry — Effect of Cluster Separation on ANN Search Effort", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    master_path = output_dir / "part4_hard_geometry_main.png"
    plt.savefig(master_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure] Saved Part 4 Main 4-Panel Figure: {master_path}")

