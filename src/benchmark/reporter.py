"""Reporting utilities: results.csv logging and terminal summary tables."""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from src.config import RESULTS_CSV, RESULTS_DIR

CSV_FIELDNAMES = [
    "dataset",
    "index_type",
    "threads",
    "query_mode",
    "build_params",
    "search_params",
    "recall_at_10",
    "qps",
    "latency_ms",
    "build_time_s",
    "index_size_bytes",
    "bytes_per_vector",
]


def log_result(
    row: Dict[str, Any],
    csv_path: Union[str, Path] = RESULTS_CSV,
) -> None:
    """
    Appends a benchmark result row to results.csv.
    Creates directory and writes headers if the file does not exist.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    # Ensure all fieldnames exist in row with defaults
    clean_row = {field: row.get(field, "") for field in CSV_FIELDNAMES}

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(clean_row)


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    """
    Prints a formatted summary table of benchmark results to stdout.
    """
    if not rows:
        print("No results to display.")
        return

    headers = [
        ("Dataset", "dataset", 10),
        ("Index", "index_type", 12),
        ("Threads", "threads", 8),
        ("Mode", "query_mode", 8),
        ("Recall@10", "recall_at_10", 11),
        ("QPS", "qps", 10),
        ("Latency (ms)", "latency_ms", 13),
        ("Build (s)", "build_time_s", 10),
        ("Size (MB)", "index_size_bytes", 10),
    ]

    header_line = " | ".join(f"{title:<{width}}" for title, _, width in headers)
    separator = "-+-".join("-" * width for _, _, width in headers)

    print("\n" + header_line)
    print(separator)

    for row in rows:
        formatted_vals = []
        for _, key, width in headers:
            val = row.get(key, "")
            if key == "recall_at_10" and isinstance(val, (int, float)):
                val_str = f"{val:.4f}"
            elif key == "qps" and isinstance(val, (int, float)):
                val_str = f"{val:.2f}"
            elif key == "latency_ms" and isinstance(val, (int, float)):
                val_str = f"{val:.2f}"
            elif key == "build_time_s" and isinstance(val, (int, float)):
                val_str = f"{val:.2f}"
            elif key == "index_size_bytes" and isinstance(val, (int, float)):
                val_str = f"{val / (1024 * 1024):.1f}"
            else:
                val_str = str(val)
            formatted_vals.append(f"{val_str:<{width}}")
        print(" | ".join(formatted_vals))
    print()
