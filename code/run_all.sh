#!/bin/bash
# ==============================================================================
# CS4.406: Information Retrieval & Extraction - Assignment 3
# One-Command Reproduction Script
#
# Usage:
#   bash run_all.sh           # Full benchmark: regenerates all numbers, tables, and figures
#   bash run_all.sh --quick   # Fast verification (< 15 min): runs on subsets (100K vectors)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

QUICK_MODE=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=true
fi

START_TIME=$(date +%s)

echo "=============================================================================="
echo "  IRE ASSIGNMENT 3: BENCHMARKING APPROXIMATE NEAREST NEIGHBOUR SEARCH"
echo "  One-Command Reproduction Harness"
if [ "$QUICK_MODE" = true ]; then
    echo "  MODE: QUICK (--quick, target < 15 minutes)"
else
    echo "  MODE: FULL (Complete evaluation on SIFT-1M, GloVe-100, and Part 3/4)"
fi
echo "=============================================================================="
echo ""

# Ensure required directories exist
mkdir -p results/figures/sweeps
mkdir -p results/figures/part3
mkdir -p results/figures/part4
mkdir -p results/cache

# ------------------------------------------------------------------------------
# Part 1: Exact Baseline & Metric Verification
# ------------------------------------------------------------------------------
echo ">>> [1/4] Running Part 1: Exact Search Baseline & Agreement Verification..."
P1_START=$(date +%s)
if [ "$QUICK_MODE" = true ]; then
    python scripts/run_part1.py --quick
else
    python scripts/run_part1.py --dataset all --threads 1
fi
P1_END=$(date +%s)
echo ">>> Part 1 completed in $((P1_END - P1_START)) seconds."
echo ""

# ------------------------------------------------------------------------------
# Part 2: Benchmark Three Indexes (HNSW, IVF-Flat, IVF-PQ)
# ------------------------------------------------------------------------------
echo ">>> [2/4] Running Part 2: Three Index Benchmarks (HNSW, IVF-Flat, IVF-PQ)..."
P2_START=$(date +%s)
if [ "$QUICK_MODE" = true ]; then
    python scripts/run_part2.py --dataset all --quick
else
    python scripts/run_part2.py --dataset all --num-runs 2
fi
P2_END=$(date +%s)
echo ">>> Part 2 completed in $((P2_END - P2_START)) seconds."
echo ""

# ------------------------------------------------------------------------------
# Part 3: Scaling Experiment on Standard Data (SIFT 100K, 300K, 1M)
# ------------------------------------------------------------------------------
echo ">>> [3/4] Running Part 3: SIFT Scaling Experiment..."
P3_START=$(date +%s)
if [ "$QUICK_MODE" = true ]; then
    python scripts/run_part3.py --quick
else
    python scripts/run_part3.py --sizes 100000 300000 1000000 --num-runs 2 --rerun-1m
fi
P3_END=$(date +%s)
echo ">>> Part 3 completed in $((P3_END - P3_START)) seconds."
echo ""

# ------------------------------------------------------------------------------
# Part 4: Hard Geometry Experiment on Synthetic Clustered Data
# ------------------------------------------------------------------------------
echo ">>> [4/4] Running Part 4: Hard Geometry (Cluster Separation) Experiment..."
P4_START=$(date +%s)
if [ "$QUICK_MODE" = true ]; then
    python scripts/run_part4.py --quick
else
    python scripts/run_part4.py --separations 0.5 1.0 2.0 4.0 --num-runs 2 --force-recompute
fi
P4_END=$(date +%s)
echo ">>> Part 4 completed in $((P4_END - P4_START)) seconds."
echo ""

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
MINUTES=$(( TOTAL_TIME / 60 ))
SECONDS=$(( TOTAL_TIME % 60 ))

echo "=============================================================================="
echo "  REPRODUCTION COMPLETE!"
echo "  Total Execution Time: ${MINUTES}m ${SECONDS}s ($TOTAL_TIME seconds)"
echo "=============================================================================="
echo "Artifacts generated / verified:"
echo "  - Results Log: results/results.csv ($(wc -l < results/results.csv) rows)"
echo "  - Part 2 Figures: results/figures/sift_recall_vs_qps.png, results/figures/glove_recall_vs_qps.png"
echo "  - Part 2 Sweep Figures: results/figures/sweeps/"
echo "  - Part 3 Scaling Figures: results/figures/part3/"
echo "  - Part 4 Hard Geometry Figures: results/figures/part4/part4_hard_geometry_main.png"
echo "=============================================================================="
