# CS4.406: Information Retrieval & Extraction - Assignment 3
## Benchmarking Approximate Nearest Neighbour Search

This repository contains the complete benchmarking harness, empirical evaluations, reproducible scripts, and analysis report for Approximate Nearest Neighbour (ANN) search across **SIFT-128** (1M image descriptors, Euclidean) and **GloVe-100** (1.18M word vectors, angular/$L_2$).

---

## Directory Structure

```text
ire_a3/
├── code/                      # Main codebase and benchmarking harness
│   ├── src/                   # Modular implementation of indexes, metrics, data loaders, and plotting
│   │   ├── index/             # FAISS index wrappers (ExactFlat, HNSW, IVF-Flat, IVF-PQ)
│   │   ├── benchmark/         # Timing harness, Pareto frontier computation, and CSV logging
│   │   ├── config.py          # Centralized hyperparameter grids and dataset configurations
│   │   ├── data.py            # HDF5 loader, unit-L2 normalization, and query splitting
│   │   ├── data_generator.py  # Synthetic clustered data generator for Part 4
│   │   └── metrics.py         # Distance-threshold recall@10 with tie-handling
│   ├── scripts/               # Individual experiment runners for Parts 1 to 4
│   │   ├── run_part1.py       # Part 1: Exact search baseline and ground-truth agreement
│   │   ├── run_part2.py       # Part 2: HNSW, IVF-Flat, IVF-PQ sweeps and Pareto plotting
│   │   ├── run_part3.py       # Part 3: SIFT database scaling (100K, 300K, 1M)
│   │   └── run_part4.py       # Part 4: Hard geometry (cluster separation) experiment
│   ├── tests/                 # Unit and integration test suite (27 tests)
│   ├── results/               # Benchmark outputs (located directly inside code/)
│   │   ├── results.csv        # Comprehensive log of all benchmark runs
│   │   ├── figures/           # Generated Pareto trade-off curves and sweep plots
│   │   │   ├── sift_recall_vs_qps.png
│   │   │   ├── glove_recall_vs_qps.png
│   │   │   ├── sweeps/        # Individual build/search sweep curves
│   │   │   ├── part3/         # Database scaling curves (100K, 300K, 1M)
│   │   │   └── part4/         # Hard geometry 4-panel master figure
│   │   └── cache/             # Precomputed subset ground truth files
│   ├── hypothesis.md          # Pre-committed hypotheses for Parts 3 and 4
│   ├── requirements.txt       # Python package dependencies
│   ├── run_all.sh             # One-command full and quick reproduction script
│   └── README.md              # In-depth architectural analysis, benchmark tables, and discussion
├── ai_log/                    # AI assistance statement and conversation logs
│   └── ai_statement.md        # Declaration of AI tool usage per assignment guidelines
├── report.pdf                 # Final 5-page report with all figures and tables
├── report.tex                 # LaTeX source code for the report
├── sift-128-euclidean.hdf5    # SIFT dataset (1M vectors, 128-d)
└── glove-100-angular.hdf5     # GloVe dataset (1.18M vectors, 100-d)
```

---

## Benchmark Results Location

> **Important Note on `results/`**:  
> All benchmark results, figures, and execution logs are stored directly inside the **`code/`** directory under **`code/results/`**:
> - **`code/results/results.csv`**: Contains one row for every benchmark run behind each figure and table in the report.
> - **`code/results/figures/`**: Contains all generated Pareto trade-off curves, individual sweep plots, scaling curves, and Part 4 geometry diagnostics.

---

## How to Run & Reproduce

For complete running instructions, reproduction options, and test suite execution, please refer to the **[How to Reproduce](code/README.md#how-to-reproduce)** section at the end of **[`code/README.md`](code/README.md)**.

### Summary of Commands (Run from `code/`):

1. **Navigate to the code directory**:
   ```bash
   cd code
   ```

2. **Fast Verification Mode (< 15 minutes on a laptop)**:
   ```bash
   bash run_all.sh --quick
   ```

3. **Full Pipeline Reproduction**:
   ```bash
   bash run_all.sh
   ```

4. **Run Unit & Integration Test Suite**:
   ```bash
   pytest tests/ -v
   ```

5. **Individual Parts Execution**:
   For running individual experiment CLI scripts (`run_part1.py`, `run_part2.py`, `run_part3.py`, `run_part4.py`), see **Section "How to Reproduce" §4** in [`code/README.md`](code/README.md).
