# Implementation Plan: Part 1 Exact Baseline & Modular Codebase Architecture

This plan covers the requirements from **Overview**, **Data**, and **Part 1: Exact baseline** of `Assignment 3 Benchmarking Approximate Nearest Neighbour Search.md`, incorporating user feedback to establish a clean, modular architecture from the start.

---

## Goal Description

1. Establish a modular, maintainable, and extensible codebase architecture that will support Part 1 through Part 5 without refactoring.
2. Implement robust data loading, vector normalization, and dataset splitting (5,000 tuning / 5,000 test) for SIFT-128 and GloVe-100.
3. Implement a mathematically sound `recall@10` evaluation metric accounting for distance ties at the 10th neighbor boundary, complete with a comprehensive `pytest` test suite.
4. Implement FAISS exact baseline search under a common `BaseIndex` abstraction.
5. Verify 100% agreement between the FAISS exact baseline and stored dataset ground-truth neighbors.
6. Benchmark single-query vs. batch-query execution speeds under controlled thread settings and document the architectural reasons for their performance divergence (GEMM cache reuse vs. GEMV memory bandwidth bottleneck).

---

## User Review Required

> [!NOTE]
> **Git Staging**: `.gitignore` has already been updated with `glove-100-angular.hdf5`. We will unstage the 485 MB dataset (`git restore --staged glove-100-angular.hdf5`) as requested before any commit so that dataset binaries are never tracked in git history.

---

## Modular Codebase Architecture

To keep the project clean, scalable, and modular from the start, we structure the codebase as follows:

```
ire_a3/
├── src/
│   ├── __init__.py
│   ├── config.py            # Dataset paths, vector dimensions, splits, tolerances, constants
│   ├── data.py              # HDF5 loader, L2 normalization for GloVe, query splits (tune/eval)
│   ├── metrics.py           # recall@k with tie handling, candidate set intersection, thresholding
│   ├── index/
│   │   ├── __init__.py
│   │   ├── base.py          # BaseIndex abstract class (build, search, size, params)
│   │   └── exact.py         # ExactFlatIndex (FAISS IndexFlatL2 implementation)
│   └── benchmark/
│       ├── __init__.py
│       ├── timing.py        # Micro-benchmarking utilities (single query, batch, median of runs)
│       └── reporter.py      # Formatting tables, CSV logging conforming to results/results.csv
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py      # Unit tests for recall@k, distance ties, edge cases
│   └── test_exact.py        # Integration test for exact baseline & ground truth agreement
├── scripts/
│   └── run_part1.py         # Standalone CLI script to run Part 1 verification & benchmarks
├── results/
│   └── results.csv          # Central benchmark log (deliverable)
├── README.md
└── requirements.txt
```

This ensures that when implementing **Part 2** (HNSW, IVF-Flat, IVF-PQ) or **Parts 3 & 4** (scaling, difficulty, synthetic generator), new indexes simply inherit from `src/index/base.py` and reuse `src/data.py`, `src/metrics.py`, and `src/benchmark/timing.py`.

---

## Proposed Changes

### 1. Repository Hygiene
- Run `git restore --staged glove-100-angular.hdf5` to ensure large dataset files are not tracked.
- Add `results/` folder and placeholder for `results.csv`.

---

### 2. Core Package (`src/`)

#### [NEW] [`src/config.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/config.py)
Centralizes configurations and hyperparameters:
- Paths: `SIFT_PATH = "sift-128-euclidean.hdf5"`, `GLOVE_PATH = "glove-100-angular.hdf5"`.
- Dataset metadata: dimensions (128 for SIFT, 100 for GloVe), distance metrics (`euclidean`, `angular`).
- Splits: `NUM_TUNING_QUERIES = 5000`, `NUM_EVAL_QUERIES = 5000`.
- Metric tolerances: default epsilon for floating-point tie checks (`1e-4`).

#### [NEW] [`src/data.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/data.py)
Encapsulates clean dataset loading and pre-processing:
- `DatasetSplits` dataclass: holds `train`, `test_tune`, `test_eval`, `gt_neighbors_tune`, `gt_neighbors_eval`, `gt_distances_tune`, `gt_distances_eval`.
- Normalizes GloVe vectors with $L_2$ (`faiss.normalize_L2`), preserving original copies if needed.
- Provides optional subset loading (e.g. for `--quick` mode, loading 100K vectors instead of 1M).

#### [NEW] [`src/metrics.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/metrics.py)
Implements robust, tie-aware evaluation:
- `recall_at_k(retrieved_indices, gt_neighbors, gt_distances, k=10, eps=1e-4)`:
  - For each query $i$, gets threshold $d_k = \text{gt\_distances}[i, k-1]$.
  - Identifies all valid ground-truth items with distance $\le d_k + \epsilon$.
  - Computes $\min(1.0, \frac{|R_i \cap \text{ValidGT}_i|}{k})$.
  - Returns aggregate mean recall and per-query array.

#### [NEW] [`src/index/base.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/index/base.py)
Abstract base class `BaseIndex` with standard methods:
- `build(train_vectors: np.ndarray) -> None`
- `search(queries: np.ndarray, k: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]` (returns distances, indices)
- `get_index_size_bytes() -> int`
- `name: str`

#### [NEW] [`src/index/exact.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/index/exact.py)
`ExactFlatIndex(BaseIndex)`:
- Wraps `faiss.IndexFlatL2`.
- Implements single-query search and batch search.
- Computes index size in bytes.

#### [NEW] [`src/benchmark/timing.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/benchmark/timing.py)
High-precision benchmarking utilities:
- `time_single_queries(index, queries, k, num_runs=3, threads=1)`: measures per-query and total latency looping one query at a time; takes the median over repeated runs.
- `time_batch_query(index, queries, k, num_runs=3, threads=1)`: measures batch search passing all queries at once; takes the median over repeated runs.
- Calculates Queries Per Second (QPS), average latency (ms/query), and speedup ratio.

#### [NEW] [`src/benchmark/reporter.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/benchmark/reporter.py)
Handles logging to console (formatted tables) and appending rows to `results/results.csv` with standardized columns:
- `dataset, index_type, build_params, search_params, threads, query_mode, recall_at_10, qps, latency_ms, build_time_s, index_size_bytes`.

---

### 3. Unit & Integration Tests (`tests/`)

#### [NEW] [`tests/test_metrics.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/tests/test_metrics.py)
Automated unit tests with `pytest`:
1. `test_perfect_recall`: identical retrieved and ground truth $\to 1.0$.
2. `test_zero_recall`: disjoint sets $\to 0.0$.
3. `test_partial_recall`: 7 / 10 matches $\to 0.7$.
4. `test_tie_at_10th_boundary`: where $d_{10} = d_{11}$; returning 11th instead of 10th neighbor gives $1.0$ (while naive intersection gives $0.9$).
5. `test_worse_item_rejected`: returning an item with distance $> d_{10} + \epsilon$ yields $0.9$.
6. `test_duplicate_retrieved_protection`: duplicate IDs in retrieved do not get extra credit.
7. `test_multiple_queries`: correct vectorization and mean over multiple queries.

#### [NEW] [`tests/test_exact.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/tests/test_exact.py)
Integration tests:
1. SIFT-128 exact search matches ground truth with recall@10 = 1.0 on test queries.
2. GloVe-100 normalized exact search matches ground truth with recall@10 = 1.0 on test queries.

---

### 4. Runner & Deliverables (`scripts/run_part1.py`)

#### [NEW] [`scripts/run_part1.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/scripts/run_part1.py)
Command-line runner supporting:
- `--dataset [sift|glove|all]`
- `--verify-only` (quickly verify ground truth agreement)
- `--threads [1|16|all]`
- Executes Part 1, verifies exact ground truth agreement, benchmarks single vs. batch query timing, outputs formatted markdown/text tables, records results in `results/results.csv`, and prints the architectural explanation.

---

## Technical Mechanism Explanation

The performance difference between single-query and batch-query exact search arises from foundational computer architecture principles:

1. **Arithmetic Intensity & BLAS Operations (GEMV vs. GEMM)**:
   - In **single-query exact search**, querying 1 vector against $N$ vectors ($1 \times d$ against $N \times d$) reduces to matrix-vector multiplication (**BLAS Level 2 GEMV**). For every single query, the entire database ($10^6 \times 128 \times 4\text{ bytes} \approx 512\text{ MB}$) must be fetched across the memory bus from RAM into CPU caches. The arithmetic intensity is low ($\approx \mathcal{O}(1)$ FLOPs/byte).
   - In **batch-query exact search**, querying $B$ vectors against $N$ vectors ($B \times d$ against $N \times d$) becomes matrix-matrix multiplication (**BLAS Level 3 GEMM**). The database vectors reside in CPU cache and are reused across all $B$ queries. The arithmetic intensity scales with $B$, saturating SIMD floating-point units (AVX2/FMA) rather than waiting on memory bandwidth.
2. **Memory Bandwidth Bottleneck**:
   - Streaming $5,000$ queries individually demands transferring $\approx 5,000 \times 512\text{ MB} = 2.56\text{ TB}$ of data over the memory bus. At typical DDR4/DDR5 bandwidth (30–50 GB/s), this creates an inescapable physical wall. Batching eliminates redundant bus transfers.
3. **Overhead Amortization**:
   - Single-query search incurs Python-to-C++ transitions, OpenMP barrier dispatches, and result heap setup 5,000 separate times, while batch search pays these costs once for the entire batch.

---

## Verification Plan

### Automated Tests
1. **Pytest Unit Tests**:
   ```bash
   /home/shrawani/miniconda3/bin/pytest tests/test_metrics.py tests/test_exact.py -v
   ```
2. **Ground Truth Verification**:
   ```bash
   /home/shrawani/miniconda3/bin/python scripts/run_part1.py --verify-only
   ```
3. **Full Part 1 Benchmark Run**:
   ```bash
   /home/shrawani/miniconda3/bin/python scripts/run_part1.py --dataset all
   ```

### Manual Verification & Reporting
- Confirm `git status` shows `glove-100-angular.hdf5` is unstaged and untracked.
- Inspect `results/results.csv` to ensure consistent logging of recall@10 and QPS numbers.
- **Document all observations in [`README.md`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/README.md)**:
  - Dataset specifications, vector counts, dimensions, and split configuration.
  - Ground truth agreement results on SIFT-128 and normalized GloVe-100.
  - Tie analysis at the 10th neighbor boundary.
  - Exact search timing tables (single query vs. batch query, QPS, latency per query, speedup ratio).
  - Clear architectural explanation of the single-query vs. batch-query performance gap (GEMV memory bandwidth bottleneck vs. GEMM cache tiling).

