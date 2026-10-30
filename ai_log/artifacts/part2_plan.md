# Implementation Plan: Part 2 Benchmarking Three Indexes (HNSW, IVF-Flat, IVF-PQ)

This plan strictly covers **Part 2: Benchmark three indexes** of `Assignment 3 Benchmarking Approximate Nearest Neighbour Search.md`.

---

## Goal Description

Benchmark three foundational approximate nearest neighbour (ANN) index families—**HNSW**, **IVF-Flat**, and **IVF-PQ**—across both **SIFT-128** and **GloVe-100** datasets using FAISS:
1. Implement extensible index wrappers inheriting from `BaseIndex`:
   - `HNSWIndex` (`faiss.IndexHNSWFlat`)
   - `IVFFlatIndex` (`faiss.IndexIVFFlat`)
   - `IVFPQIndex` (`faiss.IndexIVFPQ`)
2. Explore multiple build settings per index (varying graph degree $M$ for HNSW, cluster count `nlist` for IVF-Flat, and coarse partition `nlist` + sub-vector count $M_{PQ}$ for IVF-PQ).
3. Sweep search-time parameters (`efSearch` for HNSW, `nprobe` for IVF) so recall ranges from low (~0.2–0.5) up to $\ge 0.95$.
4. Measure search speed in queries per second (QPS), one query at a time on a single thread (`faiss.omp_set_num_threads(1)`), taking the median of multiple repetitions.
5. Generate the required deliverables:
   - **Individual Parameter Sweep Plots**: Save detailed plots for each index parameter sweep into `results/figures/sweeps/` (showing separate curves for each build configuration across search parameter sweeps).
   - **Overall Pareto Trade-Off Plots**: Save consolidated Recall@10 vs. QPS trade-off plots per dataset in `results/figures/` showing the best Pareto frontier for each index and the exact baseline point.
   - **Performance Tables**: Summary tables for QPS at recall 0.90 and 0.95, and build time / index size / bytes-per-vector at recall 0.95.
   - **Architectural Analysis**: Mechanism explaining why IVF-PQ exhibits a recall ceiling.
   - **Logging & Reporting**: Log all benchmark runs to `results/results.csv` and document full observations and figures in `README.md`.

---

## User Review Required

> [!IMPORTANT]
> **Parameter Sweep Strategy & Grid Design**:
> The specification requires: *"For each index, try a few build settings (for example $M$ for HNSW, `nlist` for IVF) and, for each, sweep the search-time parameter (`efSearch` or `nprobe`) so that recall ranges from low up to about 0.95 or higher."*
> 
> We have designed a systematic, balanced parameter grid below. Please review and confirm the proposed parameter sweeps:
> 
> 1. **HNSW (`IndexHNSWFlat`)**:
>    - **Build Settings ($M$)**: $M \in \{8, 16, 32\}$ (lightweight, standard, dense connectivity).
>    - **Construction Setting**: `efConstruction = 64` (sufficient graph quality while keeping build time under ~30s on 16 threads).
>    - **Search Sweep (`efSearch`)**: `efSearch` $\in [8, 16, 32, 64, 128, 256, 512]$.
> 
> 2. **IVF-Flat (`IndexIVFFlat`)**:
>    - **Build Settings (`nlist`)**: `nlist` $\in \{1024, 2048, 4096\}$ (coarse, balanced, and fine clustering for $N \approx 10^6$ vectors).
>    - **Search Sweep (`nprobe`)**: `nprobe` $\in [1, 2, 4, 8, 16, 32, 64, 128, 256]$.
> 
> 3. **IVF-PQ (`IndexIVFPQ`)**:
>    - **Build Settings**:
>      - `nlist = 1024` (or 2048).
>      - **SIFT-128 ($d=128$)**: $M_{PQ} \in \{16, 32, 64\}$ sub-vectors ($8\times$, $4\times$, $2\times$ vector compression).
>      - **GloVe-100 ($d=100$)**: $M_{PQ} \in \{10, 20, 25, 50\}$ sub-vectors ($10\times$, $5\times$, $4\times$, $2\times$ compression).
>      - `nbits = 8` (256 centroids per sub-space).
>    - **Search Sweep (`nprobe`)**: `nprobe` $\in [1, 2, 4, 8, 16, 32, 64, 128, 256]$.

> [!NOTE]
> **Build-Time vs. Query-Time Threading**:
> In accordance with the ground rules:
> - **Search Timing**: Strictly executed **one query at a time on a single thread** (`faiss.omp_set_num_threads(1)`), taking the median across 3 runs.
> - **Index Building**: Training and adding 1M vectors uses multi-threading (`faiss.omp_set_num_threads(16)`) to accelerate index construction (e.g., reducing build time from ~3 minutes to ~10–20 seconds per index). The build time is accurately recorded and logged.

> [!NOTE]
> **Evaluation Query Split**:
> Per the Data instructions:
> - Parameter tuning / validation is conducted on queries `0:5000` (`test_tune`).
> - Final reported curves, QPS tables, and figures are evaluated on queries `5000:10000` (`test_eval`).

---

## Open Questions

None currently blocking. Our preliminary tests confirmed that FAISS single-query search on ANN indexes achieves 5,000–10,000 QPS, so evaluating 5,000 queries per sweep point completes in under 1 second per point.

---

## Proposed Changes

```
ire_a3/
├── src/
│   ├── config.py                   # Centralized parameter sweep grids and plot paths
│   ├── index/
│   │   ├── hnsw.py                 # [NEW] HNSWIndex implementation
│   │   ├── ivf_flat.py             # [NEW] IVFFlatIndex implementation
│   │   └── ivf_pq.py               # [NEW] IVFPQIndex implementation
│   └── benchmark/
│       └── plotting.py             # [NEW] Pareto curve plotting & recall 0.90/0.95 tables
├── tests/
│   └── test_ann_indexes.py         # [NEW] Unit tests for HNSW, IVF-Flat, and IVF-PQ
├── scripts/
│   └── run_part2.py                # [NEW] CLI script to execute Part 2 benchmark sweeps
├── results/
│   ├── results.csv                     # All Part 2 runs appended
│   └── figures/
│       ├── sift_recall_vs_qps.png      # [NEW] SIFT Pareto trade-off plot
│       ├── glove_recall_vs_qps.png     # [NEW] GloVe Pareto trade-off plot
│       └── sweeps/                     # [NEW] Detailed individual sweep plots
│           ├── sift_hnsw_sweep.png
│           ├── sift_ivf_flat_sweep.png
│           ├── sift_ivf_pq_sweep.png
│           ├── glove_hnsw_sweep.png
│           ├── glove_ivf_flat_sweep.png
│           └── glove_ivf_pq_sweep.png
└── README.md                           # Full documentation of Part 2 observations and tables
```

---

### 1. Index Implementations (`src/index/`)

#### [NEW] [`src/index/hnsw.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/index/hnsw.py)
Implements `HNSWIndex(BaseIndex)` wrapping `faiss.IndexHNSWFlat`:
- Constructor accepts `dim`, `M`, `efConstruction`, `metric`.
- In `build(train_vectors)`:
  - Initializes `index = faiss.IndexHNSWFlat(dim, M)`.
  - Sets `index.hnsw.efConstruction = efConstruction`.
  - Records build time and vector count.
- In `search(queries, k=10, efSearch=32)`:
  - Dynamically sets `self.index.hnsw.efSearch = efSearch`.
  - Performs search and returns `(distances, indices)`.
- `get_size_bytes()`: returns serialized FAISS index size via `faiss.serialize_index`.

#### [NEW] [`src/index/ivf_flat.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/index/ivf_flat.py)
Implements `IVFFlatIndex(BaseIndex)` wrapping `faiss.IndexIVFFlat`:
- Constructor accepts `dim`, `nlist`, `metric`.
- In `build(train_vectors)`:
  - Uses `faiss.IndexFlatL2(dim)` as coarse quantizer.
  - Initializes `index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)`.
  - Trains centroids: `index.train(train_vectors)`.
  - Populates inverted lists: `index.add(train_vectors)`.
  - Records build time.
- In `search(queries, k=10, nprobe=1)`:
  - Dynamically sets `self.index.nprobe = nprobe`.
  - Performs search and returns `(distances, indices)`.
- `get_size_bytes()`: returns serialized byte size.

#### [NEW] [`src/index/ivf_pq.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/index/ivf_pq.py)
Implements `IVFPQIndex(BaseIndex)` wrapping `faiss.IndexIVFPQ`:
- Constructor accepts `dim`, `nlist`, `M_pq`, `nbits=8`, `metric`.
- In `build(train_vectors)`:
  - Coarse quantizer: `faiss.IndexFlatL2(dim)`.
  - Initializes `index = faiss.IndexIVFPQ(quantizer, dim, nlist, M_pq, nbits, faiss.METRIC_L2)`.
  - Trains coarse and sub-quantizers: `index.train(train_vectors)`.
  - Encodes vectors into PQ codes: `index.add(train_vectors)`.
  - Records build time.
- In `search(queries, k=10, nprobe=1)`:
  - Dynamically sets `self.index.nprobe = nprobe`.
  - Performs search and returns `(distances, indices)`.
- `get_size_bytes()`: returns serialized byte size (verifying substantial compression).

---

### 2. Plotting & Reporting Module (`src/benchmark/plotting.py`)

#### [NEW] [`src/benchmark/plotting.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/src/benchmark/plotting.py)
Generates publication-quality figures and summary tables:
1. **Individual Parameter Sweep Plots** (stored in `results/figures/sweeps/`):
   - For every index and dataset, plots the Recall@10 vs. QPS trade-off curves for each distinct build setting across its search parameter sweep:
     - `sift_hnsw_sweep.png` / `glove_hnsw_sweep.png`: curves for $M \in \{8, 16, 32\}$ across `efSearch` sweeps.
     - `sift_ivf_flat_sweep.png` / `glove_ivf_flat_sweep.png`: curves for `nlist` $\in \{1024, 2048, 4096\}$ across `nprobe` sweeps.
     - `sift_ivf_pq_sweep.png` / `glove_ivf_pq_sweep.png`: curves for $M_{PQ}$ settings across `nprobe` sweeps.
2. **Consolidated Pareto Frontier Trade-Off Plots** (stored in `results/figures/`):
   - Generates `results/figures/sift_recall_vs_qps.png` and `results/figures/glove_recall_vs_qps.png`.
   - Plots the non-dominated Pareto frontier for HNSW, IVF-Flat, and IVF-PQ together on a single comparative plot.
   - Overlays the Exact-Flat baseline point as a benchmark reference.
   - X-axis: Recall@10 (linear scale, 0.0 to 1.0), Y-axis: Queries Per Second (QPS, log scale), with grid lines, clear markers, and legend.
3. **Recall 0.90 & 0.95 Throughput Table**:
   - Extracts the QPS achieved at recall 0.90 and recall 0.95 via linear interpolation on the Pareto frontier.
4. **Build Cost Table at Recall 0.95**:
   - For each index achieving recall 0.95, reports:
     - Configuration name
     - Build time (seconds)
     - Index size (MB)
     - Bytes per vector (bytes)

---

### 3. Automated Test Suite (`tests/test_ann_indexes.py`)

#### [NEW] [`tests/test_ann_indexes.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/tests/test_ann_indexes.py)
Automated pytest tests on synthetic data and small real samples:
- `test_hnsw_build_and_search`: builds HNSW on synthetic data, verifies `efSearch` sweep increases recall.
- `test_ivf_flat_build_and_search`: builds IVF-Flat, verifies `nprobe` sweep increases recall.
- `test_ivf_pq_build_and_search`: builds IVF-PQ, verifies substantial memory reduction and search behavior.
- `test_pareto_frontier_calculation`: verifies Pareto frontier logic on known coordinate sets.

---

### 4. CLI Runner (`scripts/run_part2.py`)

#### [NEW] [`scripts/run_part2.py`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/scripts/run_part2.py)
Orchestrates the complete Part 2 benchmarking pipeline:
- Supports CLI arguments:
  - `--dataset [sift|glove|all]`
  - `--quick` (runs on 100K train vectors and smaller parameter subset in under 5 minutes)
  - `--num-runs` (repetition count for median calculation, default 3)
- Execution Flow:
  1. Loads dataset and separates queries into tuning and evaluation splits.
  2. Builds HNSW indexes across $M \in \{8, 16, 32\}$ and sweeps `efSearch`.
  3. Builds IVF-Flat indexes across `nlist` $\in \{1024, 2048, 4096\}$ and sweeps `nprobe`.
  4. Builds IVF-PQ indexes across $M_{PQ}$ settings and sweeps `nprobe`.
  5. Measures single-threaded single-query throughput and recall@10.
  6. Appends every benchmark run to `results/results.csv`.
  7. Computes Pareto frontiers and saves plots to `results/figures/`.
  8. Generates the two required tables (QPS at recall 0.90/0.95, and build cost at recall 0.95).
  9. Prints the architectural explanation for why IVF-PQ fails to reach high recall without re-ranking.

---

## Technical Mechanism: Why IVF-PQ Fails to Reach High Recall

The specification asks for: *"a sentence or two on why any index fails to reach high recall, if one does."*

In our benchmark:
- **HNSW and IVF-Flat** store uncompressed full-precision float32 vectors. As `efSearch` $\to \infty$ or `nprobe` $\to \text{nlist}$, their distance calculations are exact, enabling them to reach recall $> 0.95$ and approaching $1.0$.
- **IVF-PQ** replaces each vector with $M_{PQ}$ low-dimensional cluster centroids (e.g., 8-bit codes). During search, distances are estimated via Asymmetric Distance Computation (ADC). This introduces irreversible **quantization noise** (distortion). Even if `nprobe = nlist` (searching all Voronoi cells), the fine-grained ranking between true nearest neighbours is degraded by quantization error, causing recall to hit an **asymptotic ceiling** below 0.95 unless $M_{PQ}$ is very large or exact candidate re-ranking is applied.

---

## Verification Plan

### Automated Tests
```bash
/home/shrawani/miniconda3/bin/pytest tests/test_ann_indexes.py -v
```

### Full Part 2 Execution
```bash
/home/shrawani/miniconda3/bin/python scripts/run_part2.py --dataset all
```

### Fast Laptop Verification (`--quick`)
```bash
/home/shrawani/miniconda3/bin/python scripts/run_part2.py --quick
```

### Deliverables Checklist
- [ ] `results/results.csv` contains all benchmark sweep rows.
- [ ] `results/figures/sweeps/` contains individual parameter sweep plots for every index on SIFT and GloVe.
- [ ] `results/figures/sift_recall_vs_qps.png` exists and shows consolidated Pareto curves for HNSW, IVF-Flat, IVF-PQ, and Exact-Flat baseline.
- [ ] `results/figures/glove_recall_vs_qps.png` exists and shows consolidated Pareto curves for HNSW, IVF-Flat, IVF-PQ, and Exact-Flat baseline.
- [ ] QPS at recall 0.90 and 0.95 table generated.
- [ ] Build time, index size, and bytes per vector table at recall 0.95 generated.
- [ ] Full observations, sweep figures, and tables reported in [`README.md`](file:///home/shrawani/Desktop/sem5/Information%20Retrieval%20and%20Extraction/ire_a3/README.md).
