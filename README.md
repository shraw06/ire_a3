# CS4.406: Information Retrieval & Extraction — Assignment 3
## Benchmarking Approximate Nearest Neighbour Search

### System & Environment Specs
- **Platform**: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- **CPU**: 12th Gen Intel(R) Core(TM) i7-12650H (10 cores: 6 P-cores, 4 E-cores, 16 threads)
- **RAM**: 16 GB DDR4/DDR5
- **NumPy**: 1.24.4
- **FAISS**: 1.8.0 (faiss-cpu)
- **h5py**: 3.14.0
- **Pytest**: 8.3.3

---

## Codebase Architecture

The codebase is built with a modular, extensible architecture designed to support Parts 1 through 5 seamlessly:

```
ire_a3/
├── src/
│   ├── config.py            # Centralized dataset paths, dimensions, split sizes, tolerances
│   ├── data.py              # HDF5 loader, L2 normalization for GloVe, query splits
│   ├── metrics.py           # recall@k with distance tie handling at k-th neighbor
│   ├── index/
│   │   ├── base.py          # BaseIndex abstract base class
│   │   ├── exact.py         # ExactFlatIndex (FAISS IndexFlatL2 implementation)
│   │   ├── hnsw.py          # HNSWIndex (FAISS IndexHNSWFlat implementation)
│   │   ├── ivf_flat.py      # IVFFlatIndex (FAISS IndexIVFFlat implementation)
│   │   └── ivf_pq.py        # IVFPQIndex (FAISS IndexIVFPQ implementation)
│   └── benchmark/
│       ├── timing.py        # Single-query and batch-query timing measurement
│       ├── reporter.py      # Standardized results.csv logging and summary formatting
│       └── plotting.py      # Pareto frontier computation, interpolation & figure generation
├── tests/
│   ├── test_metrics.py      # 9 unit tests for recall@k (ties, partial, duplicates, edge cases)
│   ├── test_exact.py        # 3 integration tests for exact search and dataset agreement
│   └── test_ann_indexes.py  # 5 integration tests for HNSW, IVF-Flat, IVF-PQ & Pareto logic
├── scripts/
│   ├── run_part1.py         # Standalone CLI runner for Part 1 baseline and benchmarks
│   └── run_part2.py         # Standalone CLI runner for Part 2 ANN benchmarks & plotting
├── results/
│   ├── results.csv          # Benchmark results log conforming to assignment deliverables
│   └── figures/             # Trade-off curves and individual sweep plots
│       ├── sift_recall_vs_qps.png
│       ├── glove_recall_vs_qps.png
│       └── sweeps/          # Individual parameter sweep plots for each index and dataset
├── requirements.txt
└── README.md
```

---

## Datasets & Preprocessing

Two standard benchmark datasets are evaluated:

| Dataset | Vectors to Index (Train) | Test Queries | Dimensions | Native Metric | Preprocessing | Stored Distances |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SIFT** (`sift-128-euclidean.hdf5`) | 1,000,000 | 10,000 | 128 | Euclidean ($L_2$) | None | Euclidean ($L_2 = \sqrt{\sum (u_i - v_i)^2}$) |
| **GloVe-100** (`glove-100-angular.hdf5`) | 1,183,514 | 10,000 | 100 | Angular / Cosine | Unit $L_2$ Normalization | $1 - \cos(\theta)$ |

### Query Split Policy
In accordance with assignment guidelines:
- **Queries `0` to `4,999` (first 5,000)**: Reserved for parameter tuning and validation.
- **Queries `5,000` to `9,999` (last 5,000)**: Evaluation split used for reporting final benchmark numbers.

### Distance Equivalence for GloVe
GloVe vectors are unit normalized ($\|u\|_2 = \|v\|_2 = 1$). In unit Euclidean space:
$$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2 \langle u, v \rangle = 2 - 2 \cos(\theta) = 2 (1 - \cos(\theta))$$
Because $2(1 - \cos(\theta))$ is strictly monotonic with respect to angular distance ($1 - \cos(\theta)$), FAISS `IndexFlatL2` computes the exact same nearest neighbour rankings as angular distance, with squared Euclidean distance $D_{FAISS} = 2 \times D_{stored}$.

---

## Part 1: Exact Baseline Observations

### 1. Distance-Threshold Recall@10 Metric (Tie-Aware)
In nearest neighbour benchmarking, queries frequently exhibit distance ties at the 10th neighbour boundary ($d_{10} = d_{11}$). 
- **Direct Metric Comparison**: In accordance with the user instructions and standard `ann-benchmarks` principles, the evaluation directly uses the distances returned by FAISS aligned to the dataset ground-truth metric:
  - **SIFT-128**: FAISS `IndexFlatL2` returns squared Euclidean distance ($L_2^2$). Aligned via $\text{dist} = \sqrt{\max(D_{\text{FAISS}}, 0)}$.
  - **GloVe-100**: On unit-normalized vectors, FAISS returns $\|u - v\|^2 = 2 (1 - \cos(\theta))$. Aligned via $\text{dist} = D_{\text{FAISS}} / 2.0$.
- **Distance Thresholding (No ID-set Fallback)**:
  $$\text{Recall@}k = \frac{1}{k} \sum_{j=1}^k \mathbb{I}\left[\text{dist}(q, p_j) \le d_k + \epsilon\right]$$
  where $d_k = \text{gt\_distances}[k - 1]$ and $\epsilon = 10^{-4}$.
  - Any retrieved candidate with aligned distance $\le d_k + \epsilon$ is recognized as a valid nearest neighbour, fully crediting alternative database vectors tied at the boundary without restriction to a pre-truncated ground-truth ID set.
  - Candidate IDs are deduplicated to ensure repeated items cannot artificially inflate recall.

### 2. Ground Truth Agreement Verification
Exact search was executed on the evaluation split (queries 5,000–10,000) using FAISS `IndexFlatL2` and compared against the dataset ground truth:

| Dataset | Evaluation Queries | Queries with Boundary Ties | Mean Recall@10 | Min Recall@10 | Agreement Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SIFT-128** | 5,000 | 63 (1.3%) | **1.000000** | 1.000000 | **100% Exact Match** |
| **GloVe-100** | 5,000 | 223 (4.5%) | **1.000000** | 1.000000 | **100% Exact Match** |

*Both datasets achieved 100% agreement (`recall@10 == 1.0`) across all queries.*

### 3. Timing Benchmark: Single-Query vs. Batch-Query
Timings were measured on a single thread (`faiss.omp_set_num_threads(1)`), taking the median across runs:

| Dataset | Index Type | Threads | Query Mode | Recall@10 | Throughput (QPS) | Latency (ms/query) | Build Time (s) | Index Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT-128** | Exact-Flat | 1 | **Batch** | **1.0000** | **135.89** | **7.36 ms** | 0.18 s | 488.3 MB |
| **SIFT-128** | Exact-Flat | 1 | **Single** | **1.0000** | **37.23** | **26.86 ms** | 0.18 s | 488.3 MB |
| **GloVe-100** | Exact-Flat | 1 | **Batch** | **1.0000** | **155.79** | **6.42 ms** | 0.17 s | 451.5 MB |
| **GloVe-100** | Exact-Flat | 1 | **Single** | **1.0000** | **42.36** | **23.61 ms** | 0.17 s | 451.5 MB |

#### Speedup Observations:
- **SIFT-128**: Batch search is **3.65x faster** than single-query search (135.89 QPS vs. 37.23 QPS).
- **GloVe-100**: Batch search is **3.68x faster** than single-query search (155.79 QPS vs. 42.36 QPS).

All results are logged in [`results/results.csv`](results/results.csv).

---

## Architectural Explanation: Why Single-Query and Batch Search Differ

The significant throughput disparity between batch search and single-query search stems from fundamental computer architecture and memory hierarchy principles:

1. **Arithmetic Intensity & BLAS Operations (GEMM vs. GEMV)**:
   - **Single-query exact search**: Comparing one query against $N$ vectors ($1 \times d$ against $N \times d$) constitutes a **BLAS Level 2 matrix-vector multiplication (GEMV)**. For every individual query, the CPU must stream the entire database ($\approx 500\text{ MB}$) from RAM into CPU caches. The arithmetic intensity (FLOPs per byte transferred across the memory bus) is extremely low ($\approx \mathcal{O}(1)$), making the search strictly **memory-bandwidth bound**.
   - **Batch-query exact search**: Grouping $B$ queries ($B \times d$ against $N \times d$) transforms distance computation into a **BLAS Level 3 matrix-matrix multiplication (GEMM)**. Database vector blocks are loaded into CPU cache once and reused across all $B$ queries. This multiplies arithmetic intensity by a factor of $B$, saturating SIMD floating-point execution pipelines (AVX2/FMA) rather than stalling on RAM bus transfers.

2. **Memory Bandwidth Saturation**:
   - Running 5,000 queries individually forces the memory controller to stream:
     $$5,000 \text{ queries} \times 512\text{ MB/query} \approx 2.56\text{ Terabytes}$$
     over the system bus. With typical DDR4/DDR5 bandwidth (35–50 GB/s), simply streaming the raw bytes requires $\approx 50\text{–}70$ seconds of pure bus transfer time. Batching keeps database blocks resident in cache, cutting memory bus traffic by an order of magnitude.

3. **Software & Dispatch Overhead Amortization**:
   - Single-query search incurs Python-to-C++ wrapper transitions, index dispatch checks, and priority queue/heap initializations 5,000 separate times. Batch search executes this setup only once for the entire batch.

---

## Part 2: Benchmark Three Indexes (HNSW, IVF-Flat, IVF-PQ)

In Part 2, we evaluate three major Approximate Nearest Neighbour (ANN) index paradigms across **SIFT-128** and **GloVe-100**:
1. **Graph-based**: Hierarchical Navigable Small World (**HNSW**)
2. **Inverted File with Exact Vectors**: Inverted File Flat (**IVF-Flat**)
3. **Inverted File with Vector Quantization**: Inverted File Product Quantization (**IVF-PQ**)

### 1. Benchmarking Methodology & Ground Rules
- **Thread Count**: All searches and timing benchmarks run strictly on a **single CPU thread** (`faiss.omp_set_num_threads(1)`).
- **Execution Mode**: Evaluated strictly **one query at a time** (single-query search) to measure true interactive retrieval throughput.
- **Timing & Stability**: Median elapsed time over multiple independent runs is recorded after warm-up to eliminate cold-start cache anomalies.
- **Evaluation Split**: Benchmarked strictly on the reserved evaluation set (queries `5,000` to `9,999`, total 5,000 queries).
- **Metric**: Distance-threshold `recall@10` (no ground-truth ID restriction; candidates $\le d_{10} + \epsilon$ credited, with $\epsilon = 10^{-4}$).

### 2. Parameter Sweep Specifications
To span recall from low values up to $\ge 0.95$:
- **HNSW** (`IndexHNSWFlat`):
  - **Build setting ($M$)**: $M \in \{8, 16, 32\}$ (number of bi-directional graph links per node; $M_{0} = 2M$ at ground layer 0), with construction beam size $\text{efConstruction} = 64$.
  - **Search sweep ($\text{efSearch}$)**: $\text{efSearch} \in \{8, 16, 32, 64, 128, 256, 512\}$.
- **IVF-Flat** (`IndexIVFFlat`):
  - **Build setting ($\text{nlist}$)**: $\text{nlist} \in \{1024, 2048, 4096\}$ (number of Voronoi partitions trained via $k$-means clustering).
  - **Search sweep ($\text{nprobe}$)**: $\text{nprobe} \in \{1, 2, 4, 8, 16, 32, 64, 128, 256\}$ (number of nearest inverted lists scanned).
- **IVF-PQ** (`IndexIVFPQ`):
  - **Build setting ($\text{nlist}=1024$, varied $M_{PQ}$)**:
    - **SIFT-128**: $M_{PQ} \in \{16, 32, 64\}$ sub-quantizers (subvector dimensions $d/M_{PQ} = 8, 4, 2$, 8 bits/code, $k^*=256$ centroids per sub-space).
    - **GloVe-100**: $M_{PQ} \in \{10, 20, 25, 50\}$ sub-quantizers (subvector dimensions $d/M_{PQ} = 10, 5, 4, 2$, 8 bits/code, $k^*=256$ centroids per sub-space).
  - **Search sweep ($\text{nprobe}$)**: $\text{nprobe} \in \{1, 2, 4, 8, 16, 32, 64, 128, 256\}$.

---

### 3. Trade-off Curves & Parameter Sweeps

#### SIFT-128 Trade-off Curve
![SIFT Recall vs QPS](results/figures/sift_recall_vs_qps.png)

#### GloVe-100 Trade-off Curve
![GloVe Recall vs QPS](results/figures/glove_recall_vs_qps.png)

#### Individual Parameter Sweep Plots
Each build parameter sweep across search parameters (`efSearch` or `nprobe`) is plotted and saved in [`results/figures/sweeps/`](results/figures/sweeps/):
- **SIFT Sweeps**:
  - HNSW: [`results/figures/sweeps/sift_hnsw_sweep.png`](results/figures/sweeps/sift_hnsw_sweep.png)
  - IVF-Flat: [`results/figures/sweeps/sift_ivf_flat_sweep.png`](results/figures/sweeps/sift_ivf_flat_sweep.png)
  - IVF-PQ: [`results/figures/sweeps/sift_ivf_pq_sweep.png`](results/figures/sweeps/sift_ivf_pq_sweep.png)
- **GloVe Sweeps**:
  - HNSW: [`results/figures/sweeps/glove_hnsw_sweep.png`](results/figures/sweeps/glove_hnsw_sweep.png)
  - IVF-Flat: [`results/figures/sweeps/glove_ivf_flat_sweep.png`](results/figures/sweeps/glove_ivf_flat_sweep.png)
  - IVF-PQ: [`results/figures/sweeps/glove_ivf_pq_sweep.png`](results/figures/sweeps/glove_ivf_pq_sweep.png)

---

### 4. Benchmark Summary Tables

#### Table 1: Single-Query Throughput (QPS) at Recall 0.90 and 0.95
*Interpolated along each index's Pareto frontier (single thread, queries 5,000–10,000):*

| Dataset | Index Type | QPS @ Recall 0.90 | QPS @ Recall 0.95 | Max Recall Achieved | Exact Baseline QPS (Single) | Speedup vs. Exact @ 0.95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SIFT-128** | **HNSW** | **10,603.2** | **7,412.5** | **0.9992** | 37.23 | **199.1x** |
| **SIFT-128** | **IVF-Flat** | 4,158.4 | 1,948.0 | 1.0000 | 37.23 | 52.3x |
| **SIFT-128** | **IVF-PQ** | 2,765.5 | *N/A (< 0.95)* | 0.9323 | 37.23 | *N/A* |
| **GloVe-100** | **HNSW** | 1,174.5 | *N/A (< 0.95)* | 0.9262 | 42.36 | *N/A* |
| **GloVe-100** | **IVF-Flat** | 934.9 | **367.6** | 0.9806 | 42.36 | **8.7x** |
| **GloVe-100** | **IVF-PQ** | **23,560.8** | 269.9 | 0.9551 | 42.36 | 6.4x |

#### Table 2: Build Cost & Memory Footprint at Recall $\ge$ 0.95
*Comparing the Pareto-optimal configuration reaching recall $\ge 0.95$ for each index:*

| Dataset | Index Type | Best Config @ Recall $\ge$ 0.95 | Operating Recall | Search Param | Build Time (s) | Index Size (MB) | Bytes / Vector | Compression vs. Raw Data |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT-128** | **HNSW** | $M=32, \text{efConst}=64$ | 0.9633 | $\text{efSearch}=64$ | 37.42 s | 747.8 MB | 784.1 | 1.53x expansion |
| **SIFT-128** | **IVF-Flat** | $\text{nlist}=4096$ | 0.9768 | $\text{nprobe}=64$ | 310.12 s | 497.9 MB | 522.1 | 1.02x raw |
| **SIFT-128** | **IVF-PQ** | *Did not reach 0.95* | 0.9323 *(max)* | $\text{nprobe}=256$ | 29.90 s | 69.3 MB | 72.7 | 7.05x compressed |
| **GloVe-100** | **HNSW** | *Did not reach 0.95* | 0.9262 *(max)* | $\text{efSearch}=512$ | 59.37 s | 758.7 MB | 672.2 | 1.68x expansion |
| **GloVe-100** | **IVF-Flat** | $\text{nlist}=1024$ | 0.9511 | $\text{nprobe}=128$ | 23.11 s | 460.9 MB | 408.4 | 1.02x raw |
| **GloVe-100** | **IVF-PQ** | $\text{nlist}=1024, M_{PQ}=50$ | 0.9551 | $\text{nprobe}=256$ | 30.13 s | 66.0 MB | 58.4 | **6.85x compressed** |

*Note: Raw uncompressed vector sizes: SIFT-128 = 512.0 bytes/vector (488.3 MB); GloVe-100 = 400.0 bytes/vector (451.5 MB).*

---

### 5. Why Indexes Fail to Reach High Recall

- **SIFT-128 IVF-PQ (max recall 0.9323)**: IVF-PQ fails to reach recall 0.95 because Asymmetric Distance Computation (ADC) approximates vectors using discrete centroid codes, introducing an irreversible quantization noise floor that exceeds the narrow distance margins between top nearest neighbours.
  - *Measured Evidence*: In [`sift_ivf_pq_sweep.png`](results/figures/sweeps/sift_ivf_pq_sweep.png), recall flatlines horizontally as $\text{nprobe} \to 256$ ($M_{PQ}=16$ caps at 0.829, $M_{PQ}=64$ caps at 0.932), whereas [`sift_ivf_flat_sweep.png`](results/figures/sweeps/sift_ivf_flat_sweep.png) under identical coarse clustering (`nlist=1024`) climbs to 1.0000, isolating quantization distortion as the sole bottleneck. Furthermore, [`glove_ivf_pq_sweep.png`](results/figures/sweeps/glove_ivf_pq_sweep.png) shows the recall ceiling systematically shifting upward as $M_{PQ}$ increases ($0.935 \to 0.941 \to 0.947 \to 0.955$), crossing 0.95 only at $M_{PQ}=50$.
- **GloVe-100 HNSW (max recall 0.9262)**: HNSW struggles to reach 0.95 because angular word embeddings have high local intrinsic dimensionality and dense semantic clusters, which cause greedy graph routing with `efConstruction=64` to become trapped in local minima.
  - *Measured Evidence*: In [`glove_hnsw_sweep.png`](results/figures/sweeps/glove_hnsw_sweep.png), increasing $\text{efSearch}$ hits severe diminishing returns along single curves, whereas doubling graph degree $M$ ($8 \to 16 \to 32$) shifts the ceiling upward by 5–9% (from 0.8038 to 0.9262). In contrast, [`sift_hnsw_sweep.png`](results/figures/sweeps/sift_hnsw_sweep.png) shows $M=8$ easily reaching 0.9927, proving GloVe's clustered angular manifold specifically requires higher graph connectivity ($M \ge 64$) or larger construction exploration ($\text{efConstruction} \ge 128$) to bridge tightly packed clusters.

---

## How to Reproduce

### 1. Run Complete Test Suite
Run all 17 unit and integration tests (metrics, exact agreement, HNSW, IVF-Flat, IVF-PQ, and Pareto logic):
```bash
pytest tests/ -v
```

### 2. Verify Exact Ground Truth Agreement (Part 1)
Verify FAISS exact baseline agreement against dataset ground truth across 5,000 queries:
```bash
python scripts/run_part1.py --verify-only
```

### 3. Run Part 1 Exact Benchmarks
Benchmark single-query and batch exact search on 1 thread:
```bash
python scripts/run_part1.py --dataset all --threads 1
```

### 4. Run Part 2 ANN Benchmarks & Generate Plots
Benchmark HNSW, IVF-Flat, and IVF-PQ parameter sweeps on both datasets, compute Pareto frontiers, and save all figures:
```bash
python scripts/run_part2.py --dataset all --num-runs 2
```
*Generated plots will be written to `results/figures/` and `results/figures/sweeps/`.*

### 5. Fast Laptop Verification (`--quick`)
Run a fast benchmark verification on a 100K vector subset:
```bash
python scripts/run_part1.py --quick
python scripts/run_part2.py --dataset sift --quick
```