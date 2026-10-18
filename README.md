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
│   │   └── exact.py         # ExactFlatIndex (FAISS IndexFlatL2 implementation)
│   └── benchmark/
│       ├── timing.py        # Single-query and batch-query timing measurement
│       └── reporter.py      # Standardized results.csv logging and summary formatting
├── tests/
│   ├── test_metrics.py      # 9 unit tests for recall@k (ties, partial, duplicates, edge cases)
│   └── test_exact.py        # 3 integration tests for exact search and dataset agreement
├── scripts/
│   └── run_part1.py         # Standalone CLI runner for Part 1 baseline and benchmarks
├── results/
│   └── results.csv          # Benchmark results log conforming to assignment deliverables
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

## How to Reproduce

### 1. Run Unit Tests
To run all 14 unit and integration tests:
```bash
pytest tests/ -v
```

### 2. Verify Ground Truth Agreement
To quickly verify exact baseline agreement across all 5,000 evaluation queries:
```bash
python scripts/run_part1.py --verify-only
```

### 3. Run Part 1 Benchmarks
To run the full exact baseline benchmarks (generates `results/results.csv`):
```bash
python scripts/run_part1.py --dataset all --threads 1
```

### 4. Fast Laptop Verification (`--quick`)
To run a fast verification on a 100K vector subset:
```bash
python scripts/run_part1.py --quick
```