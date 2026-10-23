# Part 3: Scaling Experiment

## Research question
How does increasing the number of SIFT database vectors affect the cost of achieving Recall@10 = 0.95?

## Hypothesis
As the database grows from 100K to 300K to 1M vectors, exact search will require increasingly more computation per query.

HNSW and IVF-Flat may provide better query-time scaling than exact search because they can avoid exhaustively comparing every database vector.

Maintaining Recall@10 = 0.95 may require increased search effort as the database grows.

## Independent variable
Database size: 100K, 300K, and 1M vectors.

## Dependent variables
- QPS at Recall@10 = 0.95
- Query latency at Recall@10 = 0.95
- Build time
- Index memory

## Controlled variables
- Dataset: SIFT
- Distance metric: Euclidean
- Same query set
- Same machine and software versions
- Single-threaded query measurements
- Same recall implementation and timing procedure

# Part 4: Hard Geometry Hypothesis

## Research question
How does cluster separation affect the search effort required by HNSW and IVF-Flat to achieve Recall@10 = 0.95?

## Hypothesis
As clusters become less separated and overlap increases, IVF-Flat will require a higher nprobe to achieve Recall@10 = 0.95, since relevant neighbours may lie in clusters that are not probed. HNSW may exhibit a different sensitivity to cluster overlap because it searches a graph rather than selected clusters.

## Experimental Design:
Generate synthetic clustered datasets with fixed dimensionality, number of vectors, cluster count, and within-cluster variance. Vary only the cluster-separation multiplier across one control and three additional settings. Use a fixed random seed for reproducibility. Compare HNSW and IVF-Flat using recall-versus-QPS curves.