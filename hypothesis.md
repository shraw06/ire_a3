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