# Big Data Analytics & PySpark Comparative Performance Report
**Course:** Analytics and Systems of Big Data (ASBD)  
**Artifact Directory:** `outputs/pyspark/` & `outputs/benchmarks/`  

---

## 1. Executive Summary & Big Data Architecture
This report evaluates the distributed big data processing implementation of the UCI Recipe Reviews dataset using **Apache Spark (PySpark 3.5.3)** running on Java 17 LTS (OpenJDK 64-Bit). We contrast PySpark's execution model, DataFrame transformations, and MLlib algorithms against single-node Python (Pandas 3.1.1, Scikit-learn 1.9.1, MLxtend 0.25.0) to establish empirical performance benchmarks and understand the scaling properties of big data systems.

---

## 2. PySpark Implementation Details

### 2.1 Cluster Environment & SparkSession Configuration
- **Spark Version:** 3.5.3
- **Master URL:** `local[*]` (utilizing all available physical CPU cores)
- **JVM Runtime:** OpenJDK 17.0.21.0.2 64-Bit
- **Memory Configuration:** Driver Memory = 2 GB, Shuffle Partitions = 4 (optimized for local multi-core execution)
- **CSV Parser Parameters:** `header=True`, `multiLine=True`, `escape='"'`, `inferSchema=True`  
  *Crucial Big Data Implementation Detail:* Free-form recipe review text contains embedded newline characters (`\n`, `\r\n`). Standard line-by-line streaming parsers mistakenly treat internal newlines as separate rows, generating 18,268 fragmented rows. Setting `multiLine=True` enforces correct RFC 4180 parsing, recovering the exact ground truth count of **18,182 records**.

### 2.2 PySpark Distributed Processing Modules
1. **Schema Casting & Null Imputation:** Converted raw text epoch strings to native Spark SQL timestamps (`F.to_timestamp(F.from_unixtime("created_at"))`), extracted calendar partitions (`year`, `month`, `year_month`, `day_of_week`), and imputed null review text using DataFrame API `.fillna({"text": ""})`.
2. **Spark SQL GroupBy Aggregations:** Executed Catalyst-optimized SQL queries over a temporary view (`reviews_tbl`) to compute multi-column recipe performance summaries, monthly volume trends, and rating distribution percentages.
3. **Native MLlib FP-Growth:** Transformed discrete categorical features into `ArrayType(StringType)` item collections and trained Spark's parallelized FP-Growth model (`pyspark.ml.fpm.FPGrowth(minSupport=0.03, minConfidence=0.50)`). Spark identified **148 frequent itemsets** and **256 high-confidence association rules** in 2.51 seconds.
4. **PySpark ML Feature Pipeline:** Structured continuous engagement and text metrics into unified feature vectors via `VectorAssembler` and applied distributed z-score normalization using `StandardScaler(withMean=True, withStd=True)`.
5. **Big Data Clustering (BisectingKMeans):** Applied hierarchical divisive clustering directly in Spark MLlib ($k=4$, seed=42). The tree-based divisive strategy clustered the 18,182 records into 4 partitions:
   - Cluster 2: 14,521 records (80.0% — Casual Reviewers)
   - Cluster 3: 3,320 records (18.3% — Narrative Reviewers)
   - Cluster 0: 208 records (1.1% — High Engagement Disputed Items)
   - Cluster 1: 133 records (0.7% — Outliers)

---

## 3. Algorithmic Scalability & Distributed Limitations in Spark

A critical theoretical requirement of Big Data Analytics is understanding why certain classical data mining algorithms map cleanly onto MapReduce / Spark DAGs while others do not:

### 3.1 FP-Growth in Spark (PFP Algorithm)
- **Status:** **Fully Distributed Native Support** (`pyspark.ml.fpm.FPGrowth`).
- **Distributed Mechanism:** Spark implements **Parallel FP-Growth (PFP)** (Li et al., 2008). Rather than building one massive FP-tree in a single machine's memory, PFP partitions transactions across workers based on item frequency, constructs independent conditional local FP-trees in parallel, and merges frequent itemsets via parallel reduction.

### 3.2 K-Nearest Neighbors (KNN) in Spark
- **Status:** **Not Available as a Standard Native MLlib Classifier.**
- **Architectural Bottleneck:** KNN is a lazy, instance-based learner. To classify a query vector $q$, KNN requires computing distances against the *entire* training corpus. In a distributed cluster where $N$ training points are partitioned across dozens of worker nodes, every test query triggers an all-to-all broadcast or shuffle ($O(M \times N)$ communication complexity), saturating network bandwidth.
- **Spark Alternative:** In production Big Data pipelines, distributed nearest-neighbor search relies on **Locality-Sensitive Hashing (LSH)** (`pyspark.ml.feature.BucketedRandomProjectionLSH`), which approximates nearest neighbors in sub-linear time $O(M \log N)$ without global pairwise shuffling.

### 3.3 AGNES (Hierarchical Agglomerative) in Spark
- **Status:** **Not Scalable in Distributed Environments.**
- **Architectural Bottleneck:** AGNES is fundamentally sequential and greedy. At each step, it must find the global minimum pair across the entire distance matrix, fuse them, and recalculate distances ($O(N^3)$ operations). Synchronizing global minimum state across worker nodes at every merge step requires massive coordination overhead.
- **Spark Alternative:** Spark MLlib provides **Bisecting k-Means** (hierarchical divisive / top-down clustering). It recursively splits clusters using standard k-Means, mapping cleanly onto Spark's distributed RDD partitions.

### 3.4 ACLOSE (Closed Itemsets) in Spark
- **Status:** **Custom Implementation Required.**
- **Architectural Bottleneck:** ACLOSE discovers minimal generators and evaluates Galois closures through vertical transaction-id list intersections ($c(X) = \bigcap_{t \in \mathcal{T}(X)} t$). While vertical mining algorithms (such as dEclat) can be implemented in Spark, maintaining vertical inverted indices across distributed partitions induces substantial shuffle overhead.

---

## 4. Empirical Performance Benchmarking: Pandas vs. PySpark

We executed a multi-scale benchmarking protocol comparing execution times across 4 progressive dataset fractions:
- **10% Scale:** 1,818 rows
- **25% Scale:** 4,545 rows
- **50% Scale:** 9,091 rows
- **100% Scale:** 18,182 rows

Each task was benchmarked multiple times and averaged. The **Speedup Ratio** is defined as:
$$\text{Speedup} = \frac{\text{Execution Time}_{\text{Pandas}}}{\text{Execution Time}_{\text{PySpark}}}$$
*(A Speedup $< 1.0$ indicates that Pandas is faster; a Speedup $> 1.0$ indicates that PySpark is faster).*

### 4.1 Benchmark Results Table

| Scale (%) | Records ($N$) | Task | Pandas Time (s) | PySpark Time (s) | Speedup Ratio ($\times$) | Relative Performance |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **10%** | 1,818 | **Data Loading (CSV)** | 0.0156s | 2.4587s | **0.01x** | Pandas 157x faster (Spark JVM cold start) |
| 10% | 1,818 | **Cleaning & Feature Derivation** | 0.0095s | 0.2875s | **0.03x** | Pandas 30x faster |
| 10% | 1,818 | **GroupBy Aggregations** | 0.0298s | 0.3776s | **0.08x** | Pandas 12x faster |
| 10% | 1,818 | **FP-Growth Mining** | 0.0064s | 2.2296s | **0.00x** | Pandas 348x faster (Spark ML initialization) |
| **25%** | 4,545 | **Data Loading (CSV)** | 0.0317s | 0.2497s | **0.13x** | Amortized Spark execution |
| 25% | 4,545 | **Cleaning & Feature Derivation** | 0.0102s | 0.1632s | **0.06x** | Pandas faster |
| 25% | 4,545 | **GroupBy Aggregations** | 0.0056s | 0.2106s | **0.03x** | Pandas faster |
| 25% | 4,545 | **FP-Growth Mining** | 0.0180s | 0.4081s | **0.04x** | Pandas faster |
| **50%** | 9,091 | **Data Loading (CSV)** | 0.0311s | 0.2562s | **0.12x** | Pandas faster |
| 50% | 9,091 | **Cleaning & Feature Derivation** | 0.0080s | 0.1481s | **0.05x** | Pandas faster |
| 50% | 9,091 | **GroupBy Aggregations** | 0.0052s | 0.2013s | **0.03x** | Pandas faster |
| 50% | 9,091 | **FP-Growth Mining** | 0.0372s | 0.4140s | **0.09x** | Spark gap closing |
| **100%** | 18,182 | **Data Loading (CSV)** | 0.0666s | 0.2915s | **0.23x** | Spark scaling stably (~0.29s) |
| 100% | 18,182 | **Cleaning & Feature Derivation** | 0.0166s | 0.1618s | **0.10x** | Pandas faster |
| 100% | 18,182 | **GroupBy Aggregations** | 0.0034s | 0.2057s | **0.02x** | Pandas in-memory C-speed |
| 100% | 18,182 | **FP-Growth Mining** | 0.0658s | 0.4326s | **0.15x** | Spark scaling linearly |

---

## 5. Technical Root-Cause Analysis: Why PySpark is Slower on Single-Node Local Hardware

An essential analytical takeaway for exams is explaining why PySpark exhibits lower speedup numbers on this dataset:

### 1. The Fixed Latency Floor of the Spark Engine
PySpark is not a lightweight Python script; it is a full distributed query engine. Every operation incurs an unavoidable fixed latency floor (~100 to 300 ms) regardless of input size:
- **Py4J Socket Communication:** PySpark transmits commands from the Python runtime to the underlying Java Virtual Machine (JVM) over local TCP loopback sockets (`127.0.0.1:port`).
- **Catalyst Query Optimizer:** Spark parses the DataFrame query, creates an unresolved logical plan, resolves schemas against catalog references, evaluates optimization rules (filter pushdown, projection pruning), and generates physical JVM bytecode.
- **Tungsten & DAG Scheduling:** The Catalyst engine divides the execution plan into stages, schedules task tasks across threads, and instantiates task contexts.

### 2. Zero-Copy C/C++ In-Memory Execution in Pandas
In contrast, for a dataset of 18,182 rows (~6 MB in memory), Pandas loads data directly into contiguous NumPy C-arrays. There are no IPC sockets, no query compilation stages, and no threading overhead. Pandas executes grouping and aggregations at bare-metal memory speeds ($< 5\text{ ms}$).

### 3. The Big Data Crossover Threshold
PySpark was intentionally engineered for datasets that **exceed single-machine RAM capacity** (e.g., 50 GB to 100 TB):
- **When $N < 10^6$ rows (< 1 GB):** Pandas is vastly superior in development speed and runtime latency.
- **When $N > 10^7$ rows (> 10 GB):** Pandas triggers operating system swap thrashing or crashes with `MemoryError` (Out of Memory - OOM).
- **At Big Data Scale:** PySpark outscales Pandas infinitely because its execution engine automatically spills intermediate shuffle partitions to disk, routes partitions across thousands of cluster nodes, and provides fault tolerance via RDD lineage recomputation.
