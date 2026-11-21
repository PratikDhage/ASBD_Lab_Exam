# ASBD Lab Exam Project — Recipe Reviews & Big Data Analytics

An exhaustive, end-to-end, reproducible Big Data Analytics and Data Mining study on the **UCI Recipe Reviews and User Feedback Dataset**, implemented for the **Analytics and Systems of Big Data (ASBD)** examination.

---

## 1. Project Overview & Architecture
This repository implements a complete analytical lifecycle spanning traditional in-memory data science workflows and distributed Big Data architectures:
- **Exploratory Data Analysis & Understanding:** Profiling 18,182 authentic reviews across 100 recipes and 13,812 users.
- **Systematic Data Preprocessing:** Cleaning HTML entities, imputing missing review text, clipping negative values, feature engineering (text metrics, total engagement, temporal extractions), and task-specific dataset partitioning.
- **Descriptive Analytics & Visualizations:** In-depth central tendency, dispersion, correlation, temporal analyses, and **14 unique visualization archetypes** (Histogram+KDE, Box Plot, Violin Plot, Horizontal Bar Chart, Pareto 80/20 Chart, Dual-Axis Line Chart, Scatter Plot with Linear Regression, Multi-Variable Bubble Chart, Correlation Heatmap, Hexbin Plot, Word Cloud, Stacked 100% Bar Chart, Donut Chart, and Network Graph).
- **Association Rule Mining:** FP-Growth mining yielding 313 frequent itemsets and 1,467 association rules across multiple support/confidence thresholds.
- **Supervised Classification:** K-Nearest Neighbors (KNN) predicting rating classes with target leakage prevention (`best_score` excluded), hyperparameter tuning ($K \in [3..21]$, Euclidean/Manhattan, uniform/distance), classification metrics, and confusion matrix analysis.
- **Hierarchical Clustering:** AGNES (Agglomerative Nesting) with Cophenetic Correlation comparison across Single, Complete, Average, and Ward linkages, generating dendrograms, cluster phenotypes, and 2D PCA projections.
- **Closed Frequent Itemset Mining:** Exact ACLOSE algorithm implementation (minimal generators + closure calculation) verified against a combinatorial brute-force validator, proving up to 47.06% lossless redundancy reduction.
- **PySpark Big Data Implementation:** Apache Spark 3.5.3 distributed processing covering multiLine CSV parsing, Spark SQL Catalyst aggregations, native MLlib FP-Growth, and BisectingKMeans hierarchical divisive clustering.
- **Empirical Performance Benchmarking:** Comprehensive comparison between Pandas and PySpark across 4 progressive dataset scales (10%, 25%, 50%, 100%) and 4 operational tasks (Data Loading, Cleaning, GroupBy Aggregation, and FP-Growth), analyzing execution times, speedup ratios, and the single-node vs distributed crossover threshold.

---

## 2. Directory Structure

```text
ASBD_Lab_Project/
├── dataset/
│   └── dataset.csv                                     # Raw UCI Recipe Reviews Dataset (18,182 rows)
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb                     # EDA, schema, cardinality, missing values
│   ├── 02_Data_Preprocessing.ipynb                     # Data cleaning, text normalization, task datasets
│   ├── 03_Descriptive_Analytics.ipynb                  # Statistical summary & visual gallery
│   ├── 04_FP_Growth_ARM.ipynb                          # Frequent Pattern Growth association analysis
│   ├── 05_KNN_Classification.ipynb                     # Supervised KNN classification without leakage
│   ├── 06_AGNES_Clustering.ipynb                       # Agglomerative hierarchical clustering & dendrogram
│   ├── 07_ACLOSE_Closed_Itemsets.ipynb                 # ACLOSE closed itemset mining & verification
│   ├── 08_PySpark_Analytics.ipynb                      # Distributed PySpark DataFrames, SQL & MLlib
│   └── 09_Performance_Benchmarking.ipynb               # Pandas vs PySpark empirical scaling benchmarks
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                                # Cleaning, feature engineering, task datasets
│   ├── descriptive_stats.py                            # Central tendency, dispersion, correlation matrices
│   ├── visualization.py                                # 14 unique visualization plot generators
│   ├── fp_growth.py                                    # FP-Growth algorithm & association rules
│   ├── knn.py                                          # KNN classification & hyperparameter tuning
│   ├── agnes.py                                        # AGNES hierarchical clustering & linkage comparison
│   ├── aclose.py                                       # ACLOSE algorithm implementation & brute-force test
│   ├── spark_analytics.py                              # PySpark DataFrames, SQL, MLlib FP-Growth, BisectingKMeans
│   └── benchmark.py                                    # Multi-scale execution time & speedup benchmarking
│
├── outputs/
│   ├── cleaned_data/                                   # Cleaned CSV & task-specific partitioned datasets
│   ├── descriptive_statistics/                         # Numerical stats, correlations, temporal trends, profile JSON
│   ├── visualizations/                                 # 14 high-resolution PNG plots (300 DPI)
│   ├── association_rules/                              # Frequent itemsets, rule tables, support vs conf plots
│   ├── knn/                                            # Tuning tables, classification report, confusion matrix, model
│   ├── clustering/                                     # Cophenetic scores, dendrogram PNG, PCA scatter, cluster CSV
│   ├── closed_itemsets/                                # Closed frequent itemsets, FI vs CFI comparison, plots
│   ├── pyspark/                                        # Spark SQL aggregates, Spark MLlib rules, BisectingKMeans CSV
│   └── benchmarks/                                     # Multi-scale benchmark tables and speedup curves
│
├── reports/
│   ├── descriptive_analytics_report.md                 # Exhaustive EDA report & 14 plot analytical dissections
│   ├── predictive_analytics_report.md                  # FP-Growth, KNN, AGNES, ACLOSE in-depth analysis
│   ├── pyspark_comparison_report.md                    # Big data architecture, Spark limitations, benchmark analysis
│   ├── final_lab_summary.md                            # Comprehensive project synthesis & deliverables registry
│   └── EXAM_QUICK_REFERENCE.md                         # Concise examination cheat sheet & parameter quick lookup
│
├── run_all.py                                          # Master script executing all stages sequentially
├── requirements.txt                                    # Python dependencies
└── README.md                                           # Master documentation
```

---

## 3. Environment & Prerequisites
- **Operating System:** Windows 10/11 or Linux / macOS
- **Python Version:** 3.11.2 (Active in virtual environment `venv/`)
- **Java Virtual Machine:** OpenJDK 17.0.21.0.2 LTS 64-Bit (Required for PySpark / Py4J)
- **Primary Libraries:** `pyspark==3.5.3`, `pandas==3.1.1`, `numpy==2.4.6`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `mlxtend==0.25.0`, `wordcloud==1.9.6`, `networkx==3.6.1`.

---

## 4. Execution Instructions

### Option A: Run the Unified Master Pipeline (Recommended)
To execute all 9 stages sequentially, produce all models, generate all plots, save all tables, and verify the existence of all output artifacts:
```bash
.\venv\Scripts\python.exe run_all.py
```

### Option B: Execute Individual Modular Stages
```bash
# 1. Data Cleaning & Feature Engineering
.\venv\Scripts\python.exe src/preprocessing.py

# 2. Descriptive Summary Statistics
.\venv\Scripts\python.exe src/descriptive_stats.py

# 3. 14 Unique Plot Visualizations
.\venv\Scripts\python.exe src/visualization.py

# 4. FP-Growth Association Rule Mining
.\venv\Scripts\python.exe src/fp_growth.py

# 5. K-Nearest Neighbors Classification
.\venv\Scripts\python.exe src/knn.py

# 6. AGNES Hierarchical Clustering
.\venv\Scripts\python.exe src/agnes.py

# 7. ACLOSE Closed Frequent Itemset Mining
.\venv\Scripts\python.exe src/aclose.py

# 8. PySpark Distributed Analytics
.\venv\Scripts\python.exe src/spark_analytics.py

# 9. Performance Benchmarking (Pandas vs PySpark)
.\venv\Scripts\python.exe src/benchmark.py
```

### Option C: Open and Run Jupyter Notebooks
Launch Jupyter Notebook from the project root:
```bash
.\venv\Scripts\jupyter-notebook.exe notebooks/
```
Run notebooks `01` through `09` in sequential order.

---

## 5. Summary of Experimental Results

1. **Descriptive Statistics:**
   - Ratings are heavily skewed towards 5 stars (76.06%), with 9.33% unrated (0 stars) and 9.10% 4-star reviews.
   - User engagement is highly zero-inflated: over 65% of reviews receive zero upvotes or downvotes.
   - Text length serves as a key indicator of critical reviews: 1-2 star reviews have a median length of ~220 characters compared to ~150 characters for 5-star praises.

2. **FP-Growth Association Mining:**
   - 313 frequent itemsets mined at $\text{minsup} = 0.03$; 1,467 association rules mined at $\text{minconf} = 0.50$.
   - `{Feedback_Positive, Rep_Novice} => {Engage_Low}` achieved **Lift = 4.22** (Confidence = 77.18%), proving praise in recipe portals distributes in small increments (1–2 upvotes).
   - Baseline rule `{Feedback_Neutral, Rep_Novice} => {Engage_Zero}` reached **Confidence = 99.96%** (Lift = 1.35), defining the platform's primary passive browsing mode.

3. **KNN Classification:**
   - Target: `rating_3class` (Low [0-2], Medium [3-4], High [5]).
   - Mitigated target leakage by strictly excluding `best_score` ($r = 0.52$ with upvotes).
   - Best model ($K = 7$, Euclidean, Distance-Weighted) achieved **69.65% accuracy** and **0.83 F1-score on High-Star reviews**.

4. **AGNES Hierarchical Clustering:**
   - Evaluated Cophenetic Correlation across linkages: Average (0.9572), Complete (0.8762), Single (0.8399), Ward (0.5243).
   - Stratified sample ($N=2,000$) solved the $O(N^2)$ memory bottleneck.
   - Identified 4 distinct behavioral clusters: Casual Reviewers (81.8%), Narrative Essayists (15.1%), High-Engagement Discussion Leaders (3.0%), and Super-Admin Outliers (0.1%).

5. **ACLOSE Closed Itemset Mining:**
   - Mathematically verified against brute-force baseline $\to$ **100% exact match** on test sample.
   - Eliminated **37.04% to 47.06% of redundant itemsets** across support thresholds (5% to 25%) while losslessly preserving full support information.

6. **PySpark Big Data Analytics:**
   - Successfully loaded and cleaned 18,182 records with multiLine CSV handling.
   - Extracted 148 frequent itemsets and 256 association rules using native MLlib FP-Growth in 2.51 seconds.
   - Clustered data into 4 partitions using scalable BisectingKMeans.

7. **Performance Benchmarking:**
   - Benchmarked across 10%, 25%, 50%, and 100% dataset scales.
   - Proved why single-node Pandas runs faster on small in-memory datasets (< 1 GB) due to Spark's fixed JVM startup, Py4J socket IPC, and Catalyst query planning overhead (~100–300 ms).
   - Demonstrated the Big Data Crossover Principle: PySpark is essential for out-of-core scaling when datasets exceed available RAM (> 10–50 GB).

---

## 6. Exam Quick Reference
For rapid lookup of parameter configurations, metric definitions, and key inferences during examination, consult:  
👉 **`reports/EXAM_QUICK_REFERENCE.md`**
