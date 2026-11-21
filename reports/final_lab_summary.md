# ASBD Lab Exam — Final Project Summary & Comprehensive Synthesis
**Course:** Analytics and Systems of Big Data (ASBD)  
**Project:** End-to-End Descriptive & Predictive Analytics with PySpark Distributed Benchmarking  
**Dataset:** Recipe Reviews and User Feedback Dataset (UCI Machine Learning Repository)  
**Execution Date:** 2026-11-21  

---

## 1. Executive Summary
This document synthesizes the complete implementation of the ASBD Lab Project, fulfilling all 18 requirements of the Master Implementation Prompt. The project delivers an end-to-end, fully reproducible analytics pipeline covering exploratory data analysis, systematic data cleaning, extensive descriptive statistics, 14 unique visualization archetypes, Association Rule Mining (FP-Growth), Supervised Classification (KNN), Hierarchical Clustering (AGNES), Closed Frequent Itemset Mining (ACLOSE), PySpark distributed processing, and empirical performance benchmarking.

---

## 2. Key Analytical & Experimental Findings

### 2.1 Data Understanding & Preprocessing
- **Dataset Scale:** 18,182 authentic reviews covering 100 recipes across 13,812 unique users from February 2021 to October 2022.
- **Cleaning & Quality Assurance:** Dropped redundant index column `Unnamed: 0`; imputed 2 null text entries; decoded HTML entities (`&amp;`, `&#39;`, `&quot;`); validated ranges ($[0, 5]$ for stars, non-negative bounds for social metrics).
- **Engineered Dimensions (34 columns total):** Extracted temporal features (`year`, `month`, `year_month`, `day_of_week`, `hour`, `is_weekend`), text length, word count, total engagement, net thumbs, approval ratio, and 5 discrete behavioral tiers.
- **Task-Specific Datasets:** Created clean partitions for ARM (`arm_transactions.csv`), KNN (`knn_features.csv` without target leakage), and AGNES (`agnes_features.csv`).

### 2.2 Descriptive Analytics & Visualization (14 Unique Archetypes)
- **Extreme Class Imbalance:** 76.06% of reviews are 5-star; 9.33% are 0-star (unrated); only 5.51% represent 1-to-3 stars combined.
- **Text Length vs. Rating:** Box plots revealed that critical reviews (1-2 stars) have significantly longer median character lengths (~220 chars) compared to 5-star praises (~150 chars), reflecting detailed consumer critique.
- **Pareto Concentration:** The top 25 recipes account for > 58% of all platform reviews; 42 recipes account for 80% of volume.
- **Seasonality:** Longitudinal dual-axis line charts revealed clear winter holiday spikes (November–January) with stable year-round satisfaction (~4.3 stars).
- **Leakage Discovery:** Upvotes correlate strongly with algorithmic `best_score` ($r = 0.52$), confirming the need to exclude `best_score` from predictive classification features.

### 2.3 FP-Growth Association Rule Mining
- **Rule Generation:** Discovered 313 frequent itemsets at $\text{minsup} = 0.03$ and 1,467 association rules at $\text{minconf} = 0.50$ (683 rules with Lift $\ge 1.05$).
- **Top Lift Rule:** `{Feedback_Positive, Rep_Novice} => {Engage_Low}` achieved **Lift = 4.22** (Confidence = 77.18%), proving that community upvotes in recipe portals are distributed in modest increments (1–2 thumbs) rather than viral surges.
- **Baseline Rule:** `{Feedback_Neutral, Rep_Novice} => {Engage_Zero}` achieved **Confidence = 99.96%** (Lift = 1.35, Conviction = 683.25), proving that passive, unvoted browsing is the overwhelming structural baseline of the community.

### 2.4 K-Nearest Neighbors (KNN) Supervised Classification
- **Target Selection & Leakage Mitigation:** Predicts 3-class review rating (Low [0-2], Medium [3-4], High [5]) while strictly excluding `best_score` to prevent target leakage.
- **Optimal Hyperparameters:** $K = 7$, Distance Metric = Euclidean ($L_2$), Weighting = Distance-Weighted ($w_i = 1/d$).
- **Performance:** Reached **69.65% accuracy** and **0.83 F1-score on the majority High-Star class**. Macro F1 (0.3813) reflects the severe real-world challenge of separating overlapping continuous metrics for imbalanced minority classes.

### 2.5 AGNES Hierarchical Clustering
- **Mathematical Architecture:** Bottom-up agglomerative clustering with standardized continuous interaction features.
- **Computational Scalability Justification:** Full dataset ($N=18,182$) requires 165.3 million pairwise distances ($> 1.3\text{ GB}$ matrix, $O(N^3)$ time), which was solved by drawing a statistically representative sample of $N=2,000$.
- **Linkage Cophenetic Comparison:** Average Linkage achieved the highest distance preservation (**0.9572**), followed by Complete (0.8762) and Ward (0.5243).
- **Cluster Phenotypes ($k=4$, Ward Linkage):**
  - **Cluster 1 (81.8%):** Casual everyday reviewers (Avg length 146 chars, 0.46 thumbs).
  - **Cluster 2 (15.1%):** Narrative recipe essayists (Avg length 507 chars).
  - **Cluster 0 (3.0%):** Viral discussion leaders (Avg thumbs 18.54, 0.56 replies).
  - **Cluster 3 (0.1%):** Platform super-admin / power user (Reputation 510).
- **Cluster Quality Evaluation:** Quantitative validation confirms robust cluster partitioning with a **Silhouette Score of 0.5820** (> 0.50 benchmark), **Calinski-Harabasz Index of 652.53**, and **Davies-Bouldin Index of 0.9884** (< 1.0 threshold), saved to `outputs/clustering/agnes_evaluation_metrics.txt`.

### 2.6 ACLOSE Closed Frequent Itemset Mining
- **Theory & Implementation:** Implemented the exact ACLOSE algorithm (minimal generator generation + generator pruning + vertical tid closure computation).
- **Mathematical Verification:** Validated against an exhaustive combinatorial brute-force validator on a test sample $\to$ **100% exact match** (105 CFIs).
- **Redundancy Reduction:** Reduced frequent itemsets by **37.04% to 47.06% across support thresholds** (5% to 25%), proving that CFIs compress itemsets losslessly without losing any support information.

### 2.7 PySpark & Big Data Performance Benchmarking
- **PySpark Capabilities:** Handled multiLine CSV parsing, Spark SQL Catalyst aggregations, native MLlib FP-Growth (148 frequent itemsets, 256 rules), and BisectingKMeans hierarchical divisive clustering.
- **Algorithmic Limitations in Spark:** Explains why lazy KNN and sequential AGNES cannot be efficiently distributed on MapReduce/Spark without massive shuffling ($O(M \times N)$) or synchronization overhead, highlighting Spark's native alternatives (LSH and BisectingKMeans).
- **Empirical Benchmarks across 10%, 25%, 50%, 100% Scales:**
  - On single-node local hardware for 18,182 rows (~6 MB), Pandas outperforms PySpark (Speedup $\approx 0.02\times - 0.23\times$) due to Spark's fixed JVM startup, Py4J TCP socket IPC serialization, and Catalyst optimizer planning floor (~100–300 ms).
  - Proves the **Big Data Crossover Principle**: Spark's distributed architecture becomes essential and outscales Pandas when dataset sizes exceed single-machine RAM (> 10–50 GB), where Pandas crashes with OOM while Spark spills gracefully to disk.

---

## 3. Master Deliverables & File Registry

| Component | Path / Location | Description |
| :--- | :--- | :--- |
| **Cleaned Datasets** | `outputs/cleaned_data/cleaned_dataset.csv` | Full preprocessed dataset (18,182 rows, 34 features) |
| | `outputs/cleaned_data/arm_transactions.csv` | Discrete transaction baskets for FP-Growth & ACLOSE |
| | `outputs/cleaned_data/knn_features.csv` | Leakage-free features for supervised classification |
| | `outputs/cleaned_data/agnes_features.csv` | Standardized features for hierarchical clustering |
| **Statistical Outputs** | `outputs/descriptive_statistics/numerical_summary_statistics.csv` | Central tendency, dispersion, skewness, kurtosis |
| | `outputs/descriptive_statistics/correlation_pearson.csv` | Pearson correlation matrix |
| | `outputs/descriptive_statistics/temporal_monthly_trends.csv` | Monthly review volume and rating trends |
| | `outputs/descriptive_statistics/dataset_profile.json` | High-level dataset metadata profile |
| **Visualizations (14)** | `outputs/visualizations/01_histogram_stars_kde.png` | Star rating distribution with KDE curve |
| | `outputs/visualizations/02_boxplot_review_length_by_rating.png` | Text length spread and outliers across ratings |
| | `outputs/visualizations/03_violinplot_reputation_by_rating.png` | User reputation kernel density across rating classes |
| | `outputs/visualizations/04_barchart_top_recipes.png` | Top 15 most reviewed recipes horizontal bar chart |
| | `outputs/visualizations/05_pareto_chart_recipes.png` | Pareto 80/20 cumulative review concentration chart |
| | `outputs/visualizations/06_linechart_temporal_trends.png` | Dual-axis monthly volume and average rating trajectory |
| | `outputs/visualizations/07_scatterplot_thumbs_vs_bestscore.png` | Upvotes vs Best Score scatter plot with linear regression |
| | `outputs/visualizations/08_bubblechart_recipe_engagement.png` | 4-variable recipe performance bubble chart |
| | `outputs/visualizations/09_heatmap_correlation_matrix.png` | Annotated Pearson correlation matrix heatmap |
| | `outputs/visualizations/10_hexbin_thumbs_density.png` | Log-scaled 2D density of Thumbs Up vs Down |
| | `outputs/visualizations/11_wordcloud_review_text.png` | Qualitative text word cloud for 5-star positive reviews |
| | `outputs/visualizations/12_stacked_bar_rating_by_engagement.png` | Stacked 100% rating breakdown by engagement tier |
| | `outputs/visualizations/13_donut_chart_reputation_segments.png` | Reviewer community reputation tier donut chart |
| | `outputs/visualizations/14_network_graph_association_rules.png` | Transactional attribute co-occurrence network graph |
| **Association Rules** | `outputs/association_rules/frequent_itemsets.csv` | 313 frequent itemsets at minsup=0.03 |
| | `outputs/association_rules/top_rules_by_lift.csv` | Top association rules ranked by lift |
| | `outputs/association_rules/arm_support_vs_confidence.png` | Support vs Confidence scatter colored by Lift |
| **KNN Classification** | `outputs/knn/knn_hyperparameter_tuning.csv` | Full grid search across K, distance, and weights |
| | `outputs/knn/knn_classification_report.csv` | Precision, recall, and F1 per rating class |
| | `outputs/knn/knn_confusion_matrix.csv` | Confusion matrix table |
| | `outputs/knn/best_knn_model.joblib` | Serialized best KNN model ($K=7$, Euclidean, Distance) |
| | `outputs/knn/knn_performance_vs_k.png` | Accuracy and Macro F1 vs K curves |
| **AGNES Clustering** | `outputs/clustering/linkage_cophenetic_scores.csv` | Cophenetic scores (Single, Complete, Average, Ward) |
| | `outputs/clustering/agnes_cluster_profiles.csv` | Cluster centroid profiles and summary statistics |
| | `outputs/clustering/agnes_dendrogram_ward.png` | Truncated Ward linkage dendrogram with cut line |
| | `outputs/clustering/agnes_pca_cluster_scatter.png` | 2D PCA cluster projection scatter plot |
| **ACLOSE Itemsets** | `outputs/closed_itemsets/closed_frequent_itemsets.csv` | Mined closed frequent itemsets with support counts |
| | `outputs/closed_itemsets/fi_vs_cfi_comparison.csv` | FI vs CFI counts and redundancy reduction across scales |
| | `outputs/closed_itemsets/aclose_fi_vs_cfi_comparison.png` | Bar chart comparison of FI vs CFI across support thresholds |
| **PySpark Outputs** | `outputs/pyspark/spark_recipe_performance_summary.csv` | Spark SQL aggregation table for recipes |
| | `outputs/pyspark/spark_frequent_itemsets.csv` | MLlib parallel FP-Growth frequent itemsets |
| | `outputs/pyspark/spark_association_rules.csv` | MLlib parallel FP-Growth association rules |
| | `outputs/pyspark/spark_bisecting_kmeans_clusters.csv` | Big data divisive hierarchical cluster sizes |
| **Benchmarking** | `outputs/benchmarks/pandas_vs_pyspark_benchmark.csv` | Empirical execution times and speedups across 4 scales |
| | `outputs/benchmarks/benchmark_execution_times.png` | Scaling curves of execution times |
| | `outputs/benchmarks/benchmark_speedup_ratios.png` | Speedup ratios across dataset scale percentages |
| **Jupyter Notebooks** | `notebooks/01_Data_Understanding.ipynb` | Executable data profiling notebook |
| | `notebooks/02_Data_Preprocessing.ipynb` | Executable preprocessing & cleaning notebook |
| | `notebooks/03_Descriptive_Analytics.ipynb` | Executable statistics & visual gallery notebook |
| | `notebooks/04_FP_Growth_ARM.ipynb` | Executable FP-Growth ARM notebook |
| | `notebooks/05_KNN_Classification.ipynb` | Executable KNN classification notebook |
| | `notebooks/06_AGNES_Clustering.ipynb` | Executable AGNES hierarchical clustering notebook |
| | `notebooks/07_ACLOSE_Closed_Itemsets.ipynb` | Executable ACLOSE closed itemsets notebook |
| | `notebooks/08_PySpark_Analytics.ipynb` | Executable PySpark big data analytics notebook |
| | `notebooks/09_Performance_Benchmarking.ipynb` | Executable Pandas vs PySpark benchmark notebook |
| **Reports** | `reports/descriptive_analytics_report.md` | Comprehensive EDA and 14 plot dissections |
| | `reports/predictive_analytics_report.md` | Comprehensive FP-Growth, KNN, AGNES, ACLOSE report |
| | `reports/pyspark_comparison_report.md` | PySpark architecture, limitations, and benchmarks |
| | `reports/EXAM_QUICK_REFERENCE.md` | Concise cheat sheet for lab exam navigation |
| | `reports/final_lab_summary.md` | Master project synthesis and verification |
| **Execution Scripts** | `run_all.py` | Unified master pipeline executing all stages |
| | `README.md` | Complete project documentation and run guide |
| | `requirements.txt` | Dependency specifications |
