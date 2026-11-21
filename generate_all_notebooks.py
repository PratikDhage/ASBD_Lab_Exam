"""
Notebook Generator for ASBD Lab Exam Project
Generates 9 standard, fully-documented, executable Jupyter Notebooks (.ipynb)
"""

import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.2"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def generate_notebooks():
    nb_dir = "notebooks"
    os.makedirs(nb_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Notebook 01: Data Understanding
    # -------------------------------------------------------------
    nb01_cells = [
        md_cell("""# 01. Data Understanding and Initial Profiling
## ASBD Lab Exam — Analytics and Systems of Big Data
**Dataset:** Recipe Reviews and User Feedback Dataset (UCI Machine Learning Repository)  
**Author:** Candidate / ASBD Student  

---
### 1. Objective
Perform a comprehensive initial exploratory analysis of the raw UCI Recipe Reviews dataset:
- Verify dimensions, column names, datatypes, and memory usage.
- Assess cardinality, unique users, unique recipes, and duplicate records.
- Inspect numerical statistics, rating distributions, and temporal coverage.
- Classify attributes into identifiers, numerical, categorical, temporal, and text features."""),
        code_cell("""import os
import pandas as pd
import numpy as np

# Load dataset
df_raw = pd.read_csv('../dataset/dataset.csv')
print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head()"""),
        md_cell("""### 2. Attribute Classification and Cardinality Analysis
We evaluate the distinct count and null count of every column to determine their analytical utility."""),
        code_cell("""# Column Profiling
profile_data = []
for col in df_raw.columns:
    profile_data.append({
        'Attribute': col,
        'DataType': str(df_raw[col].dtype),
        'Unique_Count': df_raw[col].nunique(),
        'Missing_Count': df_raw[col].isnull().sum(),
        'Missing_Pct': round(df_raw[col].isnull().mean() * 100, 3)
    })
df_profile = pd.DataFrame(profile_data)
df_profile"""),
        md_cell("""### 3. Numerical Summary Statistics
Summary metrics for numeric attributes (`stars`, `user_reputation`, `reply_count`, `thumbs_up`, `thumbs_down`, `best_score`)."""),
        code_cell("""num_cols = ['stars', 'user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'best_score']
df_raw[num_cols].describe().T"""),
        md_cell("""### 4. Star Rating and Temporal Span Analysis"""),
        code_cell("""# Star rating counts
print("Rating Breakdown:")
print(df_raw['stars'].value_counts().sort_index())

# Datetime conversion
df_raw['datetime'] = pd.to_datetime(df_raw['created_at'], unit='s', errors='coerce')
print(f"Earliest Review: {df_raw['datetime'].min()}")
print(f"Latest Review:   {df_raw['datetime'].max()}")"""),
        md_cell("""### 5. Key Inferences and Next Steps
1. `Unnamed: 0`: Redundant index counter from CSV export (725 unique values repeating). Must be dropped.
2. `recipe_number` / `recipe_code` / `recipe_name`: Exactly 100 unique recipes represented in 18,182 reviews.
3. `user_id` (13,812 unique) & `comment_id` (18,182 unique): No duplicate comment IDs detected.
4. `stars`: Right-skewed distribution where 76.06% of reviews are 5-star ratings; 9.3% are unrated (0 stars).
5. `text`: 2 null values requiring imputation/empty string handling.
Proceed to **Notebook 02: Data Preprocessing**.""")
    ]
    with open(os.path.join(nb_dir, "01_Data_Understanding.ipynb"), "w") as f:
        json.dump(make_notebook(nb01_cells), f, indent=2)
    print("Generated 01_Data_Understanding.ipynb")

    # -------------------------------------------------------------
    # Notebook 02: Data Preprocessing
    # -------------------------------------------------------------
    nb02_cells = [
        md_cell("""# 02. Data Preprocessing and Task-Specific Transformations
## ASBD Lab Exam — Analytics and Systems of Big Data

---
### 1. Objective
Systematically apply data cleaning, text normalization, outlier inspection, feature engineering, and discretization to construct task-specific datasets for ARM, KNN, AGNES, and ACLOSE."""),
        code_cell("""import sys
sys.path.append('..')
from src.preprocessing import preprocess_data, clean_text
import pandas as pd

# Execute modular preprocessing pipeline
df_clean = preprocess_data(raw_csv_path='../dataset/dataset.csv', output_dir='../outputs/cleaned_data')
df_clean.info()"""),
        md_cell("""### 2. Feature Engineering & Discretization Breakdown
- **Temporal extraction:** Year, Month, Year-Month, Day of Week, Hour, Is_Weekend.
- **Content metrics:** Text length (characters), Word count.
- **Engagement features:** Total engagement (`thumbs_up + thumbs_down + reply_count`), Net thumbs, Approval ratio.
- **Categorical bins:** Rating categories, Reputation tiers, Engagement tiers, Length tiers, Feedback balance."""),
        code_cell("""# Inspect generated bins
print(df_clean[['rating_cat', 'reputation_cat', 'engagement_cat', 'length_cat', 'feedback_cat']].head(10))"""),
        md_cell("""### 3. Task-Specific Dataset Verification
Verify the generation of:
- `arm_transactions.csv` (Categorical token baskets for FP-Growth and ACLOSE)
- `knn_features.csv` (Leakage-free numerical/categorical features for supervised classification)
- `agnes_features.csv` (Standardized interaction features for hierarchical clustering)"""),
        code_cell("""import os
for f in ['cleaned_dataset.csv', 'arm_transactions.csv', 'knn_features.csv', 'agnes_features.csv']:
    path = os.path.join('../outputs/cleaned_data', f)
    print(f"File: {f:<22} | Exists: {os.path.exists(path)} | Size: {os.path.getsize(path)/1024:.1f} KB")""")
    ]
    with open(os.path.join(nb_dir, "02_Data_Preprocessing.ipynb"), "w") as f:
        json.dump(make_notebook(nb02_cells), f, indent=2)
    print("Generated 02_Data_Preprocessing.ipynb")

    # -------------------------------------------------------------
    # Notebook 03: Descriptive Analytics & Visualizations
    # -------------------------------------------------------------
    nb03_cells = [
        md_cell("""# 03. Comprehensive Descriptive Analytics & Visualizations
## ASBD Lab Exam — Analytics and Systems of Big Data

---
### 1. Objective
Perform thorough numerical, categorical, correlation, and temporal analyses and generate at least 10 unique plot types."""),
        code_cell("""import sys
sys.path.append('..')
import pandas as pd
from src.descriptive_stats import compute_descriptive_statistics
from src.visualization import generate_all_visualizations

# Compute and save all statistical summary tables
compute_descriptive_statistics(cleaned_csv='../outputs/cleaned_data/cleaned_dataset.csv', output_dir='../outputs/descriptive_statistics')

# Generate 14 unique visualization plot types
generate_all_visualizations(cleaned_csv='../outputs/cleaned_data/cleaned_dataset.csv', output_dir='../outputs/visualizations')"""),
        md_cell("""### 2. Numerical Summary Statistics
Displaying central tendency, dispersion, skewness, and kurtosis for all quantitative metrics."""),
        code_cell("""df_num_stats = pd.read_csv('../outputs/descriptive_statistics/numerical_summary_statistics.csv')
df_num_stats"""),
        md_cell("""### 3. Visual Gallery of Generated Unique Plots
Inspect sample generated plots across the 14 distinct visualization archetypes."""),
        code_cell("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plots_to_show = [
    '01_histogram_stars_kde.png',
    '02_boxplot_review_length_by_rating.png',
    '05_pareto_chart_recipes.png',
    '07_scatterplot_thumbs_vs_bestscore.png',
    '08_bubblechart_recipe_engagement.png',
    '09_heatmap_correlation_matrix.png'
]

fig, axes = plt.subplots(3, 2, figsize=(16, 18), dpi=150)
for ax, p in zip(axes.flatten(), plots_to_show):
    img = mpimg.imread(os.path.join('../outputs/visualizations', p))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(p, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()""")
    ]
    with open(os.path.join(nb_dir, "03_Descriptive_Analytics.ipynb"), "w") as f:
        json.dump(make_notebook(nb03_cells), f, indent=2)
    print("Generated 03_Descriptive_Analytics.ipynb")

    # -------------------------------------------------------------
    # Notebook 04: FP-Growth ARM
    # -------------------------------------------------------------
    nb04_cells = [
        md_cell("""# 04. Association Rule Mining using FP-Growth
## ASBD Lab Exam — Analytics and Systems of Big Data

---
### 1. Theory & Problem Formulation
Association Rule Mining (ARM) identifies strong correlations between co-occurring discrete items.  
Unlike the Apriori algorithm, which generates $2^k$ candidate itemsets via costly multi-pass joins, **FP-Growth (Frequent Pattern Growth)**:
1. Builds a compact **FP-Tree (Frequent Pattern Tree)** via two database scans.
2. Mines frequent itemsets directly by recursively growing conditional pattern bases, entirely avoiding candidate generation.

**Key Metrics:**
- **Support:** $P(A \\cup B) = \\frac{\\text{freq}(A, B)}{N}$
- **Confidence:** $P(B|A) = \\frac{\\text{freq}(A, B)}{\\text{freq}(A)}$
- **Lift:** $\\frac{P(A \\cup B)}{P(A) \\cdot P(B)}$ (Lift > 1 indicates positive correlation)
- **Leverage:** $P(A \\cup B) - P(A) \\cdot P(B)$
- **Conviction:** $\\frac{1 - P(B)}{1 - \\text{Confidence}(A \\to B)}$"""),
        code_cell("""import sys
sys.path.append('..')
from src.fp_growth import run_fpgrowth_analysis
import pandas as pd

# Execute FP-Growth analysis across multiple thresholds
fi, rules = run_fpgrowth_analysis(
    transactions_csv='../outputs/cleaned_data/arm_transactions.csv',
    output_dir='../outputs/association_rules'
)"""),
        md_cell("""### 2. Support Threshold Experiments
Evaluating itemset growth across support thresholds: 2%, 3%, 5%, 10%, 15%."""),
        code_cell("""df_sup_exp = pd.read_csv('../outputs/association_rules/fpgrowth_support_experiments.csv')
df_sup_exp"""),
        md_cell("""### 3. Top Association Rules by Lift
Examining the strongest non-trivial behavioural rules."""),
        code_cell("""df_top_lift = pd.read_csv('../outputs/association_rules/top_rules_by_lift.csv')
df_top_lift[['rule_str', 'support', 'confidence', 'lift', 'leverage']].head(10)"""),
        md_cell("""### 4. Rule Visualizations
Support vs Confidence scatter plot and top rules ranked by Lift."""),
        code_cell("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_arm = mpimg.imread('../outputs/association_rules/arm_support_vs_confidence.png')
plt.figure(figsize=(10, 6), dpi=150)
plt.imshow(img_arm)
plt.axis('off')
plt.show()""")
    ]
    with open(os.path.join(nb_dir, "04_FP_Growth_ARM.ipynb"), "w") as f:
        json.dump(make_notebook(nb04_cells), f, indent=2)
    print("Generated 04_FP_Growth_ARM.ipynb")

    # -------------------------------------------------------------
    # Notebook 05: KNN Classification
    # -------------------------------------------------------------
    nb05_cells = [
        md_cell("""# 05. Supervised Classification using K-Nearest Neighbors (KNN)
## ASBD Lab Exam — Analytics and Systems of Big Data

---
### 1. Objective and Problem Definition
Classify user review rating category using behavioral and engagement metrics.
- **Target:** `rating_3class` (Class 0: Low [0-2], Class 1: Medium [3-4], Class 2: High [5]).
- **Leakage Prevention:** `best_score` is intentionally excluded because it is an algorithmic function derived directly from ratings and engagement. Including it produces trivial 99% accuracy via target leakage.
- **Features:** `user_reputation`, `reply_count`, `thumbs_up`, `thumbs_down`, `text_length`, `word_count`, `hour`, `month`, `is_weekend`.
- **Scaling:** Feature standardization (zero mean, unit variance).
- **Hyperparameter Exploration:** Tuning K across [3, 5, 7, 9, 11, 15, 21], distance metrics (Euclidean, Manhattan), and weighting (uniform, distance)."""),
        code_cell("""import sys
sys.path.append('..')
from src.knn import run_knn_classification
import pandas as pd

best_model, best_params, df_tuning = run_knn_classification(
    features_csv='../outputs/cleaned_data/knn_features.csv',
    output_dir='../outputs/knn'
)"""),
        md_cell("""### 2. Hyperparameter Exploration Results"""),
        code_cell("""df_tuning.sort_values(by='Macro_F1', ascending=False).head(10)"""),
        md_cell("""### 3. Classification Report & Confusion Matrix"""),
        code_cell("""df_rep = pd.read_csv('../outputs/knn/knn_classification_report.csv', index_col=0)
print("Classification Report:")
print(df_rep)

df_cm = pd.read_csv('../outputs/knn/knn_confusion_matrix.csv', index_col=0)
print("\nConfusion Matrix:")
print(df_cm)"""),
        md_cell("""### 4. Visualizing Model Performance vs. K and Confusion Matrix"""),
        code_cell("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
img1 = mpimg.imread('../outputs/knn/knn_performance_vs_k.png')
img2 = mpimg.imread('../outputs/knn/knn_confusion_matrix.png')
axes[0].imshow(img1)
axes[0].axis('off')
axes[1].imshow(img2)
axes[1].axis('off')
plt.tight_layout()
plt.show()""")
    ]
    with open(os.path.join(nb_dir, "05_KNN_Classification.ipynb"), "w") as f:
        json.dump(make_notebook(nb05_cells), f, indent=2)
    print("Generated 05_KNN_Classification.ipynb")

    # -------------------------------------------------------------
    # Notebook 06: AGNES Clustering
    # -------------------------------------------------------------
    nb06_cells = [
        md_cell("""# 06. Hierarchical Clustering — AGNES (Agglomerative Nesting)
## ASBD Lab Exam — Analytics and Systems of Big Data

---
### 1. Theory & Methodology
AGNES is a bottom-up hierarchical clustering algorithm:
1. Every observation starts as an individual singleton cluster ($k=N$).
2. Pairwise distances between all clusters are calculated.
3. The two closest clusters are iteratively merged according to a linkage criterion:
   - **Single Linkage:** $d(A, B) = \\min \\{d(x, y) : x \\in A, y \\in B\\}$
   - **Complete Linkage:** $d(A, B) = \\max \\{d(x, y) : x \\in A, y \\in B\\}$
   - **Average Linkage:** $d(A, B) = \\frac{1}{|A||B|} \\sum d(x, y)$
   - **Ward Linkage:** Minimizes total within-cluster variance increase.
4. The **Cophenetic Correlation Coefficient** measures how accurately the dendrogram preserves original pairwise dissimilarity.

**Computational Feasibility:**  
Pairwise distance matrix for $N=18,182$ requires $\\approx 165.3$ million distances ($O(N^2)$ memory). We utilize a statistically sound representative sample of $N=2,000$ to ensure numerical stability and memory safety."""),
        code_cell("""import sys
sys.path.append('..')
from src.agnes import run_agnes_clustering
import pandas as pd

agnes_model, df_sample = run_agnes_clustering(
    features_csv='../outputs/cleaned_data/agnes_features.csv',
    output_dir='../outputs/clustering',
    sample_size=2000
)"""),
        md_cell("""### 2. Linkage Cophenetic Correlation Comparison"""),
        code_cell("""df_coph = pd.read_csv('../outputs/clustering/linkage_cophenetic_scores.csv')
df_coph"""),
        md_cell("""### 3. Cluster Profiles and Characteristics"""),
        code_cell("""df_profiles = pd.read_csv('../outputs/clustering/agnes_cluster_profiles.csv', header=[0, 1], index_col=0)
df_profiles"""),
        md_cell("""### 4. Dendrogram and PCA Cluster Visualizations"""),
        code_cell("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 1, figsize=(14, 14), dpi=150)
img_dendro = mpimg.imread('../outputs/clustering/agnes_dendrogram_ward.png')
img_pca = mpimg.imread('../outputs/clustering/agnes_pca_cluster_scatter.png')
axes[0].imshow(img_dendro)
axes[0].axis('off')
axes[1].imshow(img_pca)
axes[1].axis('off')
plt.tight_layout()
plt.show()""")
    ]
    with open(os.path.join(nb_dir, "06_AGNES_Clustering.ipynb"), "w") as f:
        json.dump(make_notebook(nb06_cells), f, indent=2)
    print("Generated 06_AGNES_Clustering.ipynb")

    # -------------------------------------------------------------
    # Notebook 07: ACLOSE Closed Itemsets
    # -------------------------------------------------------------
    nb07_cells = [
        md_cell("""# 07. Closed Frequent Itemsets — ACLOSE Algorithm Implementation
## ASBD Lab Exam — Analytics and Systems of Big Data

---
### 1. Theory: Frequent Itemsets vs. Closed Frequent Itemsets (CFIs)
- An itemset $X$ is **Frequent** if $\\text{support}(X) \\ge \\text{min\\_support}$.
- An itemset $X$ is **Closed** if there exists **NO proper superset** $Y \\supset X$ such that $\\text{support}(Y) = \\text{support}(X)$.
- **Closure Property:** Let $c(X) = g(f(X))$ be the Galois closure operator. $X$ is closed iff $c(X) = X$.
- **Redundancy Reduction:** Ordinary frequent itemset mining generates vast redundant sub-itemsets that carry identical support count. CFIs form a **lossless compact representation**: any frequent itemset's support can be derived from the unique maximal closed itemset containing it!

### 2. ACLOSE Algorithmic Architecture (Pasquier et al., 1999)
- **Phase 1:** Generates minimal generators level-by-level using Apriori join and generator pruning (pruning candidates that have the same support as a sub-generator).
- **Phase 2:** Computes closures of discovered generators via vertical tid-list intersections.
- **Verification:** Verified against an exhaustive brute-force closed-itemset checker on a sample dataset."""),
        code_cell("""import sys
sys.path.append('..')
from src.aclose import run_aclose_analysis
import pandas as pd

df_cfi, df_comp = run_aclose_analysis(
    arm_csv='../outputs/cleaned_data/arm_transactions.csv',
    output_dir='../outputs/closed_itemsets'
)"""),
        md_cell("""### 3. Comparison of Frequent Itemsets (FI) vs. Closed Frequent Itemsets (CFI)"""),
        code_cell("""df_comp = pd.read_csv('../outputs/closed_itemsets/fi_vs_cfi_comparison.csv')
df_comp"""),
        md_cell("""### 4. Discovered Closed Frequent Itemsets (Sample)"""),
        code_cell("""df_cfi = pd.read_csv('../outputs/closed_itemsets/closed_frequent_itemsets.csv')
df_cfi.head(15)"""),
        md_cell("""### 5. Redundancy Reduction Visualizations"""),
        code_cell("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
img1 = mpimg.imread('../outputs/closed_itemsets/aclose_fi_vs_cfi_comparison.png')
img2 = mpimg.imread('../outputs/closed_itemsets/aclose_redundancy_reduction_curve.png')
axes[0].imshow(img1)
axes[0].axis('off')
axes[1].imshow(img2)
axes[1].axis('off')
plt.tight_layout()
plt.show()""")
    ]
    with open(os.path.join(nb_dir, "07_ACLOSE_Closed_Itemsets.ipynb"), "w") as f:
        json.dump(make_notebook(nb07_cells), f, indent=2)
    print("Generated 07_ACLOSE_Closed_Itemsets.ipynb")

    # -------------------------------------------------------------
    # Notebook 08: PySpark Analytics
    # -------------------------------------------------------------
    nb08_cells = [
        md_cell("""# 08. Big Data Analytics Pipeline with PySpark
## ASBD Lab Exam — Analytics and Systems of Big Data

---
### 1. Objective
Implement equivalent and distributed analytics using PySpark:
- SparkSession initialization and distributed DataFrame loading with multiLine CSV handling.
- Spark SQL schema enforcement, type casting, missing value imputation, and feature derivations.
- Spark SQL GroupBy aggregations for recipe performance and monthly trends.
- Spark MLlib distributed FP-Growth implementation.
- Spark ML VectorAssembler and StandardScaler feature engineering pipelines.
- Discussion on distributed scalability and architectural limitations for KNN and AGNES."""),
        code_cell("""import sys
sys.path.append('..')
from src.spark_analytics import run_spark_analytics
import pandas as pd

# Execute complete PySpark analytics workflow
run_spark_analytics(raw_csv='../dataset/dataset.csv', output_dir='../outputs/pyspark')"""),
        md_cell("""### 2. PySpark GroupBy Aggregations Inspection"""),
        code_cell("""df_sp_recipes = pd.read_csv('../outputs/pyspark/spark_recipe_performance_summary.csv')
df_sp_recipes.head(10)"""),
        md_cell("""### 3. PySpark MLlib FP-Growth Output
Frequent itemsets and association rules extracted using Spark MLlib."""),
        code_cell("""df_sp_fi = pd.read_csv('../outputs/pyspark/spark_frequent_itemsets.csv')
print(f"Total PySpark Frequent Itemsets: {len(df_sp_fi)}")
print(df_sp_fi.head(8))

df_sp_rules = pd.read_csv('../outputs/pyspark/spark_association_rules.csv')
print(f"\nTotal PySpark Association Rules: {len(df_sp_rules)}")
print(df_sp_rules.head(8))"""),
        md_cell("""### 4. Distributed Algorithmic Considerations (Exam Discussion)
1. **FP-Growth:** Natively supported in `pyspark.ml.fpm.FPGrowth` using parallel FP-Growth (PFP), which partitions items across worker nodes.
2. **KNN in Spark:** MLlib lacks a native KNN classifier because KNN requires computing global pairwise distances between every test point and every distributed training vector ($O(M \\cdot N)$), causing severe cross-partition data shuffling. In production Spark, Locality-Sensitive Hashing (LSH) or tree-based nearest neighbors are used instead.
3. **AGNES in Spark:** Agglomerative clustering is inherently sequential ($O(N^3)$), requiring global coordination after every pair merge. Spark MLlib instead provides `BisectingKMeans` (hierarchical divisive clustering), which splits clusters top-down in a distributed manner.""")
    ]
    with open(os.path.join(nb_dir, "08_PySpark_Analytics.ipynb"), "w") as f:
        json.dump(make_notebook(nb08_cells), f, indent=2)
    print("Generated 08_PySpark_Analytics.ipynb")

    # -------------------------------------------------------------
    # Notebook 09: Performance Benchmarking
    # -------------------------------------------------------------
    nb09_cells = [
        md_cell("""# 09. Performance Benchmarking: Conventional Python (Pandas) vs. PySpark
## ASBD Lab Exam — Analytics and Systems of Big Data

---
### 1. Objective and Benchmarking Methodology
Empirically benchmark execution times and scaling characteristics between single-threaded Python (Pandas/mlxtend) and distributed PySpark across:
1. **Data Loading:** Reading CSV files into DataFrames.
2. **Data Cleaning & Feature Engineering:** Imputation, length extraction, engagement calculation.
3. **GroupBy Aggregations:** Recipe performance summaries.
4. **FP-Growth Mining:** Frequent pattern mining at min_support=0.05.

**Scales Evaluated:** 10% (1,818 rows), 25% (4,545 rows), 50% (9,091 rows), 100% (18,182 rows).  
**Speedup Metric:**  
$$\\text{Speedup} = \\frac{\\text{Execution Time}_{\\text{Python}}}{\\text{Execution Time}_{\\text{PySpark}}}$$"""),
        code_cell("""import sys
sys.path.append('..')
from src.benchmark import benchmark_pipeline
import pandas as pd

# Execute empirical benchmarking suite
df_bench = benchmark_pipeline(raw_csv='../dataset/dataset.csv', output_dir='../outputs/benchmarks')
df_bench"""),
        md_cell("""### 2. Speedup Analysis Table"""),
        code_cell("""df_bench[['Scale_Pct', 'Records', 'Speedup_Load', 'Speedup_Clean', 'Speedup_Agg', 'Speedup_FPGrowth']]"""),
        md_cell("""### 3. Scaling Curves & Speedup Visualizations"""),
        code_cell("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
img1 = mpimg.imread('../outputs/benchmarks/benchmark_execution_times.png')
img2 = mpimg.imread('../outputs/benchmarks/benchmark_speedup_ratios.png')
axes[0].imshow(img1)
axes[0].axis('off')
axes[1].imshow(img2)
axes[1].axis('off')
plt.tight_layout()
plt.show()"""),
        md_cell("""### 4. Technical Conclusion on Big Data Tool Selection
- **The Small-to-Medium Data Paradox:** For datasets fitting comfortably in RAM (< 1–2 GB, e.g. 18,182 rows / 6 MB), Pandas outperforms PySpark because Pandas operates directly in native C-arrays without process isolation.
- **Spark Overhead Components:** PySpark incurs JVM invocation latency, Py4J socket IPC serialization, Catalyst query optimization, DAG stage generation, and task scheduling overhead (~100–300 ms fixed floor).
- **The Crossover Point:** PySpark becomes essential and dramatically outperforms Python when data exceeds single-machine RAM (e.g. > 10–50 GB), where Pandas crashes with `MemoryError` while Spark automatically handles out-of-core spilling, partition-level parallel execution, and multi-node cluster scaling.""")
    ]
    with open(os.path.join(nb_dir, "09_Performance_Benchmarking.ipynb"), "w") as f:
        json.dump(make_notebook(nb09_cells), f, indent=2)
    print("Generated 09_Performance_Benchmarking.ipynb")

    print("\nAll 9 Jupyter Notebooks generated successfully in notebooks/!")

if __name__ == '__main__':
    generate_notebooks()
