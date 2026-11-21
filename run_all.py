"""
Master Execution Pipeline for ASBD Lab Project
Executes all preprocessing, descriptive analytics, data mining, PySpark, and benchmarking stages,
verifying all outputs, plots, tables, and reports.
"""

import os
import sys
import time

def run_master_pipeline():
    start_total = time.time()
    print("=" * 70)
    print("ASBD LAB EXAM — MASTER ANALYTICS & BIG DATA PIPELINE")
    print("=" * 70)

    stages = [
        ("STAGE 1: Data Preprocessing & Cleaning", "src/preprocessing.py"),
        ("STAGE 2: Descriptive Statistics Generation", "src/descriptive_stats.py"),
        ("STAGE 3: 14 Unique Plot Visualizations", "src/visualization.py"),
        ("STAGE 4: FP-Growth Association Rule Mining", "src/fp_growth.py"),
        ("STAGE 5: KNN Supervised Classification", "src/knn.py"),
        ("STAGE 6: AGNES Hierarchical Clustering", "src/agnes.py"),
        ("STAGE 7: ACLOSE Closed Itemset Mining", "src/aclose.py"),
        ("STAGE 8: PySpark Big Data Analytics", "src/spark_analytics.py"),
        ("STAGE 9: Performance Benchmarking (Pandas vs PySpark)", "src/benchmark.py")
    ]

    for title, script in stages:
        print("\n" + "-" * 70)
        print(f"Executing {title} ({script})...")
        print("-" * 70)
        t0 = time.time()
        ret = os.system(f"{sys.executable} {script}")
        elapsed = time.time() - t0
        if ret != 0:
            print(f"[ERROR] Stage failed with exit code {ret}: {script}")
            sys.exit(ret)
        else:
            print(f"[SUCCESS] {title} completed in {elapsed:.2f} seconds.")

    print("\n" + "=" * 70)
    print("VERIFYING GENERATED OUTPUT ARTIFACTS")
    print("=" * 70)

    critical_files = [
        # Cleaned Data
        "outputs/cleaned_data/cleaned_dataset.csv",
        "outputs/cleaned_data/arm_transactions.csv",
        "outputs/cleaned_data/knn_features.csv",
        "outputs/cleaned_data/agnes_features.csv",
        # Descriptive Statistics
        "outputs/descriptive_statistics/numerical_summary_statistics.csv",
        "outputs/descriptive_statistics/correlation_pearson.csv",
        "outputs/descriptive_statistics/temporal_monthly_trends.csv",
        "outputs/descriptive_statistics/dataset_profile.json",
        # Visualizations (at least 10 required, 14 provided)
        "outputs/visualizations/01_histogram_stars_kde.png",
        "outputs/visualizations/02_boxplot_review_length_by_rating.png",
        "outputs/visualizations/03_violinplot_reputation_by_rating.png",
        "outputs/visualizations/04_barchart_top_recipes.png",
        "outputs/visualizations/05_pareto_chart_recipes.png",
        "outputs/visualizations/06_linechart_temporal_trends.png",
        "outputs/visualizations/07_scatterplot_thumbs_vs_bestscore.png",
        "outputs/visualizations/08_bubblechart_recipe_engagement.png",
        "outputs/visualizations/09_heatmap_correlation_matrix.png",
        "outputs/visualizations/10_hexbin_thumbs_density.png",
        "outputs/visualizations/11_wordcloud_review_text.png",
        "outputs/visualizations/12_stacked_bar_rating_by_engagement.png",
        "outputs/visualizations/13_donut_chart_reputation_segments.png",
        "outputs/visualizations/14_network_graph_association_rules.png",
        # Association Rules
        "outputs/association_rules/frequent_itemsets.csv",
        "outputs/association_rules/all_association_rules.csv",
        "outputs/association_rules/top_rules_by_lift.csv",
        "outputs/association_rules/arm_support_vs_confidence.png",
        # KNN
        "outputs/knn/knn_hyperparameter_tuning.csv",
        "outputs/knn/knn_classification_report.csv",
        "outputs/knn/knn_confusion_matrix.csv",
        "outputs/knn/best_knn_model.joblib",
        "outputs/knn/knn_performance_vs_k.png",
        # AGNES
        "outputs/clustering/linkage_cophenetic_scores.csv",
        "outputs/clustering/agnes_cluster_profiles.csv",
        "outputs/clustering/agnes_dendrogram_ward.png",
        "outputs/clustering/agnes_pca_cluster_scatter.png",
        # ACLOSE
        "outputs/closed_itemsets/closed_frequent_itemsets.csv",
        "outputs/closed_itemsets/fi_vs_cfi_comparison.csv",
        "outputs/closed_itemsets/aclose_fi_vs_cfi_comparison.png",
        # PySpark
        "outputs/pyspark/spark_recipe_performance_summary.csv",
        "outputs/pyspark/spark_frequent_itemsets.csv",
        "outputs/pyspark/spark_association_rules.csv",
        "outputs/pyspark/spark_bisecting_kmeans_clusters.csv",
        # Benchmarks
        "outputs/benchmarks/pandas_vs_pyspark_benchmark.csv",
        "outputs/benchmarks/benchmark_execution_times.png",
        "outputs/benchmarks/benchmark_speedup_ratios.png",
        # Notebooks
        "notebooks/01_Data_Understanding.ipynb",
        "notebooks/02_Data_Preprocessing.ipynb",
        "notebooks/03_Descriptive_Analytics.ipynb",
        "notebooks/04_FP_Growth_ARM.ipynb",
        "notebooks/05_KNN_Classification.ipynb",
        "notebooks/06_AGNES_Clustering.ipynb",
        "notebooks/07_ACLOSE_Closed_Itemsets.ipynb",
        "notebooks/08_PySpark_Analytics.ipynb",
        "notebooks/09_Performance_Benchmarking.ipynb"
    ]

    all_found = True
    for item in critical_files:
        exists = os.path.exists(item)
        size_kb = os.path.getsize(item) / 1024 if exists else 0
        status = f"EXISTS ({size_kb:.1f} KB)" if exists else "MISSING"
        print(f"[{'PASS' if exists else 'FAIL'}] {item:<65} : {status}")
        if not exists:
            all_found = False

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    if all_found:
        print(f"[STATUS: ALL VERIFIED] Pipeline completed in {total_time:.2f} seconds.")
    else:
        print(f"[STATUS: WARNING] Some files were missing.")
    print("=" * 70)

if __name__ == '__main__':
    run_master_pipeline()
