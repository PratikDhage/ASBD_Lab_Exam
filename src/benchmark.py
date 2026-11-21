"""
Performance Benchmarking: Pandas vs. PySpark
ASBD Lab Exam Project
"""

import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth as pd_fpgrowth

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.fpm import FPGrowth as SparkFPGrowth

def get_bench_spark():
    spark = SparkSession.builder \
        .appName("BenchmarkSpark") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def benchmark_pipeline(raw_csv='dataset/dataset.csv', output_dir='outputs/benchmarks'):
    os.makedirs(output_dir, exist_ok=True)
    spark = get_bench_spark()
    print("Beginning Pandas vs. PySpark Benchmark Suite...")

    # Load base dataframe to generate sample partitions
    df_full = pd.read_csv(raw_csv)
    sample_fractions = [0.10, 0.25, 0.50, 1.00]
    benchmark_records = []

    # Temporary directory for partitioned sample CSVs
    temp_dir = 'outputs/benchmarks/temp_samples'
    os.makedirs(temp_dir, exist_ok=True)

    for frac in sample_fractions:
        n_records = int(len(df_full) * frac)
        df_sample = df_full.sample(n=n_records, random_state=42).copy()
        sample_path = os.path.join(temp_dir, f'sample_{int(frac*100)}.csv')
        df_sample.to_csv(sample_path, index=False)
        print(f"\n=======================================================")
        print(f"BENCHMARKING SCALE: {int(frac*100)}% ({n_records} rows)")
        print(f"=======================================================")

        # ---------------------------------------------------------
        # TASK 1: Data Loading (Read CSV)
        # ---------------------------------------------------------
        # Pandas Loading
        t0 = time.time()
        for _ in range(2):
            df_pd = pd.read_csv(sample_path)
        t_pd_load = (time.time() - t0) / 2.0

        # PySpark Loading
        t0 = time.time()
        for _ in range(2):
            df_sp = spark.read.option("header", "true").option("multiLine", "true").option("escape", "\"").option("inferSchema", "true").csv(sample_path)
            _ = df_sp.count() # Force action
        t_sp_load = (time.time() - t0) / 2.0

        speedup_load = t_pd_load / t_sp_load if t_sp_load > 0 else 1.0
        print(f"Task 1 [Load CSV]      -> Pandas: {t_pd_load:.4f}s | PySpark: {t_sp_load:.4f}s | Speedup (Pandas/Spark): {speedup_load:.2f}x")

        # ---------------------------------------------------------
        # TASK 2: Data Cleaning & Feature Engineering
        # ---------------------------------------------------------
        # Pandas Cleaning
        t0 = time.time()
        for _ in range(2):
            df_p = df_pd.copy()
            df_p['text'] = df_p['text'].fillna('')
            df_p['text_len'] = df_p['text'].str.len()
            df_p['engagement'] = df_p['thumbs_up'] + df_p['thumbs_down'] + df_p['reply_count']
            df_p['rating_cat'] = pd.cut(df_p['stars'], bins=[-1, 0, 2, 4, 5], labels=['Zero', 'Low', 'Med', 'High'])
        t_pd_clean = (time.time() - t0) / 2.0

        # PySpark Cleaning
        t0 = time.time()
        for _ in range(2):
            df_s = df_sp.fillna({"text": ""}) \
                        .withColumn("text_len", F.length("text")) \
                        .withColumn("engagement", F.col("thumbs_up") + F.col("thumbs_down") + F.col("reply_count")) \
                        .withColumn("rating_cat",
                            F.when(F.col("stars") == 0, "Zero")
                             .when(F.col("stars") <= 2, "Low")
                             .when(F.col("stars") <= 4, "Med")
                             .otherwise("High"))
            _ = df_s.count()
        t_sp_clean = (time.time() - t0) / 2.0

        speedup_clean = t_pd_clean / t_sp_clean if t_sp_clean > 0 else 1.0
        print(f"Task 2 [Cleaning]      -> Pandas: {t_pd_clean:.4f}s | PySpark: {t_sp_clean:.4f}s | Speedup: {speedup_clean:.2f}x")

        # ---------------------------------------------------------
        # TASK 3: GroupBy Aggregation (Recipe Performance)
        # ---------------------------------------------------------
        # Pandas GroupBy
        t0 = time.time()
        for _ in range(3):
            agg_pd = df_p.groupby('recipe_code').agg(
                review_count=('comment_id', 'count'),
                avg_rating=('stars', 'mean'),
                total_thumbs=('thumbs_up', 'sum'),
                avg_text_len=('text_len', 'mean')
            ).reset_index()
        t_pd_agg = (time.time() - t0) / 3.0

        # PySpark GroupBy
        t0 = time.time()
        for _ in range(3):
            agg_sp = df_s.groupBy("recipe_code").agg(
                F.count("comment_id").alias("review_count"),
                F.avg("stars").alias("avg_rating"),
                F.sum("thumbs_up").alias("total_thumbs"),
                F.avg("text_len").alias("avg_text_len")
            )
            _ = agg_sp.count()
        t_sp_agg = (time.time() - t0) / 3.0

        speedup_agg = t_pd_agg / t_sp_agg if t_sp_agg > 0 else 1.0
        print(f"Task 3 [Aggregation]   -> Pandas: {t_pd_agg:.4f}s | PySpark: {t_sp_agg:.4f}s | Speedup: {speedup_agg:.2f}x")

        # ---------------------------------------------------------
        # TASK 4: FP-Growth Association Mining (minSupport = 0.05)
        # ---------------------------------------------------------
        # Prepare transaction basket
        trans_pd = df_p[['rating_cat']].astype(str).values.tolist()
        t0 = time.time()
        te = TransactionEncoder()
        te_ary = te.fit(trans_pd).transform(trans_pd)
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_)
        fi_pd = pd_fpgrowth(encoded_df, min_support=0.05, use_colnames=True)
        t_pd_fpg = time.time() - t0

        # Spark FP-Growth
        df_sp_fpg = df_s.select(F.array(F.col("rating_cat").cast("string")).alias("items"))
        t0 = time.time()
        spark_fpg = SparkFPGrowth(itemsCol="items", minSupport=0.05, minConfidence=0.50)
        sp_fpg_model = spark_fpg.fit(df_sp_fpg)
        _ = sp_fpg_model.freqItemsets.count()
        t_sp_fpg = time.time() - t0

        speedup_fpg = t_pd_fpg / t_sp_fpg if t_sp_fpg > 0 else 1.0
        print(f"Task 4 [FP-Growth]     -> Pandas: {t_pd_fpg:.4f}s | PySpark: {t_sp_fpg:.4f}s | Speedup: {speedup_fpg:.2f}x")

        benchmark_records.append({
            'Scale_Pct': int(frac * 100),
            'Records': n_records,
            'Pandas_Load_s': round(t_pd_load, 4),
            'PySpark_Load_s': round(t_sp_load, 4),
            'Speedup_Load': round(speedup_load, 2),
            'Pandas_Clean_s': round(t_pd_clean, 4),
            'PySpark_Clean_s': round(t_sp_clean, 4),
            'Speedup_Clean': round(speedup_clean, 2),
            'Pandas_Agg_s': round(t_pd_agg, 4),
            'PySpark_Agg_s': round(t_sp_agg, 4),
            'Speedup_Agg': round(speedup_agg, 2),
            'Pandas_FPGrowth_s': round(t_pd_fpg, 4),
            'PySpark_FPGrowth_s': round(t_sp_fpg, 4),
            'Speedup_FPGrowth': round(speedup_fpg, 2)
        })

    spark.stop()

    df_bench = pd.DataFrame(benchmark_records)
    bench_csv = os.path.join(output_dir, 'pandas_vs_pyspark_benchmark.csv')
    df_bench.to_csv(bench_csv, index=False)
    print(f"\nSaved comprehensive benchmark table to {bench_csv}")

    # Visualizations: Performance Scaling Plots
    # Plot 1: Execution Time across Scales for Aggregation & Cleaning
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(df_bench['Scale_Pct'], df_bench['Pandas_Agg_s'], marker='o', linewidth=2.2, color='#1f77b4', label='Pandas Aggregation')
    plt.plot(df_bench['Scale_Pct'], df_bench['PySpark_Agg_s'], marker='s', linewidth=2.2, color='#ff7f0e', linestyle='--', label='PySpark Aggregation')
    plt.plot(df_bench['Scale_Pct'], df_bench['Pandas_Clean_s'], marker='^', linewidth=2.2, color='#2ca02c', label='Pandas Cleaning')
    plt.plot(df_bench['Scale_Pct'], df_bench['PySpark_Clean_s'], marker='d', linewidth=2.2, color='#d62728', linestyle='--', label='PySpark Cleaning')
    plt.title('Benchmark: Execution Time vs. Dataset Scale (Pandas vs. PySpark)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Dataset Scale Percentage (%)', fontsize=11)
    plt.ylabel('Execution Time (Seconds)', fontsize=11)
    plt.xticks(df_bench['Scale_Pct'], [f"{p}% ({r} rows)" for p, r in zip(df_bench['Scale_Pct'], df_bench['Records'])])
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_execution_times.png'))
    plt.close()

    # Plot 2: Speedup Ratio (Pandas Time / PySpark Time)
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(df_bench['Scale_Pct'], df_bench['Speedup_Agg'], marker='o', linewidth=2.5, color='#8e44ad', label='Aggregation Speedup')
    plt.plot(df_bench['Scale_Pct'], df_bench['Speedup_Clean'], marker='s', linewidth=2.5, color='#2980b9', label='Cleaning Speedup')
    plt.plot(df_bench['Scale_Pct'], df_bench['Speedup_Load'], marker='^', linewidth=2.5, color='#27ae60', label='CSV Loading Speedup')
    plt.axhline(1.0, color='red', linestyle=':', linewidth=1.5, label='Parity Threshold (Speedup=1.0)')
    plt.title('Speedup Ratio (Pandas Time / PySpark Time) across Dataset Scales', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Dataset Scale Percentage (%)', fontsize=11)
    plt.ylabel('Speedup Ratio (>1 means Spark is slower; <1 means Spark is faster)', fontsize=10)
    plt.xticks(df_bench['Scale_Pct'])
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_speedup_ratios.png'))
    plt.close()
    print("Saved benchmark visualization plots.")

    # Cleanup temp sample files
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    return df_bench

if __name__ == '__main__':
    benchmark_pipeline()
