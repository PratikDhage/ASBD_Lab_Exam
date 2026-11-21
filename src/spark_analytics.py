"""
PySpark Big Data Analytics Pipeline
ASBD Lab Exam Project
"""

import os
import sys
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, DoubleType, ArrayType
)
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.clustering import BisectingKMeans

def get_spark_session(app_name="RecipeAnalyticsSpark"):
    """Initialize and configure local SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def run_spark_analytics(raw_csv='dataset/dataset.csv', output_dir='outputs/pyspark'):
    os.makedirs(output_dir, exist_ok=True)
    spark = get_spark_session("ASBD_Spark_Analytics")
    print(f"PySpark Session Active. Spark Version: {spark.version}")

    # 1. Spark Data Loading with Schema & MultiLine Handling
    t0 = time.time()
    df_raw = spark.read \
        .option("header", "true") \
        .option("multiLine", "true") \
        .option("escape", "\"") \
        .option("inferSchema", "true") \
        .csv(raw_csv)
    load_time = time.time() - t0
    total_records = df_raw.count()
    print(f"Loaded {total_records} records via PySpark in {load_time:.4f}s.")

    # 2. Spark Data Cleaning and Feature Engineering
    # Drop index if present
    if "Unnamed: 0" in df_raw.columns:
        df_clean = df_raw.drop("Unnamed: 0")
    else:
        df_clean = df_raw

    # Handle missing values & cast types
    df_clean = df_clean.fillna({"text": ""})
    df_clean = df_clean.withColumn("stars", F.col("stars").cast(IntegerType())) \
                       .withColumn("thumbs_up", F.col("thumbs_up").cast(IntegerType())) \
                       .withColumn("thumbs_down", F.col("thumbs_down").cast(IntegerType())) \
                       .withColumn("reply_count", F.col("reply_count").cast(IntegerType())) \
                       .withColumn("user_reputation", F.col("user_reputation").cast(IntegerType()))

    # Temporal feature extraction
    df_clean = df_clean.withColumn("created_timestamp", F.to_timestamp(F.from_unixtime(F.col("created_at")))) \
                       .withColumn("year", F.year("created_timestamp")) \
                       .withColumn("month", F.month("created_timestamp")) \
                       .withColumn("year_month", F.date_format("created_timestamp", "yyyy-MM")) \
                       .withColumn("day_of_week", F.date_format("created_timestamp", "EEEE"))

    # Feature Derivation
    df_clean = df_clean.withColumn("text_length", F.length(F.col("text"))) \
                       .withColumn("engagement_total", F.col("thumbs_up") + F.col("thumbs_down") + F.col("reply_count")) \
                       .withColumn("is_perfect_rating", F.when(F.col("stars") == 5, 1).otherwise(0))

    # Discretization in PySpark
    df_clean = df_clean.withColumn("rating_cat",
        F.when(F.col("stars") == 0, "Rating_0_Unrated")
         .when(F.col("stars") <= 2, "Rating_Low_1_2")
         .when(F.col("stars") <= 4, "Rating_Medium_3_4")
         .otherwise("Rating_High_5")
    ).withColumn("reputation_cat",
        F.when(F.col("user_reputation") == 1, "Rep_Novice")
         .when(F.col("user_reputation") <= 5, "Rep_Intermediate")
         .otherwise("Rep_Veteran")
    ).withColumn("engagement_cat",
        F.when(F.col("engagement_total") == 0, "Engage_Zero")
         .when(F.col("engagement_total") <= 2, "Engage_Low")
         .when(F.col("engagement_total") <= 6, "Engage_Medium")
         .otherwise("Engage_High")
    ).withColumn("length_cat",
        F.when(F.col("text_length") < 80, "Len_Short")
         .when(F.col("text_length") <= 220, "Len_Medium")
         .otherwise("Len_Long")
    ).withColumn("reply_cat",
        F.when(F.col("reply_count") > 0, "Has_Reply").otherwise("No_Reply")
    )

    print("PySpark Data Cleaning & Feature Transformation Complete.")

    # 3. PySpark SQL & DataFrame Aggregations
    print("\n--- Running Spark SQL GroupBy Aggregations ---")
    df_clean.createOrReplaceTempView("reviews_tbl")

    # A) Recipe Performance Summary in Spark SQL
    spark_recipe_summary = spark.sql("""
        SELECT 
            recipe_code,
            recipe_name,
            COUNT(*) AS review_count,
            ROUND(AVG(stars), 3) AS avg_stars,
            ROUND(AVG(is_perfect_rating), 3) AS perfect_rating_ratio,
            SUM(thumbs_up) AS total_thumbs_up,
            ROUND(AVG(engagement_total), 3) AS avg_engagement,
            ROUND(AVG(text_length), 1) AS avg_review_length
        FROM reviews_tbl
        GROUP BY recipe_code, recipe_name
        ORDER BY review_count DESC
    """)
    recipe_pd = spark_recipe_summary.toPandas()
    recipe_pd.to_csv(os.path.join(output_dir, 'spark_recipe_performance_summary.csv'), index=False)
    print("Saved Spark Recipe Performance Aggregation.")

    # B) Monthly Temporal Trends in Spark SQL
    spark_monthly = spark.sql("""
        SELECT 
            year_month,
            COUNT(*) AS review_count,
            ROUND(AVG(stars), 3) AS avg_stars,
            SUM(thumbs_up) AS total_thumbs_up
        FROM reviews_tbl
        GROUP BY year_month
        ORDER BY year_month ASC
    """)
    monthly_pd = spark_monthly.toPandas()
    monthly_pd.to_csv(os.path.join(output_dir, 'spark_monthly_trends.csv'), index=False)
    print("Saved Spark Monthly Trends Aggregation.")

    # C) Rating Breakdown
    spark_ratings = spark.sql("""
        SELECT 
            stars,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
        FROM reviews_tbl
        GROUP BY stars
        ORDER BY stars ASC
    """)
    ratings_pd = spark_ratings.toPandas()
    ratings_pd.to_csv(os.path.join(output_dir, 'spark_rating_distribution.csv'), index=False)

    # 4. PySpark MLlib FP-Growth Implementation
    print("\n--- Running Native PySpark MLlib FP-Growth ---")
    # Assemble categorical items into array<string>
    df_transactions = df_clean.select(
        F.array("rating_cat", "reputation_cat", "engagement_cat", "length_cat", "reply_cat").alias("items")
    )

    fp_growth = FPGrowth(itemsCol="items", minSupport=0.03, minConfidence=0.50)
    t0 = time.time()
    fp_model = fp_growth.fit(df_transactions)
    spark_fpgrowth_time = time.time() - t0
    print(f"PySpark FP-Growth executed in {spark_fpgrowth_time:.4f}s.")

    # Extract frequent itemsets
    spark_freq_itemsets = fp_model.freqItemsets
    fi_count = spark_freq_itemsets.count()
    print(f"PySpark Frequent Itemsets Found: {fi_count}")
    
    fi_pd = spark_freq_itemsets.withColumn("items_str", F.concat_ws(", ", "items")) \
                               .select("items_str", "freq") \
                               .toPandas()
    fi_pd['support_ratio'] = (fi_pd['freq'] / total_records).round(4)
    fi_pd.sort_values(by='freq', ascending=False).to_csv(os.path.join(output_dir, 'spark_frequent_itemsets.csv'), index=False)

    # Extract association rules
    spark_rules = fp_model.associationRules
    rules_count = spark_rules.count()
    print(f"PySpark Association Rules Generated: {rules_count}")

    rules_pd = spark_rules.withColumn("antecedent_str", F.concat_ws(", ", "antecedent")) \
                          .withColumn("consequent_str", F.concat_ws(", ", "consequent")) \
                          .select("antecedent_str", "consequent_str", "confidence", "lift", "support") \
                          .toPandas()
    for col in ['confidence', 'lift', 'support']:
        rules_pd[col] = rules_pd[col].round(4)
    rules_pd.sort_values(by='lift', ascending=False).to_csv(os.path.join(output_dir, 'spark_association_rules.csv'), index=False)
    print("Saved PySpark FP-Growth frequent itemsets and association rules.")

    # 5. PySpark Feature Transformation Pipeline (VectorAssembler & StandardScaler)
    print("\n--- PySpark ML Feature Engineering Pipeline ---")
    assembler = VectorAssembler(
        inputCols=["user_reputation", "thumbs_up", "thumbs_down", "reply_count", "text_length"],
        outputCol="raw_features"
    )
    df_assembled = assembler.transform(df_clean)

    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)

    # 6. Big Data Clustering in Spark: Bisecting k-Means (Hierarchical Divisive Tree in Spark)
    # Spark MLlib provides BisectingKMeans (hierarchical divisive clustering), which scales across cluster nodes,
    # as an alternative to single-machine AGNES (agglomerative)
    print("\n--- PySpark Hierarchical Divisive Clustering (BisectingKMeans) ---")
    bkm = BisectingKMeans(k=4, featuresCol="scaled_features", predictionCol="cluster_id", seed=42)
    bkm_model = bkm.fit(df_scaled)
    df_clustered = bkm_model.transform(df_scaled)
    
    cluster_counts = df_clustered.groupBy("cluster_id").count().toPandas()
    cluster_counts.to_csv(os.path.join(output_dir, 'spark_bisecting_kmeans_clusters.csv'), index=False)
    print("PySpark BisectingKMeans Cluster Sizes:\n", cluster_counts)

    spark.stop()
    print("SparkSession stopped cleanly.")

if __name__ == '__main__':
    run_spark_analytics()
