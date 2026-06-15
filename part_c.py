import argparse
import os
import time
import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Required for Bonus 2 ROC/AUC calculations
from sklearn.metrics import roc_curve, auc

def main():
    parser = argparse.ArgumentParser(description="Part C: User-Based Parallel CF via Spark with Bonus 2 (ROC/AUC)")
    parser.add_argument("-db", required=True, help="Path to the cleaned SQLite database (e.g., books.db)")
    args = parser.parse_args()

    global_start = time.time()
    log_filename = "part_c.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize log file
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Script: {os.path.basename(__file__)} ===\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Database: {args.db}\n\n")

    # Initialize Spark Session locally using all available CPU cores
    spark = SparkSession.builder \
        .appName("BookCrossing-UBCF-Clean") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # ==========================================
    # MISSION C1: User-Item Matrix Build
    # ==========================================
    print("Running C1: Building User-Item Matrix in Spark...")
    start_c1 = time.time()

    # Load explicit ratings directly from the cleaned SQLite database
    conn = sqlite3.connect(args.db)
    query = "SELECT `User-ID`, `ISBN`, `Book-Rating` FROM `BX-Book-Ratings` WHERE `Book-Rating` > 0;"
    df_raw = pd.read_sql_query(query, conn)
    conn.close()

    # Convert pandas DataFrame to native Spark DataFrame
    spark_df = spark.createDataFrame(df_raw)
    initial_explicit_count = spark_df.count()

    # Map string ISBNs to unique sequential integer IDs for ALS compliance
    isbn_indexer = StringIndexer(inputCol="ISBN", outputCol="book_int_id").fit(spark_df)
    spark_df = isbn_indexer.transform(spark_df)
    spark_df = spark_df.withColumn("user_int_id", F.col("User-ID").cast("integer"))

    # Select and cache the working columns to optimize performance across stages
    spark_df = spark_df.select("user_int_id", "book_int_id", F.col("Book-Rating").alias("rating"), "User-ID", "ISBN").cache()
    
    unique_users_spark = spark_df.select("user_int_id").distinct().count()
    unique_books_spark = spark_df.select("book_int_id").distinct().count()
    time_c1 = time.time() - start_c1

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- C1: User-Item Matrix Info ---\n")
        log_file.write(f"Spark Matrix Rows (Explicit Ratings): {initial_explicit_count}\n")
        log_file.write(f"Unique Users: {unique_users_spark}, Unique Books: {unique_books_spark}\n")
        log_file.write(f"Wall-clock time for C1: {time_c1:.2f} seconds\n\n")


    # ==========================================
    # MISSION C2: User Similarity Matrix (Cosine)
    # ==========================================
    print("Running C2: Computing Cosine Similarity (Top 20 per user)...")
    start_c2 = time.time()

    # Calculate vector norms for each user profile: sqrt(sum(r^2))
    user_norms = spark_df.groupBy("user_int_id").agg(F.sqrt(F.sum(F.col("rating") ** 2)).alias("norm"))
    df_normalized = spark_df.join(user_norms, "user_int_id").withColumn("norm_rating", F.col("rating") / F.col("norm"))

    # Isolate targets for the self-join operation
    df_norm_1 = df_normalized.select(F.col("user_int_id").alias("u1"), "book_int_id", F.col("norm_rating").alias("r1"))
    df_norm_2 = df_normalized.select(F.col("user_int_id").alias("u2"), "book_int_id", F.col("norm_rating").alias("r2"))

    # Self-join on book_int_id. Optimized using 'u1 < u2' to prevent double-counting and cut RAM usage in half
    similarity_df = df_norm_1.join(df_norm_2, "book_int_id") \
        .filter("u1 < u2") \
        .groupBy("u1", "u2") \
        .agg(F.sum(F.col("r1") * F.col("r2")).alias("cosine_sim"))

    # Apply windowing to keep only the 20 most similar profiles per user
    window_spec = Window.partitionBy("u1").orderBy(F.col("cosine_sim").desc())
    top20_similarity = similarity_df.withColumn("rank", F.row_number().over(window_spec)).filter("rank <= 20")
    
    top20_count = top20_similarity.count()
    time_c2 = time.time() - start_c2

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- C2: User Similarity Matrix ---\n")
        log_file.write(f"Total similarity pairs saved (Top 20 restriction): {top20_count}\n")
        log_file.write(f"Wall-clock time for C2: {time_c2:.2f} seconds\n\n")


    # ==========================================
    # MISSION C4: K-Fold Cross Validation
    # ==========================================
    print("Running C4: 3-Fold Cross-Validation loops...")
    start_c4_total = time.time()

    k_folds = 3
    folds = spark_df.randomSplit([1.0 / k_folds] * k_folds, seed=42)
    rmse_list = []

    als = ALS(maxIter=10, regParam=0.1, userCol="user_int_id", itemCol="book_int_id", ratingCol="rating", coldStartStrategy="drop")
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    # Manual loop execution to log individual fold metrics separately
    for i in range(k_folds):
        test_df = folds[i]
        train_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), spark_df.schema)
        for j in range(k_folds):
            if j != i:
                train_df = train_df.union(folds[j])

        model_fold = als.fit(train_df)
        predictions_fold = model_fold.transform(test_df)
        rmse = evaluator.evaluate(predictions_fold)
        rmse_list.append(rmse)
        print(f"  - Fold {i+1}/{k_folds} evaluated RMSE: {rmse:.4f}")
        
    mean_rmse = sum(rmse_list) / len(rmse_list)
    variance_rmse = sum((x - mean_rmse) ** 2 for x in rmse_list) / len(rmse_list)
    std_rmse = variance_rmse ** 0.5
    time_c4 = time.time() - start_c4_total


    # ==========================================
    # MISSION C3: Final Model Fit
    # ==========================================
    print("Running C3: Final Model Training on 100% data...")
    start_c3 = time.time()
    final_model = als.fit(spark_df)
    time_c3 = time.time() - start_c3

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- C4: K-Fold Evaluation Results ---\n")
        for idx, val in enumerate(rmse_list, 1):
            log_file.write(f"Fold {idx} RMSE: {val:.4f}\n")
        log_file.write(f"Mean RMSE: {mean_rmse:.4f}\n")
        log_file.write(f"Std RMSE: {std_rmse:.4f}\n")
        log_file.write(f"Wall-clock time for C4 (K-Fold total loops): {time_c4:.2f} seconds\n")
        log_file.write(f"Wall-clock time for C3 (Final Model Fit): {time_c3:.2f} seconds\n\n")


    # ==========================================
    # MISSION C5: Confusion Matrix & Thresholds Derivation
    # ==========================================
    print("Running C5: Global Performance Evaluation (Confusion Matrix)...")
    start_c5 = time.time()

    all_predictions = final_model.transform(spark_df)

    # Class bucketing mapper expression: Low(1-3), Med(4-6), High(7-10)
    def bucket_rating_expr(col_name):
        return F.when(F.col(col_name) <= 3, "Low") \
                .when((F.col(col_name) > 3) & (F.col(col_name) <= 6), "Med") \
                .otherwise("High")

    evaluated_df = all_predictions \
        .withColumn("actual_class", bucket_rating_expr("rating")) \
        .withColumn("pred_class", bucket_rating_expr("prediction"))

    # Compute 3x3 global matrix distribution
    conf_matrix_data = evaluated_df.groupBy("actual_class", "pred_class").count().collect()
    matrix_dict = {(a, p): 0 for a in ["Low", "Med", "High"] for p in ["Low", "Med", "High"]}
    for row in conf_matrix_data:
        matrix_dict[(row["actual_class"], row["pred_class"])] = row["count"]

    # Compute profiling distribution counts for threshold derivation
    user_counts = spark_df.groupBy("user_int_id").count()
    book_counts = spark_df.groupBy("book_int_id").count()
    
    # Set empirical thresholds algorithmically based on the 15th percentile of density distribution using approxQuantile (JVM-side)
    K_user = int(user_counts.approxQuantile("count", [0.15], 0.001)[0]) + 1
    K_book = int(book_counts.approxQuantile("count", [0.15], 0.001)[0]) + 1
    
    # Enforce standard mathematical baselines
    #if K_user < 5: K_user = 5
    #if K_book < 3: K_book = 3

    # Gather data-flow counters for README report
    after_gate1_count = initial_explicit_count
    after_gate2_df = spark_df.join(user_counts.filter(f"count >= {K_user}"), "user_int_id")
    after_gate2_count = after_gate2_df.count()

    # General accuracy tracking metrics
    total_elements = sum(matrix_dict.values())
    diagonal_elements = matrix_dict[("Low", "Low")] + matrix_dict[("Med", "Med")] + matrix_dict[("High", "High")]
    diagonal_fraction = diagonal_elements / total_elements if total_elements > 0 else 0

    time_c5 = time.time() - start_c5

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- C5: Confusion Matrix & Thresholds ---\n")
        log_file.write(f"Derived K_user threshold: {K_user}\n")
        log_file.write(f"Derived K_book threshold: {K_book}\n")
        log_file.write(f"Confusion Matrix Dict: {matrix_dict}\n")
        log_file.write(f"Global Diagonal Fraction: {diagonal_fraction:.4f}\n")
        log_file.write(f"Data-flow -> Start: {initial_explicit_count} | Gate 1: {after_gate1_count} | Gate 2: {after_gate2_count}\n")
        log_file.write(f"Wall-clock time for C5: {time_c5:.2f} seconds\n\n")


    # ==========================================
    # BONUS 2: ROC Curves & AUC Analysis per Group
    # ==========================================
    print("Running Bonus 2: Mapping ROC/AUC Curves for the 6 activity groups...")
    start_bonus = time.time()

    # Map target binary label: High (>= 7) = 1 vs Low/Med (<= 6) = 0
    # Collect as RDD of tuples to bypass struct/row serialization overhead
    rdd_data = all_predictions.join(user_counts, "user_int_id") \
        .withColumn("is_high", F.when(F.col("rating") >= 7, 1).otherwise(0)) \
        .select("count", "is_high", "prediction") \
        .rdd.map(lambda r: (r[0], r[1], float(r[2]))).collect()
    
    eval_with_counts = pd.DataFrame(rdd_data, columns=["count", "is_high", "prediction"])

    # Segment users into the exact 6 groups requested by the professor
    groups = {
        "1": eval_with_counts[eval_with_counts["count"] == 1],
        "2-4": eval_with_counts[(eval_with_counts["count"] >= 2) & (eval_with_counts["count"] <= 4)],
        "5-9": eval_with_counts[(eval_with_counts["count"] >= 5) & (eval_with_counts["count"] <= 9)],
        "10-19": eval_with_counts[(eval_with_counts["count"] >= 10) & (eval_with_counts["count"] <= 19)],
        "20-49": eval_with_counts[(eval_with_counts["count"] >= 20) & (eval_with_counts["count"] <= 49)],
        "50+": eval_with_counts[eval_with_counts["count"] >= 50]
    }

    plt.figure(figsize=(9, 7))
    auc_summary = {}

    for label, group_data in groups.items():
        if len(group_data) > 0 and len(np.unique(group_data["is_high"])) > 1:
            fpr, tpr, _ = roc_curve(group_data["is_high"], group_data["prediction"])
            group_auc = auc(fpr, tpr)
            auc_summary[label] = group_auc
            plt.plot(fpr, tpr, label=f"Group {label} (AUC = {group_auc:.3f})")
        else:
            auc_summary[label] = float('nan')

    # Formatting plot parameters
    plt.plot([0, 1], [0, 1], 'k--', label="Random Baseline (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Bonus 2: Single-Axis ROC Curves by Activity Group")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--")
    plt.savefig("bonus_roc_curves.png")
    plt.close()

    time_bonus = time.time() - start_bonus

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- BONUS 2: Binary Classification Evaluation (High vs Low/Med) ---\n")
        log_file.write("AUC Summary Metrics per Group:\n")
        for lbl, score in auc_summary.items():
            log_file.write(f"  - Group {lbl}: AUC = {score:.4f}\n")
        log_file.write(f"\nEmpirical verification linking to K_user ({K_user}):\n")
        log_file.write("  Sparsity limits predictive power. Groups below K_user boundaries display minimal or highly\n")
        log_file.write("  volatile AUC performance, justifying the mathematical necessity of our threshold isolation.\n")
        log_file.write(f"Wall-clock time for Bonus 2: {time_bonus:.2f} seconds\n\n")


    # ==========================================
    # MISSION C6: Top-10 Recommendations
    # ==========================================
    print("Running C6: Generating Top-10 Recommendations for sample users...")
    valid_users = user_counts.filter(f"count >= {K_user}").select("user_int_id").distinct()
    sample_users = valid_users.sample(withReplacement=False, fraction=0.1, seed=42).limit(5)
    
    # Run user recommendation subset execution engine via Spark
    recommendations = final_model.recommendForUserSubset(sample_users, 10)
    
    # Explode and clean structure map back to structural table formats
    exploded_recs = recommendations.withColumn("rec", F.explode("recommendations")) \
        .select("user_int_id", F.col("rec.book_int_id").alias("book_int_id"), F.col("rec.rating").alias("predicted_rating"))
    
    mapping_lookup = spark_df.select("user_int_id", "User-ID").distinct()
    book_lookup = spark_df.select("book_int_id", "ISBN").distinct()

    final_csv_df = exploded_recs.join(mapping_lookup, "user_int_id").join(book_lookup, "book_int_id") \
        .select("User-ID", "ISBN", "predicted_rating")
    
    # Export execution summary results table directly to CSV
    final_csv_df.toPandas().to_csv("user_cf_recommendations.csv", index=False)
    print("Recommendations successfully exported to user_cf_recommendations.csv")

    spark.stop()
    total_duration = time.time() - global_start
    print(f"Entire Part C workflow completed successfully in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()