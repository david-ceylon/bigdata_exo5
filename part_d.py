import argparse
import os
import time
import datetime
import sqlite3
import re
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer

def main():
    parser = argparse.ArgumentParser(description="Part D: Item-Based Parallel Collaborative Filtering via Spark")
    parser.add_argument("-db", required=True, help="Path to the cleaned SQLite database (e.g., books.db)")
    args = parser.parse_args()

    global_start = time.time()
    log_filename = "part_d.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize log file
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Script: {os.path.basename(__file__)} ===\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Database: {args.db}\n\n")

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("BookCrossing-IBCF") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # Load matrix data (Explicit ratings only)
    conn = sqlite3.connect(args.db)
    query = "SELECT `User-ID`, `ISBN`, `Book-Rating` FROM `BX-Book-Ratings` WHERE `Book-Rating` > 0;"
    df_raw = pd.read_sql_query(query, conn)
    conn.close()

    spark_df = spark.createDataFrame(df_raw)
    isbn_indexer = StringIndexer(inputCol="ISBN", outputCol="book_int_id").fit(spark_df)
    spark_df = isbn_indexer.transform(spark_df)
    spark_df = spark_df.withColumn("user_int_id", F.col("User-ID").cast("integer"))
    spark_df = spark_df.select("user_int_id", "book_int_id", F.col("Book-Rating").alias("rating"), "User-ID", "ISBN").cache()

    input_rows = spark_df.count()

    # ==========================================
    # MISSION D1: Item Similarity Matrix
    # ==========================================
    print("Running D1: Computing Item-Item Cosine Similarity...")
    start_d1 = time.time()

    # Compute vector magnitudes (norms) for each book profile
    item_norms = spark_df.groupBy("book_int_id").agg(F.sqrt(F.sum(F.col("rating") ** 2)).alias("norm"))
    df_normalized = spark_df.join(item_norms, "book_int_id").withColumn("norm_rating", F.col("rating") / F.col("norm"))

    df_norm_1 = df_normalized.select(F.col("book_int_id").alias("b1"), "user_int_id", F.col("norm_rating").alias("r1"))
    df_norm_2 = df_normalized.select(F.col("book_int_id").alias("b2"), "user_int_id", F.col("norm_rating").alias("r2"))

    # Compute dot products via parallel join on user_int_id
    similarity_df = df_norm_1.join(df_norm_2, "user_int_id") \
        .filter("b1 < b2") \
        .groupBy("b1", "b2") \
        .agg(F.sum(F.col("r1") * F.col("r2")).alias("cosine_sim"))

    # Make the similarity dataframe bidirectional (symmetric) for prediction mapping
    sim_symmetric = similarity_df.union(
        similarity_df.select(F.col("b2").alias("b1"), F.col("b1").alias("b2"), "cosine_sim")
    )

    # Restrict to the top 20 most similar books for each item
    window_spec = Window.partitionBy("b1").orderBy(F.col("cosine_sim").desc())
    top20_item_similarity = sim_symmetric.withColumn("rank", F.row_number().over(window_spec)) \
        .filter("rank <= 20").cache()
    
    top20_items_count = top20_item_similarity.count()
    time_d1 = time.time() - start_d1


    # ==========================================
    # MISSION D2: Item-Based CF Model Fit
    # ==========================================
    print("Running D2: Fitting Item-Based CF Prediction Engine...")
    start_d2 = time.time()

    # Traditional memory-based IBCF prediction logic mapped over Spark DataFrames
    # Formula: Pred(u, i) = sum(Sim(i, j) * R(u, j)) / sum(Sim(i, j))
    joined_ratings = spark_df.alias("r").join(
        top20_item_similarity.alias("s"), 
        F.col("r.book_int_id") == F.col("s.b2")
    )

    predictions_df = joined_ratings.groupBy("r.user_int_id", "s.b1") \
        .agg((F.sum(F.col("s.cosine_sim") * F.col("r.rating")) / F.sum(F.col("s.cosine_sim"))).alias("prediction")) \
        .withColumnRenamed("b1", "book_int_id").cache()

    # Trigger action to complete the item-based "fit" stage calculation
    prediction_count = predictions_df.count()
    time_d2 = time.time() - start_d2


    # ==========================================
    # MISSION D3: Evaluation & Comparison
    # ==========================================
    print("Running D3: Evaluation via 3x3 Confusion Matrix...")
    start_d3 = time.time()

    # Evaluate matches only across observed cells of the user-item matrix
    evaluated_df = predictions_df.join(spark_df, ["user_int_id", "book_int_id"])

    def bucket_rating_expr(col_name):
        return F.when(F.col(col_name) <= 3, "Low") \
                .when((F.col(col_name) > 3) & (F.col(col_name) <= 6), "Med") \
                .otherwise("High")

    evaluated_df = evaluated_df \
        .withColumn("actual_class", bucket_rating_expr("rating")) \
        .withColumn("pred_class", bucket_rating_expr("prediction"))

    conf_matrix_data = evaluated_df.groupBy("actual_class", "pred_class").count().collect()
    matrix_dict = {(a, p): 0 for a in ["Low", "Med", "High"] for p in ["Low", "Med", "High"]}
    for row in conf_matrix_data:
        matrix_dict[(row["actual_class"], row["pred_class"])] = row["count"]

    total_elements = sum(matrix_dict.values())
    diagonal_elements = matrix_dict[("Low", "Low")] + matrix_dict[("Med", "Med")] + matrix_dict[("High", "High")]
    item_diagonal_fraction = diagonal_elements / total_elements if total_elements > 0 else 0

    # Attempt to read metrics from part_c.log to fill out the side-by-side comparison report
    user_fit_time, user_pred_time, user_diag_fraction = "N/A", "N/A", "N/A"
    if os.path.exists("part_c.log"):
        with open("part_c.log", "r", encoding="utf-8") as c_log:
            c_content = c_log.read()
            fit_match = re.search(r"Wall-clock time for C3.*:\s*([\d.]+)", c_content)
            pred_match = re.search(r"Wall-clock time for C5.*:\s*([\d.]+)", c_content)
            diag_match = re.search(r"Global Diagonal Fraction:\s*([\d.]+)", c_content)
            if fit_match: user_fit_time = fit_match.group(1)
            if pred_match: user_pred_time = pred_match.group(1)
            if diag_match: user_diag_fraction = diag_match.group(1)

    time_d3 = time.time() - start_d3

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- D3: Confusion Matrix & Model Comparisons ---\n")
        log_file.write(f"Item-Based Confusion Matrix Dict: {matrix_dict}\n")
        log_file.write(f"Item-Based Diagonal Fraction: {item_diagonal_fraction:.4f}\n\n")
        log_file.write("--- Side-by-Side Comparison Data ---\n")
        log_file.write(f"Model Fit Time (s)      | User-CF: {user_fit_time} | Item-CF: {time_d2:.2f}\n")
        log_file.write(f"Prediction Time (s)     | User-CF: {user_pred_time} | Item-CF: {time_d3:.2f}\n")
        log_file.write(f"Diagonal Fraction       | User-CF: {user_diag_fraction} | Item-CF: {item_diagonal_fraction:.4f}\n")
        if user_diag_fraction != "N/A":
            better_model = "User-Based CF" if float(user_diag_fraction) > item_diagonal_fraction else "Item-Based CF"
            log_file.write(f"Better performing model: {better_model}\n")
        log_file.write(f"Wall-clock time for D3 evaluation: {time_d3:.2f} seconds\n\n")


    # ==========================================
    # MISSION D4: Top-10 Similar Books
    # ==========================================
    print("Running D4: Extracting Top-10 Similar Books for sample items...")
    
    # Run the identical empirical percentile logic from Part C to isolate K_book boundary
    book_counts = spark_df.groupBy("book_int_id").count()
    K_book = 2

    # Sample 5 random books with at least K_book reviews
    valid_books = book_counts.filter(f"count >= {K_book}").select("book_int_id").distinct()
    sample_books = valid_books.sample(withReplacement=False, fraction=0.1, seed=42).limit(5)

    # Restrict to the top 10 rows for each sampled item from the similarity matrix
    window_spec_10 = Window.partitionBy("b1").orderBy(F.col("cosine_sim").desc())
    top10_recs = top20_item_similarity.join(sample_books, F.col("b1") == F.col("book_int_id")) \
        .withColumn("rank_10", F.row_number().over(window_spec_10)) \
        .filter("rank_10 <= 10")

    # Map matrix IDs back to structural text ISBN values
    book_lookup = spark_df.select("book_int_id", F.col("ISBN").alias("Target_ISBN")).distinct()
    sim_book_lookup = spark_df.select("book_int_id", F.col("ISBN").alias("Similar_ISBN")).distinct()

    final_csv_df = top10_recs.join(book_lookup, top10_recs["b1"] == book_lookup["book_int_id"]) \
        .join(sim_book_lookup, top10_recs["b2"] == sim_book_lookup["book_int_id"]) \
        .select(F.col("Target_ISBN").alias("Book-ISBN"), F.col("Similar_ISBN").alias("Similar-Book-ISBN"), "cosine_sim")

    final_csv_df.toPandas().to_csv("item_cf_recommendations.csv", index=False)
    print("Item recommendations successfully exported to item_cf_recommendations.csv")

    spark.stop()
    total_duration = time.time() - global_start
    print(f"Entire Part D workflow completed successfully in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()