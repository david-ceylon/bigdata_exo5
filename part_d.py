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
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt

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

    # Initialize Spark Session locally using all available CPU cores with 8GB RAM and 8MB Stack size
    spark = SparkSession.builder \
        .appName("BookCrossing-IBCF") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.extraJavaOptions", "-Xss8m") \
        .config("spark.executor.extraJavaOptions", "-Xss8m") \
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
    # MISSION D2: Hyperparameter Tuning via Grid Search & CrossValidation
    # ==========================================
    print("Running D2: ALS Hyperparameter Tuning and Grid Search...")
    start_d2 = time.time()
    
    # Base ALS estimator with nonnegative=True
    als_base = ALS(
        userCol="user_int_id", 
        itemCol="book_int_id", 
        ratingCol="rating", 
        coldStartStrategy="drop",
        nonnegative=True
    )

    # Param grid to search
    param_grid = ParamGridBuilder() \
        .addGrid(als_base.rank, [10, 20, 50]) \
        .addGrid(als_base.regParam, [0.05, 0.1, 0.2]) \
        .addGrid(als_base.maxIter, [10, 15]) \
        .build()

    evaluator = RegressionEvaluator(
        metricName="rmse", 
        labelCol="rating", 
        predictionCol="prediction"
    )

    # 3-Fold CrossValidator with parallelism=4 to speed up execution
    cv = CrossValidator(
        estimator=als_base, 
        estimatorParamMaps=param_grid, 
        evaluator=evaluator, 
        numFolds=3,
        parallelism=4
    )

    print("Starting CrossValidation to find optimal ALS parameters...")
    cv_model = cv.fit(spark_df)
    time_d2 = time.time() - start_d2

    # Extract the best model and parameters
    best_als_model = cv_model.bestModel
    
    best_index = np.argmin(cv_model.avgMetrics)
    best_params = cv_model.getEstimatorParamMaps()[best_index]
    best_rank = best_params[als_base.rank]
    best_regParam = best_params[als_base.regParam]
    best_maxIter = best_params[als_base.maxIter]

    print(f"Optimal Rank: {best_rank}")
    print(f"Optimal Regularization Param: {best_regParam}")
    print(f"Optimal Max Iterations: {best_maxIter}")


    # ==========================================
    # MISSION D3: K-Fold Cross Validation of Best Model
    # ==========================================
    print("Running D3: 3-Fold Cross-Validation loops for the best model configuration...")
    start_d3_total = time.time()

    k_folds = 3
    folds = spark_df.randomSplit([1.0 / k_folds] * k_folds, seed=42)
    rmse_list = []

    # Configure a model with the optimal parameters
    best_als_estimator = ALS(
        userCol="user_int_id",
        itemCol="book_int_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=best_rank,
        regParam=best_regParam,
        maxIter=best_maxIter
    )

    # Manual loop execution to log individual fold metrics separately
    for i in range(k_folds):
        test_df = folds[i]
        train_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), spark_df.schema)
        for j in range(k_folds):
            if j != i:
                train_df = train_df.union(folds[j])

        model_fold = best_als_estimator.fit(train_df)
        predictions_fold = model_fold.transform(test_df)
        rmse = evaluator.evaluate(predictions_fold)
        rmse_list.append(rmse)
        print(f"  - Fold {i+1}/{k_folds} evaluated RMSE: {rmse:.4f}")
        
    mean_rmse = sum(rmse_list) / len(rmse_list)
    variance_rmse = sum((x - mean_rmse) ** 2 for x in rmse_list) / len(rmse_list)
    std_rmse = variance_rmse ** 0.5
    time_d3 = time.time() - start_d3_total

    final_model = best_als_model  # Keep the best model trained on 100% of the data


    # ==========================================
    # MISSION D4: Global Performance Evaluation (Confusion Matrix) & K_book Derivation
    # ==========================================
    print("Running D4: Global Performance Evaluation (Confusion Matrix)...")
    start_d4 = time.time()

    all_predictions = final_model.transform(spark_df)

    def bucket_rating_expr(col_name):
        return F.when(F.col(col_name) <= 3, "Low") \
                .when((F.col(col_name) > 3) & (F.col(col_name) <= 6), "Med") \
                .otherwise("High")

    evaluated_df = all_predictions \
        .withColumn("actual_class", bucket_rating_expr("rating")) \
        .withColumn("pred_class", bucket_rating_expr("prediction"))

    conf_matrix_data = evaluated_df.groupBy("actual_class", "pred_class").count().collect()
    matrix_dict = {(a, p): 0 for a in ["Low", "Med", "High"] for p in ["Low", "Med", "High"]}
    for row in conf_matrix_data:
        matrix_dict[(row["actual_class"], row["pred_class"])] = row["count"]

    total_elements = sum(matrix_dict.values())
    diagonal_elements = matrix_dict[("Low", "Low")] + matrix_dict[("Med", "Med")] + matrix_dict[("High", "High")]
    item_diagonal_fraction = diagonal_elements / total_elements if total_elements > 0 else 0

    # Calculate class metrics (Low, Med, High)
    classes = ["Low", "Med", "High"]
    metrics_report = {}
    for c in classes:
        tp = matrix_dict[(c, c)]
        fp = sum(matrix_dict[(a, c)] for a in classes if a != c)
        fn = sum(matrix_dict[(c, p)] for p in classes if p != c)
        tn = sum(matrix_dict[(a, p)] for a in classes if a != c for p in classes if p != c)
        
        acc = (tp + tn) / total_elements if total_elements > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        metrics_report[c] = {"accuracy": acc, "recall": rec, "f1": f1}

    matrix_str = (
        "             | Pred Low   | Pred Med   | Pred High  \n"
        "-------------|------------|------------|------------\n"
        f"Actual Low   | {matrix_dict[('Low', 'Low')]:<10} | {matrix_dict[('Low', 'Med')]:<10} | {matrix_dict[('Low', 'High')]:<10} \n"
        f"Actual Med   | {matrix_dict[('Med', 'Low')]:<10} | {matrix_dict[('Med', 'Med')]:<10} | {matrix_dict[('Med', 'High')]:<10} \n"
        f"Actual High  | {matrix_dict[('High', 'Low')]:<10} | {matrix_dict[('High', 'Med')]:<10} | {matrix_dict[('High', 'High')]:<10} "
    )

    metrics_str = "Class Performance Metrics:\n"
    for c in classes:
        metrics_str += f"  - Class {c:4}: Accuracy = {metrics_report[c]['accuracy']:.4f} | Recall = {metrics_report[c]['recall']:.4f} | F1 = {metrics_report[c]['f1']:.4f}\n"

    # Book activity grouping analysis
    book_counts = spark_df.groupBy("book_int_id").count()
    df_book_eval = evaluated_df.join(book_counts, "book_int_id")
    book_group_bins = [
        ("1-4", 1, 4),
        ("5-9", 5, 9),
        ("10-19", 10, 19),
        ("20-49", 20, 49),
        ("50+", 50, 9999999)
    ]
    
    book_accuracies = []
    book_matrix_texts = []
    
    for label, min_c, max_c in book_group_bins:
        group_df = df_book_eval.filter((F.col("count") >= min_c) & (F.col("count") <= max_c))
        group_conf = group_df.groupBy("actual_class", "pred_class").count().collect()
        group_dict = {(a, p): 0 for a in ["Low", "Med", "High"] for p in ["Low", "Med", "High"]}
        for row in group_conf:
            group_dict[(row["actual_class"], row["pred_class"])] = row["count"]
            
        total = sum(group_dict.values())
        diag_adj = total - group_dict[("Low", "High")] - group_dict[("High", "Low")]
        acc = diag_adj / total if total > 0 else 0
        book_accuracies.append(acc)
        
        m_str = (
            f"Group Book {label} (Total: {total}, Diagonal/Adjacent Acc: {acc:.4f}):\n"
            "             | Pred Low   | Pred Med   | Pred High  \n"
            "-------------|------------|------------|------------\n"
            f"Actual Low   | {group_dict[('Low', 'Low')]:<10} | {group_dict[('Low', 'Med')]:<10} | {group_dict[('Low', 'High')]:<10} \n"
            f"Actual Med   | {group_dict[('Med', 'Low')]:<10} | {group_dict[('Med', 'Med')]:<10} | {group_dict[('Med', 'High')]:<10} \n"
            f"Actual High  | {group_dict[('High', 'Low')]:<10} | {group_dict[('High', 'Med')]:<10} | {group_dict[('High', 'High')]:<10} "
        )
        book_matrix_texts.append(m_str)

    # Plot single analysis graph for books
    x_values = [1, 5, 10, 20, 50]
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, book_accuracies, marker='s', linestyle='--', color='r', label='Book-Based Accuracy')
    plt.xlabel('Minimum Rating Count of Group (Threshold)', fontsize=12)
    plt.ylabel('Diagonal or Adjacent Accuracy Fraction', fontsize=12)
    plt.title('Empirical K_book Inflection Analysis', fontsize=14)
    plt.xticks(x_values)
    plt.grid(True, linestyle=':')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("inflection_analysis_kbook.png")
    plt.close()

    def find_inflection_point(x_vals, accs, threshold=0.015):
        if len(accs) < 2:
            return 2
        start_idx = 0
        if accs[1] < accs[0]:
            start_idx = 1
            
        max_acc = max(accs[start_idx:])
        for i in range(start_idx, len(accs)):
            if accs[i] >= 0.98 * max_acc:
                val = x_vals[i]
                return val if val >= 2 else 2
            if i < len(accs) - 1:
                diff = accs[i+1] - accs[i]
                if diff < threshold:
                    val = x_vals[i]
                    return val if val >= 2 else 2
        return x_vals[-1]

    K_book = find_inflection_point(x_values, book_accuracies)

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

    time_d4 = time.time() - start_d4

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- D3: Confusion Matrix & Model Comparisons ---\n")
        log_file.write(f"Derived K_book threshold: {K_book}\n")
        log_file.write(f"Confusion Matrix:\n{matrix_str}\n")
        log_file.write(metrics_str)
        log_file.write(f"Item-Based Diagonal Fraction: {item_diagonal_fraction:.4f}\n\n")
        log_file.write("--- Side-by-Side Comparison Data ---\n")
        log_file.write(f"Model Fit Time (s)      | User-CF: {user_fit_time} | Item-CF: {time_d2:.2f}\n")
        log_file.write(f"Prediction Time (s)     | User-CF: {user_pred_time} | Item-CF: {time_d4:.2f}\n")
        log_file.write(f"Diagonal Fraction       | User-CF: {user_diag_fraction} | Item-CF: {item_diagonal_fraction:.4f}\n")
        if user_diag_fraction != "N/A":
            better_model = "User-Based CF" if float(user_diag_fraction) > item_diagonal_fraction else "Item-Based CF"
            log_file.write(f"Better performing model: {better_model}\n")
        log_file.write(f"Wall-clock time for D4 evaluation: {time_d4:.2f} seconds\n\n")
        log_file.write(f"Optimal Rank: {best_rank} | Optimal RegParam: {best_regParam} | Optimal MaxIter: {best_maxIter}\n")
        log_file.write(f"Fold 1 RMSE: {rmse_list[0]:.4f} | Fold 2 RMSE: {rmse_list[1]:.4f} | Fold 3 RMSE: {rmse_list[2]:.4f}\n")
        log_file.write(f"Mean RMSE: {mean_rmse:.4f} | Std RMSE: {std_rmse:.4f}\n")
        log_file.write(f"Wall-clock time for D3 (K-Fold total loops): {time_d3:.2f} seconds\n")
        log_file.write(f"Wall-clock time for D2 (Tuning Grid Search): {time_d2:.2f} seconds\n\n")
        
        log_file.write("--- D3: Book Group Confusion Matrices ---\n")
        for text in book_matrix_texts:
            log_file.write(text + "\n")

    # ==========================================
    # MISSION D4: Top-10 Similar Books
    # ==========================================
    print("Running D4: Extracting Top-10 Similar Books for sample items...")
    
    # Use the empirically derived K_book threshold
    book_counts = spark_df.groupBy("book_int_id").count()

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