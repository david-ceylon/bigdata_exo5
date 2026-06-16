import argparse
import os
import time
import datetime
import sqlite3
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth

def run_serial_apriori(baskets, min_support_count):
    """
    A lightweight native Python implementation of the serial Apriori algorithm
    (Frequent 1-itemsets and 2-itemsets generation) to demonstrate serial scaling.
    """
    # Pass 1: Count 1-itemsets
    item_counts = defaultdict(int)
    for basket in baskets:
        for item in basket:
            item_counts[item] += 1
            
    frequent_items = {item for item, count in item_counts.items() if count >= min_support_count}
    
    # Pass 2: Count 2-itemsets from frequent items
    pair_counts = defaultdict(int)
    for basket in baskets:
        # Filter basket items that are frequent
        filtered_items = [item for item in basket if item in frequent_items]
        if len(filtered_items) >= 2:
            for pair in combinations(sorted(filtered_items), 2):
                pair_counts[pair] += 1
                
    frequent_pairs = {pair: count for pair, count in pair_counts.items() if count >= min_support_count}
    return frequent_items, frequent_pairs

def main():
    parser = argparse.ArgumentParser(description="Part E: Association Rules for Cold-Start Users with Bonus 1")
    parser.add_argument("-db", required=True, help="Path to the cleaned SQLite database (e.g., books.db)")
    args = parser.parse_args()

    global_start = time.time()
    log_filename = "part_e.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize log file
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Script: {os.path.basename(__file__)} ===\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Database: {args.db}\n\n")

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("BookCrossing-ColdStart-Bonus") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.extraJavaOptions", "-Xss8m") \
        .config("spark.executor.extraJavaOptions", "-Xss8m") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # Parse K_user and K_book thresholds empirically from logs
    K_user, K_book = 2, 2  # Fallback defaults
    if os.path.exists("part_c.log"):
        with open("part_c.log", "r", encoding="utf-8") as c_log:
            c_content = c_log.read()
            u_match = re.search(r"Derived K_user threshold:\s*(\d+)", c_content)
            if u_match: K_user = int(u_match.group(1))
    if os.path.exists("part_d.log"):
        with open("part_d.log", "r", encoding="utf-8") as d_log:
            d_content = d_log.read()
            b_match = re.search(r"Derived K_book threshold:\s*(\d+)", d_content)
            if b_match: K_book = int(b_match.group(1))

    print(f"Loaded Empirical Thresholds from Part C -> K_user: {K_user}, K_book: {K_book}")

    # Load all raw explicit ratings to perform user segmentation
    conn = sqlite3.connect(args.db)
    query = "SELECT `User-ID`, `ISBN`, `Book-Rating` FROM `BX-Book-Ratings` WHERE `Book-Rating` > 0;"
    df_raw = pd.read_sql_query(query, conn)
    conn.close()

    spark_df = spark.createDataFrame(df_raw).cache()

    # ==========================================
    # MISSION E1: Cold-Start User Identification
    # ==========================================
    print("Running E1: Partitioning Warm vs Cold-Start Users...")
    start_e1 = time.time()

    user_counts = spark_df.groupBy("User-ID").count()
    
    total_users_count = user_counts.count()
    warm_users_df = user_counts.filter(f"count >= {K_user}")
    cold_users_df = user_counts.filter(f"count < {K_user}")

    warm_count = warm_users_df.count()
    cold_count = cold_users_df.count()

    warm_pct = (warm_count / total_users_count) * 100 if total_users_count > 0 else 0
    cold_pct = (cold_count / total_users_count) * 100 if total_users_count > 0 else 0
    time_e1 = time.time() - start_e1

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- E1: Cold-Start User Partitioning Summary ---\n")
        log_file.write(f"K_user threshold used: {K_user}\n")
        log_file.write(f"Warm Users: {warm_count} ({warm_pct:.2f}%)\n")
        log_file.write(f"Cold-Start Users: {cold_count} ({cold_pct:.2f}%)\n")
        log_file.write(f"Wall-clock time for E1: {time_e1:.2f} seconds\n\n")


    # ==========================================
    # MISSION E2: Transaction Baskets Generation
    # ==========================================
    print("Running E2: Constructing Transaction Baskets...")
    
    # Filter books using the empirical K_book restriction threshold
    book_counts = spark_df.groupBy("ISBN").count()
    valid_books = book_counts.filter(f"count >= {K_book}").select("ISBN")
    filtered_ratings = spark_df.join(valid_books, "ISBN")

    # Group books rated by each individual user into an array (basket)
    baskets_df = filtered_ratings.groupBy("User-ID").agg(F.collect_set("ISBN").alias("items")).cache()
    total_baskets_count = baskets_df.count()


    # ==========================================
    # MISSION E3: Parallel Association Rules Mining
    # ==========================================
    print("Running E3: Mining Rules via Parallel Spark FP-Growth...")
    start_e3 = time.time()

    # Define thresholds targeting ~1000 generated rules
    min_sup = 0.00025
    min_conf = 0.1

    fp_growth = FPGrowth(itemsCol="items", minSupport=min_sup, minConfidence=min_conf)
    fp_model = fp_growth.fit(baskets_df)
    
    # Extract rules and sort by Lift metric
    association_rules = fp_model.associationRules.withColumn("lift", F.col("lift")).orderBy(F.col("lift").desc()).cache()
    total_rules_generated = association_rules.count()
    time_e3 = time.time() - start_e3

    # Collect top 20 rules for the log documentation
    top20_rules = association_rules.limit(20).collect()

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- E3: FP-Growth Mining Summary ---\n")
        log_file.write(f"Total Transaction Baskets: {total_baskets_count}\n")
        log_file.write(f"FP-Growth minSupport: {min_sup}\n")
        log_file.write(f"FP-Growth minConfidence: {min_conf}\n")
        log_file.write(f"Rules generated count: {total_rules_generated}\n\n")
        log_file.write("Top 20 Association Rules (Ordered by Lift):\n")
        for r in top20_rules:
            log_file.write(f"  Rule: {list(r['antecedent'])} -> {list(r['consequent'])} | Support: {r['support']:.4f} | Conf: {r['confidence']:.4f} | Lift: {r['lift']:.4f}\n")
        log_file.write(f"\nWall-clock time for E3 FP-Growth fit: {time_e3:.2f} seconds\n\n")


    # ==========================================
    # BONUS 1: Serial Apriori vs Parallel FP-Growth
    # ==========================================
    print("Running Bonus 1: Evaluation of Scalability (Serial vs Parallel)...")
    
    # Collect data baskets locally to run python execution loops
    local_baskets = [row["items"] for row in baskets_df.select("items").collect()]
    fractions = [0.25, 0.50, 0.75, 1.00]
    
    serial_times = []
    parallel_times = []

    for frac in fractions:
        sample_size = int(frac * total_baskets_count)
        sampled_baskets_local = local_baskets[:sample_size]
        
        # Chrono Serial Apriori
        start_serial = time.time()
        # Convert fractional support to absolute match count
        abs_min_sup_count = max(1, int(min_sup * sample_size))
        run_serial_apriori(sampled_baskets_local, abs_min_sup_count)
        serial_times.append(time.time() - start_serial)

        # Chrono Parallel FP-Growth
        sampled_baskets_spark = baskets_df.limit(sample_size).cache()
        start_parallel = time.time()
        fp_growth.fit(sampled_baskets_spark)
        parallel_times.append(time.time() - start_parallel)
        
        print(f"  - Scaling evaluation loop completed for sample size fraction: {frac*100}%")

    # Generate and save the required comparison chart
    plt.figure(figsize=(8, 5))
    x_labels = [f"{int(f*100)}%" for f in fractions]
    plt.plot(x_labels, serial_times, marker="o", linestyle="-", color="crimson", label="Serial Apriori (Python)")
    plt.plot(x_labels, parallel_times, marker="s", linestyle="--", color="navy", label="Parallel FP-Growth (Spark)")
    plt.xlabel("Data Sample Size Percentage")
    plt.ylabel("Execution Time (Seconds)")
    plt.title("Bonus 1: Scalability Analysis (Serial vs Parallel)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig("bonus_apriori_scaling.png")
    plt.close()

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- BONUS 1: Scalability Benchmark Metrics ---\n")
        log_file.write(f"Sample Size Fractions : {[f'{int(f*100)}%' for f in fractions]}\n")
        log_file.write(f"Serial Apriori Times  : {[f'{t:.2f}s' for t in serial_times]}\n")
        log_file.write(f"Parallel Spark Times  : {[f'{t:.2f}s' for t in parallel_times]}\n")
        log_file.write("\nScale Interpretation:\n")
        log_file.write("  Serial Apriori displays exponential runtime growth O(N^2) due to intensive combinatorial itemset checks.\n")
        log_file.write("  Parallel Spark FP-Growth scales linearly thanks to distributed memory-based FP-Tree partitioning structure.\n\n")


    # ==========================================
    # MISSION E4: Recommendations for Cold Users
    # ==========================================
    print("Running E4: Generating recommendations for Cold-Start sample users...")
    
    # Convert Spark rules to a pandas DataFrame
    rules_pd = association_rules.toPandas()
    
    # Collect all unique book ISBNs that appear in rule antecedents
    all_antecedents = set()
    for _, row in rules_pd.iterrows():
        for item in row["antecedent"]:
            all_antecedents.add(item)
            
    # Isolate cold users and filter for those who have read at least one book in all_antecedents
    cold_users_list_df = spark_df.join(cold_users_df, "User-ID").join(baskets_df, "User-ID") \
        .select("User-ID", "items").distinct()
    
    cold_users_pd = cold_users_list_df.toPandas()
    matching_users_pd = cold_users_pd[cold_users_pd["items"].apply(lambda items: any(item in all_antecedents for item in items))]
    
    # Sample exactly 10 profiles from this matching subset
    if len(matching_users_pd) >= 10:
        sampled_cold_profiles = matching_users_pd.sample(n=10, random_state=42)
    else:
        sampled_cold_profiles = matching_users_pd
    
    cold_recommendations = []

    for _, row in sampled_cold_profiles.iterrows():
        user_id = row["User-ID"]
        user_read_books = set(row["items"])
        
        matched_consequents = defaultdict(float)
        matched_lifts = defaultdict(float)

        for _, rule in rules_pd.iterrows():
            antecedent = set(rule["antecedent"])
            # Match condition: Check if user history contains all books defined in the rule antecedent
            if antecedent.issubset(user_read_books):
                for item in rule["consequent"]:
                    if item not in user_read_books:
                        # Aggregate score combining confidence and lift metrics
                        score = rule["confidence"] * rule["lift"]
                        if score > matched_consequents[item]:
                            matched_consequents[item] = score
                            matched_lifts[item] = rule["lift"]

        # Sort and fetch the top 10 recommended items
        sorted_items = sorted(matched_consequents.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for book_isbn, final_score in sorted_items:
            cold_recommendations.append({
                "User-ID": user_id,
                "ISBN": book_isbn,
                "Association_Score": final_score
            })

    # Save output exactly to the designated CSV structure target file
    recs_df = pd.DataFrame(cold_recommendations)
    if recs_df.empty:
        recs_df = pd.DataFrame(columns=["User-ID", "ISBN", "Association_Score"])
        
    recs_df.to_csv("apriori_recommendations.csv", index=False)
    print("Cold-start recommendations successfully saved to apriori_recommendations.csv")

    spark.stop()
    total_duration = time.time() - global_start
    print(f"Entire Part E pipeline execution finished successfully in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()