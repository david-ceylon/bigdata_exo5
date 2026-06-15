import argparse
import datetime
import os
import time
import sqlite3
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Part B: Exploratory Data Analysis")
    parser.add_argument("-db", required=True, help="Path to the cleaned SQLite database (e.g., books.db)")
    args = parser.parse_args()

    log_filename = "part_b.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize log file
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Script: {os.path.basename(__file__)} ===\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Database: {args.db}\n\n")

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    # ==========================================
    # MISSION B1: Basic Statistics
    # ==========================================
    print("Running B1: Basic Statistics...")
    start_b1 = time.time()

    # Unique counts
    cursor.execute("SELECT COUNT(DISTINCT `User-ID`) FROM `BX-Users`;")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT `ISBN`) FROM `BX-Books`;")
    total_books = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM `BX-Book-Ratings`;")
    total_ratings = cursor.fetchone()[0]

    # Implicit (0) vs Explicit (1-10) ratings
    cursor.execute("SELECT COUNT(*) FROM `BX-Book-Ratings` WHERE `Book-Rating` = 0;")
    implicit_count = cursor.fetchone()[0]
    explicit_count = total_ratings - implicit_count

    implicit_pct = (implicit_count / total_ratings) * 100 if total_ratings > 0 else 0
    explicit_pct = (explicit_count / total_ratings) * 100 if total_ratings > 0 else 0

    # User Age stats (Treating NULL, < 5, or > 100 as missing/invalid)
    cursor.execute("""
        SELECT MIN(`Age`), MAX(`Age`), AVG(`Age`) 
        FROM `BX-Users` 
        WHERE `Age` IS NOT NULL AND `Age` >= 5 AND `Age` <= 120;
    """)
    age_min, age_max, age_avg = cursor.fetchone()

    # Calculate Median Age
    cursor.execute("""
        SELECT `Age` FROM `BX-Users` 
        WHERE `Age` IS NOT NULL AND `Age` >= 5 AND `Age` <= 100 
        ORDER BY `Age`;
    """)
    ages = [row[0] for row in cursor.fetchall()]
    if ages:
        n = len(ages)
        age_median = ages[n // 2] if n % 2 != 0 else (ages[(n // 2) - 1] + ages[n // 2]) / 2
    else:
        age_median = None

    cursor.execute("""
        SELECT COUNT(*) FROM `BX-Users` 
        WHERE `Age` IS NULL OR `Age` < 5 OR `Age` > 100;
    """)
    invalid_ages = cursor.fetchone()[0]

    time_b1 = time.time() - start_b1

    # Log B1 results
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- B1: Basic Statistics ---\n")
        log_file.write(f"Total Users: {total_users}\n")
        log_file.write(f"Total Books: {total_books}\n")
        log_file.write(f"Total Ratings: {total_ratings}\n")
        log_file.write(f"Implicit Ratings (0): {implicit_count} ({implicit_pct:.2f}%)\n")
        log_file.write(f"Explicit Ratings (1-10): {explicit_count} ({explicit_pct:.2f}%)\n")
        log_file.write(f"Age Distribution -> Min: {age_min}, Max: {age_max}, Mean: {age_avg:.2f}, Median: {age_median}\n")
        log_file.write(f"Missing/Invalid Ages count: {invalid_ages}\n")
        log_file.write(f"Execution time for B1: {time_b1:.2f} seconds\n\n")


    # ==========================================
    # MISSION B2: Rating Histograms & Rank Plots
    # ==========================================
    print("Running B2: User Aggregations & Rank Plot...")
    start_b2_user = time.time()
    
    # Ratings per user distribution
    cursor.execute("SELECT `User-ID`, COUNT(*) as cnt FROM `BX-Book-Ratings` GROUP BY `User-ID` ORDER BY cnt DESC;")
    user_ratings = cursor.fetchall()
    time_b2_user = time.time() - start_b2_user

    print("Running B2: Book Aggregations & Rank Plot...")
    start_b2_book = time.time()
    
    # Ratings per book distribution
    cursor.execute("SELECT `ISBN`, COUNT(*) as cnt FROM `BX-Book-Ratings` GROUP BY `ISBN` ORDER BY cnt DESC;")
    book_ratings = cursor.fetchall()
    time_b2_book = time.time() - start_b2_book

    # Frequency binning logic for README markdown template tables
    user_bins = {"1": 0, "2-4": 0, "5-9": 0, "10-19": 0, "20-49": 0, "50+": 0}
    for _, count in user_ratings:
        if count == 1: user_bins["1"] += 1
        elif 2 <= count <= 4: user_bins["2-4"] += 1
        elif 5 <= count <= 9: user_bins["5-9"] += 1
        elif 10 <= count <= 19: user_bins["10-19"] += 1
        elif 20 <= count <= 49: user_bins["20-49"] += 1
        else: user_bins["50+"] += 1

    book_bins = {"1": 0, "2-4": 0, "5-9": 0, "10-19": 0, "20+": 0}
    for _, count in book_ratings:
        if count == 1: book_bins["1"] += 1
        elif 2 <= count <= 4: book_bins["2-4"] += 1
        elif 5 <= count <= 9: book_bins["5-9"] += 1
        elif 10 <= count <= 19: book_bins["10-19"] += 1
        else: book_bins["20+"] += 1

    # Plot 1: Ratings per User (Log-Log Scale Rank Plot)
    plt.figure()
    user_counts = [row[1] for row in user_ratings]
    ranks_user = list(range(1, len(user_counts) + 1))
    plt.loglog(ranks_user, user_counts, marker="o", linestyle="none", color="blue", alpha=0.6)
    plt.xlabel("User Rank (Log Scale)")
    plt.ylabel("Number of Ratings (Log Scale)")
    plt.title("Ratings per User Distribution")
    plt.grid(True, which="both", ls="--")
    plt.savefig("hist_ratings_per_user.png")
    plt.close()

    # Plot 2: Ratings per Book (Log-Log Scale Rank Plot)
    plt.figure()
    book_counts = [row[1] for row in book_ratings]
    ranks_book = list(range(1, len(book_counts) + 1))
    plt.loglog(ranks_book, book_counts, marker="s", linestyle="none", color="green", alpha=0.6)
    plt.xlabel("Book Rank (Log Scale)")
    plt.ylabel("Number of Ratings (Log Scale)")
    plt.title("Ratings per Book Distribution")
    plt.grid(True, which="both", ls="--")
    plt.savefig("hist_ratings_per_book.png")
    plt.close()

    # Plot 3: Rating Values Distribution (0 to 10)
    cursor.execute("SELECT `Book-Rating`, COUNT(*) FROM `BX-Book-Ratings` GROUP BY `Book-Rating`;")
    rating_vals = dict(cursor.fetchall())
    all_rating_values = [rating_vals.get(i, 0) for i in range(11)]

    plt.figure()
    plt.bar(range(11), all_rating_values, color="purple", edgecolor="black", alpha=0.7)
    plt.xlabel("Rating Value")
    plt.ylabel("Frequency")
    plt.xticks(range(11))
    plt.title("Distribution of Rating Values")
    plt.grid(axis="y", linestyle="--")
    plt.savefig("hist_rating_values.png")
    plt.close()

    # Log B2 statistics for README tables
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- B2: Frequency Distribution Bins for README ---\n")
        log_file.write(f"User Bins table data: {user_bins}\n")
        log_file.write(f"Book Bins table data: {book_bins}\n")
        log_file.write(f"Rating Value mapping (0-10): {rating_vals}\n")
        log_file.write(f"Time - User aggregation: {time_b2_user:.2f} s\n")
        log_file.write(f"Time - Book aggregation: {time_b2_book:.2f} s\n\n")


    # ==========================================
    # MISSION B3: Top-10 Lists (Explicit Only)
    # ==========================================
    print("Running B3: Top-10 Lists...")
    
    # Top-10 Books
    start_b3_books = time.time()
    cursor.execute("""
        SELECT b.`Book-Title`, b.`Book-Author`, COUNT(*) as explicit_cnt
        FROM `BX-Book-Ratings` r
        JOIN `BX-Books` b ON r.`ISBN` = b.`ISBN`
        WHERE r.`Book-Rating` > 0
        GROUP BY r.`ISBN`
        ORDER BY explicit_cnt DESC
        LIMIT 10;
    """)
    top10_books_data = cursor.fetchall()
    time_b3_books = time.time() - start_b3_books

    # Top-10 Users
    start_b3_users = time.time()
    cursor.execute("""
        SELECT `User-ID`, COUNT(*) as explicit_cnt
        FROM `BX-Book-Ratings`
        WHERE `Book-Rating` > 0
        GROUP BY `User-ID`
        ORDER BY explicit_cnt DESC
        LIMIT 10;
    """)
    top10_users_data = cursor.fetchall()
    time_b3_users = time.time() - start_b3_users

    # Plot B3: Save Top 10 Books Chart
    plt.figure(figsize=(10, 5))
    book_titles = [row[0][:25] + "..." if len(row[0]) > 25 else row[0] for row in top10_books_data]
    book_counts_b3 = [row[2] for row in top10_books_data]
    plt.barh(book_titles[::-1], book_counts_b3[::-1], color="orange", edgecolor="black")
    plt.xlabel("Explicit Ratings Count")
    plt.title("Top 10 Most Rated Books (Explicit Only)")
    plt.tight_layout()
    plt.savefig("top10_books.png")
    plt.close()

    # Plot B3: Save Top 10 Users Chart
    plt.figure(figsize=(10, 5))
    user_ids = [str(row[0]) for row in top10_users_data]
    user_counts_b3 = [row[1] for row in top10_users_data]
    plt.bar(user_ids, user_counts_b3, color="teal", edgecolor="black")
    plt.xlabel("User ID")
    plt.ylabel("Explicit Ratings Count")
    plt.title("Top 10 Most Active Users (Explicit Only)")
    plt.tight_layout()
    plt.savefig("top10_users.png")
    plt.close()

    # Log B3 lists
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- B3: Top-10 Explicit Lists Data ---\n")
        log_file.write("Top 10 Books:\n")
        for idx, row in enumerate(top10_books_data, 1):
            log_file.write(f"  {idx}. {row[0]} by {row[1]} ({row[2]} ratings)\n")
        log_file.write("\nTop 10 Users:\n")
        for idx, row in enumerate(top10_users_data, 1):
            log_file.write(f"  {idx}. User-ID: {row[0]} ({row[1]} ratings)\n")
        log_file.write(f"\nTime - Top-10 Books: {time_b3_books:.2f} s\n")
        log_file.write(f"Time - Top-10 Users: {time_b3_users:.2f} s\n")

    conn.close()
    print("Part B executed and completed successfully. Check generated plots and part_b.log.")

if __name__ == "__main__":
    main()