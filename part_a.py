import argparse
import os
import re
import sqlite3
import time
from datetime import datetime


def get_table_dimensions(cursor, table_name):
    """Helper function to get (rows, columns) of a table."""
    cursor.execute(f"PRAGMA table_info(`{table_name}`);")
    columns_count = len(cursor.fetchall())

    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
    rows_count = cursor.fetchone()[0]

    return rows_count, columns_count


def main():
    # 1. Start overall wall-clock timer
    script_start_time = time.time()

    # 2. Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Part A: Pipeline Ingestion and Data Cleaning"
    )
    parser.add_argument(
        "-input",
        required=True,
        help="Path to the input SQL dump file (e.g., BX-dump.sql)",
    )
    parser.add_argument(
        "-db",
        required=True,
        help="Path to the output database file (e.g., books.db)",
    )
    args = parser.parse_args()

    log_filename = "part_a.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize log file
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Script: {os.path.basename(__file__)} ===\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Input file: {args.input}\n")
        log_file.write(f"Output DB: {args.db}\n\n")

    print(f"[{timestamp}] Starting Step A1: Database Population...")
    a1_start_time = time.time()

    # Connect to SQLite database
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    # Read and map MySQL syntax to SQLite
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        sql_content = f.read()

    # Syntax conversions
    sql_content = sql_content.replace("\\'", "''")
    sql_content = re.sub(
        r"(?i)INSERT\s+IGNORE\s+INTO", "INSERT OR IGNORE INTO", sql_content
    )
    sql_content = re.sub(r"(?i)LOCK\s+TABLES\s+[^;]+;", "", sql_content)
    sql_content = re.sub(r"(?i)UNLOCK\s+TABLES\s*;", "", sql_content)
    sql_content = re.sub(
        r"ENGINE=\w+\s*(DEFAULT\s+CHARSET=\w+)?", "", sql_content
    )
    sql_content = re.sub(r"int\(\d+\)", "INTEGER", sql_content)
    sql_content = re.sub(r"varchar\(\d+\)", "TEXT", sql_content)
    sql_content = re.sub(r"char\(\d+\)", "TEXT", sql_content)

    # Execute A1 Database Population
    try:
        cursor.execute("PRAGMA foreign_keys = OFF;")
        cursor.execute("PRAGMA journal_mode = OFF;")
        cursor.execute("PRAGMA synchronous = OFF;")
        cursor.execute("PRAGMA cache_size = 1000000;")
        cursor.executescript(sql_content)
        conn.commit()
        print("Database successfully populated.")
    except sqlite3.Error as e:
        print(f"An error occurred during SQL execution: {e}")
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write(f"SQL Loading Error: {e}\n")
        return

    a1_end_time = time.time()
    a1_duration = a1_end_time - a1_start_time

    # Get initial table dimensions right after population
    tables = ["BX-Books", "BX-Users", "BX-Book-Ratings"]
    initial_dims = {}
    for table in tables:
        initial_dims[table] = get_table_dimensions(cursor, table)

    # Write A1 stats to log
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- Step A1: Initial Data Dimensions ---\n")
        for table, dims in initial_dims.items():
            log_file.write(
                f"Table `{table}`: {dims[0]} rows, {dims[1]} columns\n"
            )
        log_file.write(f"Wall-clock time for A1: {a1_duration:.2f} seconds\n\n")

    print(f"Starting Step A2: Data Cleaning...")
    a2_start_time = time.time()

    # Track row reductions
    removed_summary = {}

    # --- 1. Clean BX-Books ---
    # Keep only ISBN of length 13 or standard 10, and non-empty title/author
    cursor.execute(
        """
        DELETE FROM `BX-Books`
        WHERE (LENGTH(TRIM(`ISBN`)) != 13 AND LENGTH(TRIM(`ISBN`)) != 10)
           OR `Book-Title` IS NULL OR TRIM(`Book-Title`) = ''
           OR `Book-Author` IS NULL OR TRIM(`Book-Author`) = '';
    """
    )
    # Remove duplicate primary keys (keeping the first occurrence)
    cursor.execute(
        """
        DELETE FROM `BX-Books`
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM `BX-Books` GROUP BY `ISBN`
        );
    """
    )
    conn.commit()

    # --- 2. Clean BX-Users ---
    # Remove duplicate primary keys
    cursor.execute(
        """
        DELETE FROM `BX-Users`
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM `BX-Users` GROUP BY `User-ID`
        );
    """
    )
    conn.commit()

    # --- 3. Clean BX-Book-Ratings ---
    # Remove duplicate (User-ID, ISBN) pairs keeping the first occurrence
    cursor.execute(
        """
        DELETE FROM `BX-Book-Ratings`
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM `BX-Book-Ratings` GROUP BY `User-ID`, `ISBN`
        );
    """
    )
    # Integrity constraints (ISBN must exist in Books, User-ID must exist in Users)
    cursor.execute(
        """
        DELETE FROM `BX-Book-Ratings`
        WHERE `ISBN` NOT IN (SELECT `ISBN` FROM `BX-Books`)
           OR `User-ID` NOT IN (SELECT `User-ID` FROM `BX-Users`);
    """
    )
    conn.commit()

    a2_end_time = time.time()
    a2_duration = a2_end_time - a2_start_time

    # Get final dimensions after cleaning
    final_dims = {}
    for table in tables:
        final_dims[table] = get_table_dimensions(cursor, table)

    # --- 4. Perform Final Validation JOIN ---
    print("Performing post-cleaning validation JOIN...")
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM `BX-Book-Ratings` r
        JOIN `BX-Books` b ON r.`ISBN` = b.`ISBN`
        JOIN `BX-Users` u ON r.`User-ID` = u.`User-ID`;
    """
    )
    join_row_count = cursor.fetchone()[0]

    # Write A2 stats to log
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("--- Step A2: Cleaned Data Dimensions & Summary ---\n")
        for table in tables:
            removed_rows = initial_dims[table][0] - final_dims[table][0]
            log_file.write(f"Table `{table}`:\n")
            log_file.write(f"  - Rows Removed: {removed_rows}\n")
            log_file.write(
                f"  - Rows Remaining: {final_dims[table][0]} (Columns: {final_dims[table][1]})\n"
            )

        log_file.write(f"\nRow count after complete JOIN: {join_row_count}\n")
        log_file.write(f"Wall-clock time for A2: {a2_duration:.2f} seconds\n\n")

    # Close DB Connection
    conn.close()

    # Calculate overall script execution time
    script_end_time = time.time()
    total_duration = script_end_time - script_start_time

    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"Total Wall-clock time for part_a.py: {total_duration:.2f} seconds\n"
        )

    print(f"Step A2 completed in {a2_duration:.2f} seconds.")
    print(f"Entire script executed successfully in {total_duration:.2f} seconds.")


if __name__ == "__main__":
    main()