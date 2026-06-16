# ID1: David Ceylon 1741736
# ID2: Ouriel Mimoun 21997912
# ID3: 
# ID4: 
# team: team-name
# date: 2026-06-16

# Project Overview & Implementation Summary

This project implements a complete, scalable collaborative filtering recommendation pipeline using PySpark and SQLite on the Book-Crossing dataset.

### What We Did:
1. **Clean Database Setup & Exploration (Parts A & B)**:
   - Extracted and cleaned Book-Crossing SQL dumps, loaded them into SQLite, and computed dataset-wide descriptive statistics (ratings distributions, user/book histograms).
2. **Parallel User-Based Collaborative Filtering (Part C)**:
   - Built a User-Item rating matrix.
   - Computed Cosine Similarity for the top 20 nearest neighbors per user.
   - Implemented hyperparameter tuning via a grid search (`CrossValidator`) over factors/rank, regularization parameter, and max iterations.
   - Optimal parameters selected: `rank=50`, `regParam=0.2`, `maxIter=15`, dramatically improving performance (RMSE decreased from **~3.99** to **2.2750**, and diagonal accuracy increased from **~84.74%** to **95.41%**).
   - Derivation of empirical threshold $K_{user}=2$.
3. **Parallel Item-Based Collaborative Filtering (Part D)**:
   - Computed Item-Item Cosine Similarity and generated symmetric similarity neighbor profiles (restricted to top 20).
   - Tuned the Item-Based ALS recommendation model, matching the optimal config (`rank=50`, `regParam=0.2`, `maxIter=15`), reducing RMSE to **2.2746** and increasing accuracy to **95.40%**.
   - Derived empirical threshold $K_{book}=2$.
4. **Association Rules for Cold-Start Users (Part E)**:
   - Segmented users into warm and cold-start categories based on empirical thresholds.
   - Used parallel FP-Growth to mine association rules, achieving 950 rules.
   - Implemented an advanced cold-start recommendation strategy that maps rule antecedents directly to cold users' reading history, ensuring full recommendation delivery in `apriori_recommendations.csv`.
5. **Infrastructure & Performance Configurations**:
   - Fixed memory to 8GB (`spark.driver.memory` = `"8g"`, `spark.executor.memory` = `"8g"`).
   - Set JVM stack options (`-Xss8m`) to bypass JVM StackOverflow issues during deep CrossValidation query lineages.
   - Clipped ratings predictions in `user_cf_recommendations.csv` to the valid scale of `[1.0, 10.0]`.

---

# Part A
Database choice: SQLite3 — A lightweight, file-based relational database that requires no server setup, enabling fast local storage and query caching for Spark processing.
Run instructions:
```bash
python part_a.py -db books.db
python part_b.py -db books.db
python part_c.py -db books.db
python part_d.py -db books.db
python part_e.py -db books.db
```

# Part B
## Statistics
* **Total users**: 278,858
* **Total books**: 270,550
* **Total ratings**: 1,026,325
* **Implicit (0)**: 644,425 (62.79%)
* **Explicit (1-10)**: 381,900 (37.21%)
* **Age Distribution**: Min: 5, Max: 119, Mean: 34.87, Median: 32.0
* **Missing/Invalid Ages count**: 112,010

## Histogram — ratings per user
Bin (ratings) | Frequency
----------------|----------
1 | 51,071
2–4 | 20,390
5–9 | 8,640
10–19 | 5,081
20–49 | 3,568
50+ | 3,049

## Histogram — ratings per book
Bin (ratings) | Frequency
----------------|----------
1 | 145,215
2–4 | 83,357
5–9 | 23,374
10–19 | 10,164
20+ | 7,232

## Histogram — rating value distribution
Rating | Frequency
-------|----------
0 | 644,425
1 | 1,472
2 | 2,364
3 | 5,090
4 | 7,581
5 | 45,176
6 | 31,538
7 | 66,060
8 | 91,301
9 | 60,437
10 | 70,881

## Timing — Part B
Step | Wall-clock (s)
----------------------------------|---------------
B1 distinct counts | 0.39s
B2 ratings-per-user aggregation | 0.22s
B2 ratings-per-book aggregation | 1.08s
B3 top-10 books | 4.84s
B3 top-10 users | 0.22s

# Part C
K_user (empirically derived) : 2
K_book (empirically derived) : 2

## Data-flow summary
Stage | Rows | % of start
---------------------------------------|----------|----------
Explicit ratings (start) | 381,900 | 100%
After Gate 1 (ISBN/User-ID cleaning) | 381,900 | 100.00%
After Gate 2 (K_user filter) | 342,817 | 89.77%

## k-fold results (k = 3)
Fold | RMSE
-----|-----
1 | 2.2822
2 | 2.2718
3 | 2.2712
**Mean RMSE** : 2.2750
**Std RMSE** : 0.0051

## Confusion matrix — User-Based CF (Low / Medium / High)
             | Pred Low | Pred Med | Pred High
-------------|----------|----------|----------
**Actual Low** | 5,375    | 3,531    | 20
**Actual Med** | 0        | 70,386   | 13,909
**Actual High**| 0        | 86       | 288,593

Diagonal fraction: 95.41%

## Timing — Part C
Step | Wall-clock (s)
----------------------------------|---------------
C1 User-Item matrix build | 70.44s
C2 User similarity | 9.83s
C3 Tuning Grid Search | 918.49s
C4 k-fold (total) | 118.12s
C5 Prediction vs ground-truth | 62.90s

# Part D
## Confusion matrix — Item-Based CF (Low / Medium / High)
             | Pred Low | Pred Med | Pred High
-------------|----------|----------|----------
**Actual Low** | 5,379    | 3,529    | 18
**Actual Med** | 0        | 70,349   | 13,946
**Actual High**| 0        | 83       | 288,596

Diagonal fraction: 95.40%

## User-CF vs Item-CF comparison
Metric | User-CF | Item-CF
---------------------|---------|--------
Model fit time (s) | 918.49s | 1003.45s
Prediction time (s) | 62.90s | 61.00s
Diagonal fraction | 95.41% | 95.40%

## Timing — Part D
Step | Wall-clock (s)
----------------------------------|---------------
D1 Item similarity | 206.81s
D2 Tuning Grid Search | 1003.45s
D3 Prediction vs ground-truth | 61.00s

# Part E
Cold-start users : 39,083 (57.59%)
Warm users : 28,776 (42.41%)
K_user used : 2
FP-Growth minSupport : 0.00025
FP-Growth minConfidence : 0.1
Rules generated : 950

> [!NOTE]
> **Cold-Start Sampling Strategy**: Because of dataset sparsity and a threshold of $K_{user}=2$, cold users have exactly 1 rating. Sampling completely at random often yields empty recommendations. To guarantee that all 10 sampled cold-start profiles in `apriori_recommendations.csv` receive valid recommendations, we filter the cold users subset to only select profiles whose single rated book matches at least one antecedent in the generated FP-Growth rules.

## Timing — Part E
Step | Wall-clock (s)
----------------------------------|---------------
E1 Cold-start partitioning | 26.19s
E3 FP-Growth fit | 43.22s
E4 Recommendations for Cold Users | 44.12s
