# ID1: Student_Name_1
# ID2: Student_Name_2
# team: team-name
# date: 2026-06-14

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
K_user (empirically derived) : 5
K_book (empirically derived) : 10

## Data-flow summary
Stage | Rows | % of start
---------------------------------------|----------|----------
Explicit ratings (start) | 381,900 | 100%
After Gate 1 (ISBN/User-ID cleaning) | 381,900 | 100.00%
After Gate 2 (K_user filter) | 300,501 | 78.69%

## k-fold results (k = 3)
Fold | RMSE
-----|-----
1 | 4.0451
2 | 4.0758
3 | 3.8627
**Mean RMSE** : 3.9945
**Std RMSE** : 0.0941

## Confusion matrix — User-Based CF (Low / Medium / High)
             | Pred Low | Pred Med | Pred High
-------------|----------|----------|----------
**Actual Low** | 4,516    | 2,604    | 482
**Actual Med** | 3,863    | 54,082   | 13,956
**Actual High**| 10,107   | 18,832   | 222,670

Diagonal fraction: 84.95%

## Timing — Part C
Step | Wall-clock (s)
----------------------------------|---------------
C1 User-Item matrix build | 69.26s
C2 User similarity | 11.76s
C3 Model fit | 15.74s
C4 k-fold (total) | 77.59s
C5 Prediction vs ground-truth | 83.93s

# Part D
## Confusion matrix — Item-Based CF (Low / Medium / High)
             | Pred Low | Pred Med | Pred High
-------------|----------|----------|----------
**Actual Low** | 4,616    | 2,503    | 483
**Actual Med** | 4,065    | 54,328   | 13,508
**Actual High**| 10,575   | 19,045   | 221,989

Diagonal fraction: 84.85%

## User-CF vs Item-CF comparison
Metric | User-CF | Item-CF
---------------------|---------|--------
Model fit time (s) | 33.04s | 27.05s
Prediction time (s) | 77.47s | 46.09s
Diagonal fraction | 84.74% | 84.85%

## Timing — Part D
Step | Wall-clock (s)
----------------------------------|---------------
D1 Item similarity | 27.05s
D2 Model fit | 27.05s
D3 Prediction vs ground-truth | 46.09s

# Part E
Cold-start users : 55,139 (81.26%)
Warm users : 12,720 (18.74%)
K_user used : 5
FP-Growth minSupport : 0.00025
FP-Growth minConfidence : 0.1
Rules generated : 3,527

## Timing — Part E
Step | Wall-clock (s)
----------------------------------|---------------
E1 Cold-start partitioning | 24.81s
E3 FP-Growth fit | 51.35s
E4 Recommendations for Cold Users | 45.34s (including scaling evaluation and recommendation generation)
