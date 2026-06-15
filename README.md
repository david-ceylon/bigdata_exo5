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
1 | 3.9196
2 | 4.1104
3 | 3.9358
**Mean RMSE** : 3.9886
**Std RMSE** : 0.0864

## Confusion matrix — User-Based CF (Low / Medium / High)
             | Pred Low | Pred Med | Pred High
-------------|----------|----------|----------
**Actual Low** | 5,499    | 3,353    | 74
**Actual Med** | 1        | 68,051   | 16,243
**Actual High**| 0        | 280      | 288,399

Diagonal fraction: 94.78%

## Timing — Part C
Step | Wall-clock (s)
----------------------------------|---------------
C1 User-Item matrix build | 73.55s
C2 User similarity | 13.75s
C3 Model fit | 13.24s
C4 k-fold (total) | 91.29s
C5 Prediction vs ground-truth | 18.44s

# Part D
## Confusion matrix — Item-Based CF (Low / Medium / High)
             | Pred Low | Pred Med | Pred High
-------------|----------|----------|----------
**Actual Low** | 137      | 839      | 2,290
**Actual Med** | 189      | 12,956   | 27,867
**Actual High**| 399      | 9,775    | 164,819

Diagonal fraction: 81.14%

## User-CF vs Item-CF comparison
Metric | User-CF | Item-CF
---------------------|---------|--------
Model fit time (s) | 13.24s | 50.17s
Prediction time (s) | 18.44s | 8.73s
Diagonal fraction | 94.78% | 81.14%

## Timing — Part D
Step | Wall-clock (s)
----------------------------------|---------------
D1 Item similarity | 50.17s
D2 Model fit | 50.17s
D3 Prediction vs ground-truth | 8.73s

# Part E
Cold-start users : 39,083 (57.59%)
Warm users : 28,776 (42.41%)
K_user used : 2
FP-Growth minSupport : 0.00025
FP-Growth minConfidence : 0.1
Rules generated : 950

## Timing — Part E
Step | Wall-clock (s)
----------------------------------|---------------
E1 Cold-start partitioning | 24.36s
E3 FP-Growth fit | 39.37s
E4 Recommendations for Cold Users | 39.06s (including scaling evaluation and recommendation generation)
