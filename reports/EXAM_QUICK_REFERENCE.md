# ASBD LAB EXAM QUICK REFERENCE CARD
**Course:** Analytics and Systems of Big Data (ASBD)  
**Dataset:** Recipe Reviews and User Feedback Dataset (UCI)  
**Quick Summary:** 18,182 records | 100 recipes | 13,812 users | Feb 2021 – Oct 2022  

---

## 1. Data Preprocessing Quick Reference
- **Dropped Column:** `Unnamed: 0` (redundant CSV export index counter).
- **Missing Value Handling:** `text` column had 2 nulls $\to$ imputed with empty string `""` / normalized text.
- **HTML Cleanup:** Decoded HTML entities (`&amp;`, `&#39;`, `&quot;`) via `html.unescape()` and regex whitespace stripping.
- **Duplicate Records:** 0 duplicate rows, 0 duplicate `comment_id`s (all 18,182 reviews are unique).
- **Invalid Value Correction:** Clipped negative values in engagement metrics to 0; validated `stars` in $[0, 5]$.
- **Temporal Transformations:** Extracted from unix epoch `created_at` $\to$ `datetime`, `year`, `month`, `year_month`, `day_of_week`, `hour`, `is_weekend`.
- **Feature Engineering:** `text_length`, `word_count`, `engagement_total = thumbs_up + thumbs_down + reply_count`, `net_thumbs`, `approval_ratio`.
- **Discretization / Binning:**
  - `rating_cat`: `Rating_0_Unrated` (0), `Rating_Low_1_2` (1-2), `Rating_Medium_3_4` (3-4), `Rating_High_5` (5).
  - `reputation_cat`: `Rep_Novice` (1), `Rep_Intermediate` (2-5), `Rep_Veteran` (>5).
  - `engagement_cat`: `Engage_Zero` (0), `Engage_Low` (1-2), `Engage_Medium` (3-6), `Engage_High` (>6).
  - `length_cat`: `Len_Short` (<80 chars), `Len_Medium` (80-220 chars), `Len_Long` (>220 chars).
  - `feedback_cat`: `Feedback_Neutral` (0 thumbs), `Feedback_Positive` (up > down), `Feedback_Critical` (down > up).
  - `reply_cat`: `No_Reply`, `Has_Reply`.
- **Task Datasets:**
  - `arm_transactions.csv`: 6 categorical columns for itemset mining.
  - `knn_features.csv`: 9 continuous/calendar features + `rating_3class` (Leakage-free: excludes `best_score`).
  - `agnes_features.csv`: 6 continuous interaction features for hierarchical distance computation.

---

## 2. Visualization Cheat Sheet (14 Unique Plot Types)

| Plot # | Plot Name | Attributes Used | When to Use | Key Analytical Inference |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Histogram + KDE** | `stars` (0-5) | Univariate distribution of discrete ordinal ratings. | Extreme positive rating bias: 76.06% are 5-star reviews; 9.33% are 0-star (unrated). |
| **02** | **Box Plot** | `stars` vs `text_length` | Five-number summary + outlier spread across categories. | Low-star/critical reviews have significantly longer median lengths (~220 chars) than 5-star (~150 chars). |
| **03** | **Violin Plot** | `rating_cat` vs `user_reputation` | Multi-modal density distribution across groups. | Bulge at rep=1 across all ratings; 5-star reviews have a thicker tail of high-reputation users. |
| **04** | **Horizontal Bar Chart** | `recipe_name` vs `Review_Count` | Ranking nominal categories with long labels. | Top comfort-food recipes (white chili, cookies, roasted chicken) exceed 700 reviews each. |
| **05** | **Pareto Chart** | Top 25 recipes, counts + cum % | Testing 80/20 concentration rule. | Top 25 recipes drive > 58% of all feedback; 42 recipes generate 80% of volume. |
| **06** | **Dual-Axis Line Chart** | `year_month` vs Count & Avg Rating | Longitudinal time-series trends across two metrics. | Review activity peaks heavily in winter holiday months (Nov–Jan); average rating stays constant (~4.3). |
| **07** | **Scatter + Regression** | `thumbs_up` vs `best_score` | Bivariate correlation & linear trend modeling. | Upvotes strongly correlate with algorithm best score ($r=0.52$); justifies omitting `best_score` to prevent leakage. |
| **08** | **Bubble Chart** | Avg Stars (X), Volume (Y), Engagement (Size), Rep (Color) | 4-variable entity profiling simultaneously. | Highly debated/controversial recipes generate larger bubbles (higher engagement) than universal 5-star items. |
| **09** | **Correlation Heatmap** | 9 numerical features | Pairwise linear dependencies matrix ($r \in [-1, 1]$). | Thumbs up and engagement total correlate at 0.94; star ratings have near-zero linear correlation with text length ($r=-0.05$). |
| **10** | **Hexbin Density Plot** | `thumbs_up` vs `thumbs_down` | 2D joint density under heavy origin overplotting. | Over 65% of reviews sit at $(0, 0)$ (zero engagement); positive endorsement is twice as frequent as downvoting. |
| **11** | **Word Cloud** | Text corpus of 5-star reviews | Qualitative text keyword prominence. | Prominent keywords: `family`, `husband`, `easy`, `substitute`, `chicken`, `garlic`, `minute`. |
| **12** | **Stacked 100% Bar** | `engagement_cat` vs `rating_cat` | Subgroup composition comparison across ordered tiers. | High-engagement reviews have 3x higher proportion of critical/low ratings (~24%) compared to unengaged reviews (~8%). |
| **13** | **Donut Chart** | `reputation_cat` proportions | Part-to-whole composition with visual emphasis. | `Rep_Novice`: 78.4%, `Rep_Intermediate`: 16.2%, `Rep_Veteran`: 5.4%. |
| **14** | **Network Graph** | Behavioral attribute tokens | Association topology & co-occurrence cliques. | Dense central clique between `Rating_High_5`, `Rep_Novice`, `Engage_Zero`, `No_Reply`. |

---

## 3. FP-Growth Association Rule Mining Cheat Sheet
- **Algorithm:** Frequent Pattern Growth (Han et al., 2000). Avoids candidate generation via FP-tree.
- **Transaction Formulation:** 6 discrete items per review record (`rating_cat`, `reputation_cat`, `engagement_cat`, `length_cat`, `feedback_cat`, `reply_cat`).
- **Tuned Thresholds:** $\text{min\_support} = 0.03$ (3%), $\text{min\_confidence} = 0.50$ (50%).
- **Outputs:** 313 frequent itemsets, 1,467 association rules (683 rules with Lift $\ge 1.05$).
- **Key Formulae:**
  - $\text{Support}(A \implies B) = \frac{\text{count}(A \cup B)}{N}$
  - $\text{Confidence}(A \implies B) = \frac{\text{count}(A \cup B)}{\text{count}(A)}$
  - $\text{Lift}(A \implies B) = \frac{\text{Confidence}(A \implies B)}{\text{Support}(B)}$
- **Top Rules by Lift:**
  - `{Feedback_Positive, Rep_Novice} => {Engage_Low}` : **Lift = 4.22**, Conf = 77.2%
  - `{Feedback_Positive} => {Engage_Low}` : **Lift = 4.13**, Conf = 75.5%
- **Baseline Rule:** `{Feedback_Neutral, Rep_Novice} => {Engage_Zero}` : Lift = 1.35, **Conf = 99.96%**.

---

## 4. K-Nearest Neighbors (KNN) Cheat Sheet
- **Target Variable:** `rating_3class` (0: Low [0-2], 1: Medium [3-4], 2: High [5]).
- **Features Used (9):** `user_reputation`, `reply_count`, `thumbs_up`, `thumbs_down`, `text_length`, `word_count`, `hour`, `month`, `is_weekend`.
- **Target Leakage Note:** `best_score` was strictly excluded because it is computed from ratings/upvotes.
- **Preprocessing:** 80/20 stratified train/test split; `StandardScaler` (z-score normalization).
- **Hyperparameter Grid:** $K \in [3, 5, 7, 9, 11, 15, 21]$, metrics: Euclidean ($L_2$) and Manhattan ($L_1$), weights: `uniform` and `distance`.
- **Optimal Model:** **$K = 7$, Metric = Euclidean, Weight = Distance**.
- **Metrics (Test Set $N=3,637$):**
  - **Accuracy:** **69.65%**
  - **Class 2 (High Stars):** Precision = 0.79, Recall = 0.87, **F1 = 0.83**
  - **Class 0 (Low Stars):** Precision = 0.25, Recall = 0.17, **F1 = 0.20**
  - **Class 1 (Medium Stars):** Precision = 0.17, Recall = 0.11, **F1 = 0.13**
  - **Macro F1:** **0.3813** | **Weighted F1:** **0.6974**
- **Exam Explanation:** High accuracy (73%) is driven by the majority class (76% 5-star reviews). Distance metrics struggle to separate minority classes due to overlapping continuous feature distributions.

---

## 5. AGNES Hierarchical Clustering Cheat Sheet
- **Algorithm:** Agglomerative Nesting (Kaufman & Rousseeuw, 1990) — bottom-up hierarchical clustering.
- **Features (6):** `user_reputation`, `thumbs_up`, `thumbs_down`, `reply_count`, `text_length`, `word_count` (Standardized).
- **Sampling Rationale:** Full dataset ($N=18,182$) requires 165.3 million pairwise distances ($O(N^2)$ memory, $O(N^3)$ linkage time), causing memory exhaustion. A statistically representative sample ($N=2,000$) was used.
- **Linkage Cophenetic Correlation Comparison:**
  - **Average Linkage:** **0.9572** (Best metric topology preservation)
  - **Complete Linkage:** **0.8762**
  - **Single Linkage:** **0.8399** (Suffers from chaining effect)
  - **Ward Linkage:** **0.5243** (Used for final cut: minimizes within-cluster variance)
- **Optimal Clusters ($k=4$, Ward Linkage):**
  - **Cluster 1 (81.8%):** Casual Reviewers (Avg Rep 2.0, Length 146 chars, Thumbs 0.46, Replies 0).
  - **Cluster 2 (15.1%):** Narrative Reviewers (Avg Length 507 chars, detailed culinary essays).
  - **Cluster 0 (3.0%):** High-Engagement Influencers (Avg Thumbs 18.5, Avg Replies 0.56).
  - **Cluster 3 (0.1%):** Platform Power User Outlier (Reputation 510).

---

## 6. ACLOSE Closed Itemsets Cheat Sheet
- **Definition:** An itemset $X$ is **Closed** iff no proper superset $Y \supset X$ has $\text{support}(Y) = \text{support}(X)$.
- **Closure Operator:** $c(X) = g(f(X))$. Itemset $X$ is closed iff $c(X) = X$.
- **Why Closed Itemsets Matter:** Lossless representation. Eliminates redundancy while preserving exact support counts for all frequent itemsets.
- **Verification:** Verified against an exhaustive brute-force closed itemset checker on a sample ($N=100$, minsup=0.10) $\to$ **100% exact match** (105 CFIs).
- **Redundancy Reduction Benchmark:**
  - **At minsup = 5%:** FI = 216 $\to$ CFI = 136 (**37.04% redundancy reduction**)
  - **At minsup = 10%:** FI = 145 $\to$ CFI = 81 (**44.14% redundancy reduction**)
  - **At minsup = 15%:** FI = 119 $\to$ CFI = 63 (**47.06% redundancy reduction**)
- **Top Closed Itemset:** `{Rating_High_5, Rep_Novice, No_Reply, Feedback_Neutral, Engage_Zero}` (Size 5, Support = 54.22%).

---

## 7. PySpark & Big Data Cheat Sheet
- **Spark Version:** PySpark 3.5.3 | Master: `local[*]` | JVM: OpenJDK 17.
- **MultiLine CSV Option:** Must pass `.option("multiLine", "true")` because review text contains embedded newlines (`\n`). Failure to do so corrupts line parsing.
- **Spark DataFrame Operations:**
  - `df.withColumn("created_timestamp", F.to_timestamp(F.from_unixtime("created_at")))`
  - `df.groupBy("recipe_code").agg(F.count("*"), F.avg("stars"))`
  - `FPGrowth(itemsCol="items", minSupport=0.03, minConfidence=0.50)`
- **Why KNN/AGNES/ACLOSE are Limited in Spark:**
  - **KNN:** Lacks native classifier in Spark MLlib because testing requires all-to-all pairwise distances across all partitions ($O(M \times N)$ network shuffling). Production alternative: `BucketedRandomProjectionLSH`.
  - **AGNES:** Inherently sequential greedy merging ($O(N^3)$), incompatible with MapReduce/Spark DAGs. Spark native alternative: `BisectingKMeans` (hierarchical divisive tree).
  - **ACLOSE:** Vertical tid-list intersections create heavy partition shuffle across distributed nodes.

---

## 8. Performance Benchmarking Cheat Sheet
- **Speedup Formula:** $\text{Speedup} = \frac{\text{Pandas Execution Time}}{\text{PySpark Execution Time}}$.
- **Empirical Speedup on 18,182 Rows (Single Local Node):**
  - Data Loading: Pandas (0.066s) vs PySpark (0.291s) $\to$ **Speedup = 0.23x**
  - Cleaning & Features: Pandas (0.017s) vs PySpark (0.162s) $\to$ **Speedup = 0.10x**
  - GroupBy Aggregation: Pandas (0.003s) vs PySpark (0.206s) $\to$ **Speedup = 0.02x**
  - FP-Growth Mining: Pandas (0.066s) vs PySpark (0.433s) $\to$ **Speedup = 0.15x**
- **Why Spark is Slower on Small Data (< 1 GB):**
  1. Py4J TCP socket serialization overhead between Python and the JVM.
  2. Catalyst query optimizer logical/physical plan generation latency (~100–250 ms fixed floor).
  3. DAG stage scheduling and multi-threading coordination.
  4. In contrast, Pandas executes contiguous in-memory C/NumPy arrays with zero IPC overhead.
- **When Does Spark Win?** At the **Big Data Crossover Threshold** ($> 10\text{ GB} - 100\text{ GB}$). Pandas crashes with `MemoryError` (OOM), whereas Spark distributes partitions across cluster nodes and handles disk spilling smoothly.
