# Descriptive Analytics & Exploratory Data Analysis Report
**Course:** Analytics and Systems of Big Data (ASBD)  
**Dataset:** Recipe Reviews and User Feedback Dataset (UCI Machine Learning Repository)  
**Artifact Directory:** `outputs/descriptive_statistics/` & `outputs/visualizations/`  

---

## 1. Executive Summary & Dataset Architecture
This report presents an exhaustive empirical descriptive investigation of the **Recipe Reviews and User Feedback Dataset**, sourced from the UCI Machine Learning Repository. The dataset encompasses authentic online recipe engagement spanning from **February 11, 2021 to October 14, 2022**.

### 1.1 Dataset Profile & Cardinality
- **Total Records:** 18,182 reviews
- **Raw Features:** 15 columns (including a redundant index column `Unnamed: 0` that was removed during cleaning)
- **Engineered Analytical Features:** 34 total columns
- **Unique Recipes:** 100 unique culinary recipes (e.g., `recipe_code` 1 to 100)
- **Unique Reviewers:** 13,812 distinct user accounts (`user_id`)
- **Review Content:** 18,182 distinct comment IDs with 2 null review text entries handled through empty text normalization.
- **Engagement Sum:** 19,805 total upvotes (`thumbs_up`), 9,988 downvotes (`thumbs_down`), and 266 author replies.

---

## 2. Quantitative Summary Statistics

The table below summarizes the central tendency, dispersion, and distribution shapes computed across quantitative metrics:

| Attribute | Mean | Median | Mode | Std Dev | Variance | Min | Q1 (25%) | Q3 (75%) | IQR | Max | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`stars`** | 4.2888 | 5.0 | 5.0 | 1.5449 | 2.3866 | 0 | 4.0 | 5.0 | 1.0 | 5 | -1.9796 | 2.6710 |
| **`user_reputation`** | 2.1596 | 1.0 | 1.0 | 10.0820 | 101.646 | 1 | 1.0 | 1.0 | 0.0 | 520 | 29.8335 | 1146.95 |
| **`reply_count`** | 0.0146 | 0.0 | 0.0 | 0.1378 | 0.0190 | 0 | 0.0 | 0.0 | 0.0 | 3 | 10.3705 | 129.215 |
| **`thumbs_up`** | 1.0893 | 0.0 | 0.0 | 4.1956 | 17.6033 | 0 | 0.0 | 0.0 | 0.0 | 106 | 9.7712 | 134.120 |
| **`thumbs_down`** | 0.5493 | 0.0 | 0.0 | 3.5414 | 12.5415 | 0 | 0.0 | 0.0 | 0.0 | 126 | 17.5878 | 417.433 |
| **`best_score`** | 153.16 | 100.0 | 100.0 | 140.85 | 19839.2 | 100 | 100.0 | 100.0 | 0.0 | 946 | 2.5020 | 5.5786 |
| **`text_length`** | 203.89 | 161.0 | 53.0 | 172.46 | 29742.2 | 0 | 90.0 | 266.0 | 176.0 | 2742 | 2.9126 | 15.3629 |
| **`word_count`** | 38.07 | 30.0 | 10.0 | 31.96 | 1021.2 | 0 | 17.0 | 50.0 | 33.0 | 502 | 2.8711 | 15.0112 |
| **`engagement_total`**| 1.6533 | 0.0 | 0.0 | 7.0421 | 49.5912 | 0 | 0.0 | 0.0 | 0.0 | 197 | 11.2335 | 165.702 |

### Key Statistical Observations:
1. **Severe Positive Skewness in Social Metrics:** `user_reputation` (skew = +29.83), `thumbs_up` (skew = +9.77), and `thumbs_down` (skew = +17.59) exhibit extreme Pareto-like distributions. The vast majority of reviewers have a reputation score of 1 ($IQR = 0$), while a small elite community of power users attains scores up to 520.
2. **Negative Skewness in Rating Distribution:** `stars` has a skewness of -1.98. Over 76% of all recorded reviews award the maximum possible rating of 5 stars, while 9.33% are 0-star reviews (users who submitted narrative text without an active rating click).
3. **Review Density:** The average review contains 203.9 characters (~38 words), but exhibits high kurtosis (15.36) with descriptive culinary essays reaching up to 2,742 characters.

---

## 3. Visualization Suite — 14 Unique Plot Archetypes

Every plot was generated using high-resolution rendering (300 DPI) and saved to `outputs/visualizations/`. Below is the complete analytical dissection of each plot type:

### Plot 1: Histogram with KDE Overlay (`01_histogram_stars_kde.png`)
- **Attributes Used:** `stars` (0 to 5)
- **Why Suitable:** Discrete rating choices are best evaluated through discrete frequency bars with a Kernel Density Estimate (KDE) curve overlay to highlight skewness.
- **What It Represents:** The distribution of recipe ratings across the entire user base.
- **Key Observations:** 13,829 reviews (76.06%) are 5-star ratings; 1,696 reviews (9.33%) are 0-star (unrated); 1,655 (9.10%) are 4-star; 1 to 3 stars constitute only 5.51% combined.
- **Analytical Inference:** Online recipe feedback suffers from the classic "J-shaped" or positive evaluation bias, where users only bother reviewing recipes they either loved or had a significant issue with.
- **Conclusion:** Binary or ordinal groupings must account for this extreme class imbalance during machine learning.

### Plot 2: Box Plot with Mean Markers (`02_boxplot_review_length_by_rating.png`)
- **Attributes Used:** `stars` (X-axis), `text_length` (Y-axis)
- **Why Suitable:** Visualizes five-number summaries (minimum, Q1, median, Q3, maximum) and outlier density across multiple independent categories simultaneously.
- **What It Represents:** The variation in review text length (character count) across star rating tiers.
- **Key Observations:** Reviews with 1-star and 2-star ratings have significantly higher median lengths (~220 chars) compared to 5-star reviews (~150 chars). The red mean markers exceed medians across all tiers, reflecting long-tail descriptive reviews.
- **Analytical Inference:** Dissatisfied or critical users write significantly longer reviews explaining substitute ingredients, baking failures, or technique modifications, whereas satisfied 5-star reviewers frequently leave concise compliments ("Loved this!", "Made it twice!").
- **Conclusion:** Text length serves as an informative predictor of critical versus routine positive feedback.

### Plot 3: Violin Plot with Quartile Cuts (`03_violinplot_reputation_by_rating.png`)
- **Attributes Used:** `rating_cat` (X-axis), `user_reputation` clipped at 15 (Y-axis)
- **Why Suitable:** Combines the box plot's summary statistics with a rotated non-parametric kernel density estimate to reveal multi-modal behavior that box plots hide.
- **What It Represents:** Reviewer community reputation density across categorical rating groups (`Rating_0_Unrated`, `Rating_Low_1_2`, `Rating_Medium_3_4`, `Rating_High_5`).
- **Key Observations:** The widest violin bulge sits at reputation = 1 across all rating categories. However, the `Rating_High_5` category has a broader tail of intermediate and veteran contributors (reputation 3–15).
- **Analytical Inference:** High ratings are driven by both newcomers and veteran members, whereas 0-star and low-star ratings are predominantly contributed by novice or guest accounts who experienced an immediate failure.
- **Conclusion:** Account maturity correlates with constructive platform engagement.

### Plot 4: Horizontal Bar Chart (`04_barchart_top_recipes.png`)
- **Attributes Used:** `recipe_name` (Y-axis), `Review_Count` (X-axis)
- **Why Suitable:** Horizontal bars provide ample typography space to read long recipe titles without overlap, ranked in descending frequency order.
- **What It Represents:** Top 15 most reviewed recipes across the platform.
- **Key Observations:** The most reviewed recipes exceed 700 reviews each, dominated by classic comfort foods like white chili beans, chocolate chip cookies, and roasted chicken.
- **Analytical Inference:** User traffic concentrates around staple, accessible comfort food recipes that are cooked repeatedly.
- **Conclusion:** System recommendations should leverage top-volume items as cold-start seeds.

### Plot 5: Pareto Chart (`05_pareto_chart_recipes.png`)
- **Attributes Used:** Top 25 `recipe_name`s, individual review counts (bars), cumulative percentage curve (line), 80% cutoff line.
- **Why Suitable:** The Pareto principle chart proves whether a small fraction of entities drive the majority of feedback volume (the 80/20 rule).
- **What It Represents:** Cumulative percentage contribution of top recipes to total review volume.
- **Key Observations:** The top 25 recipes account for over 58% of the platform's 18,182 reviews; 42 recipes satisfy the 80% volume threshold.
- **Analytical Inference:** Reviewer attention follows a power-law distribution. Platform health and average rating benchmarks depend heavily on the top third of recipes.
- **Conclusion:** Recipe quality assurance must prioritize top-tier traffic recipes.

### Plot 6: Dual-Axis Temporal Line Chart (`06_linechart_temporal_trends.png`)
- **Attributes Used:** `year_month` (X-axis), `Review_Count` (left Y-axis), `Avg_Rating` (right Y-axis).
- **Why Suitable:** Line charts effectively reveal continuous trends, seasonal cycles, and rate of change over time across two metrics with different units.
- **What It Represents:** Monthly review activity and average star rating trajectory from February 2021 to October 2022.
- **Key Observations:** Review volume peaks heavily in late autumn and winter months (November–January, coinciding with holiday cooking) and experiences a mid-summer dip. Concurrently, the average star rating remains remarkably stable between 4.15 and 4.38 stars.
- **Analytical Inference:** Platform traffic is highly seasonal, driven by winter holidays, but satisfaction remains constant across seasons.
- **Conclusion:** Resource autoscaling in production Big Data clusters should anticipate November–February traffic surges.

### Plot 7: Scatter Plot with Linear Regression (`07_scatterplot_thumbs_vs_bestscore.png`)
- **Attributes Used:** `thumbs_up` (X-axis), `best_score` (Y-axis), fitted linear regression line.
- **Why Suitable:** Depicts continuous bivariate correlation and reveals whether one metric is a direct mathematical linear function of another.
- **What It Represents:** The relationship between community upvotes and the algorithmic `best_score`.
- **Key Observations:** A strong linear relationship ($r = 0.52$, $p < 0.001$) exists with an upward regression slope. Reviews with thumbs up > 50 consistently reach best scores > 600.
- **Analytical Inference:** The platform's proprietary `best_score` formula weights social upvotes heavily as an indicator of high-value feedback.
- **Conclusion:** Confirms our decision to exclude `best_score` from KNN features to prevent predictive target leakage.

### Plot 8: Multi-Variable Bubble Chart (`08_bubblechart_recipe_engagement.png`)
- **Attributes Used:** X: Average Star Rating, Y: Total Reviews Received, Size: Total Engagement (Thumbs + Replies), Color: Average Reviewer Reputation.
- **Why Suitable:** Encodes four distinct dimensions into a single 2D visualization without loss of readability.
- **What It Represents:** Global recipe landscape comparing popularity, satisfaction, community interaction, and reviewer prestige.
- **Key Observations:** High-volume recipes (Y > 400) cluster tightly around Average Rating 4.1–4.5. The largest bubbles (highest social engagement) occur on moderately contested recipes rather than universal 5-star items.
- **Analytical Inference:** Recipes that generate debate or require technique adjustments stimulate the highest user feedback conversations.
- **Conclusion:** High engagement does not always equal high rating; controversial recipes generate the richest user discussions.

### Plot 9: Annotated Pearson Correlation Heatmap (`09_heatmap_correlation_matrix.png`)
- **Attributes Used:** `stars`, `user_reputation`, `reply_count`, `thumbs_up`, `thumbs_down`, `best_score`, `text_length`, `engagement_total`, `net_thumbs`.
- **Why Suitable:** Provides a comprehensive color-coded square matrix of all pairwise Pearson correlation coefficients ($r \in [-1, 1]$).
- **What It Represents:** The linear dependency between all engineered quantitative features.
- **Key Observations:**
  - `thumbs_up` and `engagement_total` have a near-perfect correlation ($r = 0.94$).
  - `thumbs_up` and `best_score` exhibit strong positive correlation ($r = 0.52$).
  - `stars` has virtually zero linear correlation with `text_length` ($r = -0.05$) and `thumbs_up` ($r = -0.02$).
- **Analytical Inference:** Social engagement is decoupled from star rating: highly upvoted reviews can be either effusive praises or detailed critical warnings.
- **Conclusion:** Predictive models cannot rely solely on simple linear features to classify ratings.

### Plot 10: Log-Scaled Hexbin Density Plot (`10_hexbin_thumbs_density.png`)
- **Attributes Used:** `thumbs_up` (X-axis), `thumbs_down` (Y-axis), logarithmic color scale.
- **Why Suitable:** Hexagonal binning solves extreme overplotting in large datasets where tens of thousands of points overlap near the origin.
- **What It Represents:** The joint bivariate interaction density of community endorsement (`thumbs_up`) vs criticism (`thumbs_down`).
- **Key Observations:** The origin cell $(0, 0)$ contains over 12,000 observations (Log10 > 4.0). Points that leave the origin spread predominantly along the X-axis (more thumbs up than thumbs down).
- **Analytical Inference:** Passive browsing dominates: over 65% of reviews receive zero upvotes or downvotes. When users do react, positive endorsement is twice as frequent as downvoting.
- **Conclusion:** Engagement data is zero-inflated and requires thresholding/discretization for pattern mining.

### Plot 11: Text Mining Word Cloud (`11_wordcloud_review_text.png`)
- **Attributes Used:** Text corpus of 5-star positive reviews (stop-words filtered).
- **Why Suitable:** Visualizes the most prominent semantic vocabulary in unstructured textual data.
- **What It Represents:** High-frequency culinary keywords associated with top-rated recipes.
- **Key Observations:** Dominant terms include `family`, `husband`, `easy`, `substitute`, `chicken`, `onion`, `garlic`, `baked`, `soup`, `minute`.
- **Analytical Inference:** Home cooks highly value family approval ("husband loved it") and ease of preparation ("easy weeknight meal"), frequently sharing minor substitution tweaks.
- **Conclusion:** Text features capture domestic emotional satisfaction that numeric metrics miss.

### Plot 12: Stacked 100% Bar Chart (`12_stacked_bar_rating_by_engagement.png`)
- **Attributes Used:** `engagement_cat` on X-axis (`Engage_Zero`, `Engage_Low`, `Engage_Medium`, `Engage_High`), stacked percentage of rating categories.
- **Why Suitable:** Compares the relative percentage composition of subgroups across an ordered independent factor.
- **What It Represents:** The distribution of rating tiers across levels of review engagement.
- **Key Observations:** As engagement increases from `Engage_Zero` to `Engage_High`, the proportion of `Rating_0_Unrated` and `Rating_Low_1_2` reviews grows from ~8% to over 24%.
- **Analytical Inference:** Highly engaged reviews (those with many upvotes, downvotes, or replies) are substantially more critical than unengaged reviews. Reviews that point out flaws attract greater community discussion.
- **Conclusion:** Social engagement acts as a filter that amplifies detailed critical reviews.

### Plot 13: Community Reputation Donut Chart (`13_donut_chart_reputation_segments.png`)
- **Attributes Used:** `reputation_cat` (`Rep_Novice`, `Rep_Intermediate`, `Rep_Veteran`).
- **Why Suitable:** Displays proportional composition of a whole with a modern hollow center that emphasizes segment percentages.
- **What It Represents:** Proportions of novice, intermediate, and veteran reviewers in the community.
- **Key Observations:** `Rep_Novice` accounts for 78.4% of all reviews; `Rep_Intermediate` accounts for 16.2%; `Rep_Veteran` accounts for 5.4%.
- **Analytical Inference:** The platform relies on casual, one-time or infrequent reviewers for volume, while a dedicated core of 5.4% power users provides continuous high-reputation activity.
- **Conclusion:** User segmentation is essential when modeling user lifetime value or credibility weighting.

### Plot 14: Behavioral Co-occurrence Network Graph (`14_network_graph_association_rules.png`)
- **Attributes Used:** Discretized transactional tokens (`Rating_High_5`, `Rep_Novice`, `Engage_Zero`, `No_Reply`, `Feedback_Neutral`, etc.), co-occurrence edge weights.
- **Why Suitable:** Graph visualization exposes topological structures, hub nodes, and attribute communities in transactional data.
- **What It Represents:** The interaction topology connecting reviewer status, feedback, and ratings.
- **Key Observations:** A massive central clique connects `Rating_High_5` $\leftrightarrow$ `Rep_Novice` $\leftrightarrow$ `No_Reply` $\leftrightarrow$ `Engage_Zero`. Peripheral nodes connect `Engage_High` to `Has_Reply` and `Rep_Veteran`.
- **Analytical Inference:** The baseline transaction in online recipe communities is an unengaged novice user awarding 5 stars without receiving replies. Meaningful discussions form distinct peripheral clusters.
- **Conclusion:** Lays the theoretical foundation for FP-Growth and ACLOSE itemset mining.
