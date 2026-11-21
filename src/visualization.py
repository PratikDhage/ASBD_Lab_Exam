"""
Comprehensive Visualizations Suite (14 Unique Plot Types)
ASBD Lab Exam Project
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import networkx as nx

def generate_all_visualizations(cleaned_csv='outputs/cleaned_data/cleaned_dataset.csv', output_dir='outputs/visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(cleaned_csv)
    print(f"Loaded {len(df)} rows for visualization suite.")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    palette = sns.color_palette("deep")

    # -------------------------------------------------------------
    # 1. HISTOGRAM: Star Rating Distribution with KDE Overlay
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    ax = sns.histplot(df['stars'], discrete=True, kde=True, color='#2b5c8f', edgecolor='black', alpha=0.7)
    plt.title('Plot 1: Distribution of Recipe Ratings (Stars 0-5) with KDE Curve', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Star Rating (0 = Unrated / Neutral, 1-5 = Explicit Rating)', fontsize=11)
    plt.ylabel('Review Frequency', fontsize=11)
    plt.xticks(range(6))
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{int(h)} ({h/len(df)*100:.1f}%)', (p.get_x() + p.get_width() / 2., h + 150),
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_histogram_stars_kde.png'))
    plt.close()
    print("Generated Plot 1: Histogram with KDE")

    # -------------------------------------------------------------
    # 2. BOX PLOT: Review Text Length Spread and Outliers by Star Rating
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    sns.boxplot(x='stars', y='text_length', data=df, palette='Spectral', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"8"})
    plt.title('Plot 2: Spread and Outliers of Review Text Length across Rating Classes', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Star Rating', fontsize=11)
    plt.ylabel('Review Text Length (Characters)', fontsize=11)
    plt.ylim(0, 1200) # Trim extreme display range for readability while keeping box statistics accurate
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_boxplot_review_length_by_rating.png'))
    plt.close()
    print("Generated Plot 2: Box Plot")

    # -------------------------------------------------------------
    # 3. VIOLIN PLOT: User Reputation Density across Rating Categories
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    # Clip reputation for visualization readability of density kernel
    df_temp = df.copy()
    df_temp['reputation_clipped'] = df_temp['user_reputation'].clip(upper=15)
    sns.violinplot(x='rating_cat', y='reputation_clipped', data=df_temp, palette='Set2', inner='quartile', cut=0)
    plt.title('Plot 3: Violin Plot of User Reputation Density across Rating Categories', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Rating Category', fontsize=11)
    plt.ylabel('User Reputation Score (Clipped at 15 for Kernel Visibility)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_violinplot_reputation_by_rating.png'))
    plt.close()
    print("Generated Plot 3: Violin Plot")

    # -------------------------------------------------------------
    # 4. HORIZONTAL BAR CHART: Top 15 Most Reviewed Recipes
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 7), dpi=300)
    top_recipes = df['recipe_name'].value_counts().head(15).reset_index()
    top_recipes.columns = ['Recipe', 'Review_Count']
    sns.barplot(x='Review_Count', y='Recipe', data=top_recipes, palette='viridis')
    plt.title('Plot 4: Top 15 Most Reviewed Recipes in Dataset', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Number of Reviews', fontsize=11)
    plt.ylabel('Recipe Name', fontsize=11)
    for index, value in enumerate(top_recipes['Review_Count']):
        plt.text(value + 3, index, f'{value}', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_barchart_top_recipes.png'))
    plt.close()
    print("Generated Plot 4: Bar Chart")

    # -------------------------------------------------------------
    # 5. PARETO CHART: Cumulative Review Volume of Top 25 Recipes (80/20 Rule)
    # -------------------------------------------------------------
    recipe_counts = df['recipe_name'].value_counts().head(25).reset_index()
    recipe_counts.columns = ['Recipe', 'Count']
    recipe_counts['CumPercentage'] = recipe_counts['Count'].cumsum() / df.shape[0] * 100

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    ax2 = ax1.twinx()
    ax1.bar(recipe_counts['Recipe'], recipe_counts['Count'], color='#3470a3', edgecolor='black', alpha=0.85)
    ax2.plot(recipe_counts['Recipe'], recipe_counts['CumPercentage'], color='#c9302c', marker='D', linewidth=2, markersize=5)
    ax2.axhline(80, color='darkgreen', linestyle='--', linewidth=1.5, label='80% Pareto Threshold')

    ax1.set_xticklabels(recipe_counts['Recipe'], rotation=75, ha='right', fontsize=8)
    ax1.set_ylabel('Review Count (Bars)', fontsize=11, color='#3470a3')
    ax2.set_ylabel('Cumulative Percentage of Total Dataset (%)', fontsize=11, color='#c9302c')
    plt.title('Plot 5: Pareto Chart of Review Concentration Across Top 25 Recipes', fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='lower right')
    ax1.grid(axis='y', linestyle=':')
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_pareto_chart_recipes.png'))
    plt.close()
    print("Generated Plot 5: Pareto Chart")

    # -------------------------------------------------------------
    # 6. LINE CHART: Temporal Monthly Review Volume & Avg Rating Trend
    # -------------------------------------------------------------
    monthly = df.groupby('year_month').agg(
        Review_Count=('comment_id', 'count'),
        Avg_Rating=('stars', 'mean')
    ).reset_index().sort_values('year_month')

    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=300)
    ax2 = ax1.twinx()
    ax1.plot(monthly['year_month'], monthly['Review_Count'], color='#1f77b4', marker='o', linewidth=2.5, label='Monthly Review Count')
    ax2.plot(monthly['year_month'], monthly['Avg_Rating'], color='#ff7f0e', marker='s', linewidth=2.5, linestyle='--', label='Average Rating')

    ax1.set_ylabel('Monthly Review Volume', color='#1f77b4', fontsize=11)
    ax2.set_ylabel('Average Star Rating', color='#ff7f0e', fontsize=11)
    ax1.set_xticklabels(monthly['year_month'], rotation=45, ha='right', fontsize=9)
    plt.title('Plot 6: Temporal Trends in Monthly Review Volume and Average Star Rating', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_linechart_temporal_trends.png'))
    plt.close()
    print("Generated Plot 6: Line Chart")

    # -------------------------------------------------------------
    # 7. SCATTER PLOT WITH REGRESSION: Thumbs Up vs. Best Score
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    sns.regplot(x='thumbs_up', y='best_score', data=df,
                scatter_kws={'alpha':0.4, 'color':'#207398', 's':25},
                line_kws={'color':'#e74c3c', 'linewidth':2.5})
    plt.title('Plot 7: Scatter Plot with Fitted Regression: Thumbs Up vs. Best Score', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Thumbs Up Count (User Endorsement)', fontsize=11)
    plt.ylabel('Algorithm Best Score', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_scatterplot_thumbs_vs_bestscore.png'))
    plt.close()
    print("Generated Plot 7: Scatter Plot with Regression")

    # -------------------------------------------------------------
    # 8. BUBBLE CHART: Recipe-Level Analysis (Avg Rating vs Volume vs Engagement)
    # -------------------------------------------------------------
    recipe_agg = df.groupby('recipe_name').agg(
        Avg_Rating=('stars', 'mean'),
        Review_Count=('comment_id', 'count'),
        Total_Engagement=('engagement_total', 'sum'),
        Avg_Reputation=('user_reputation', 'mean')
    ).reset_index()

    plt.figure(figsize=(10, 6), dpi=300)
    scatter = plt.scatter(
        recipe_agg['Avg_Rating'],
        recipe_agg['Review_Count'],
        s=recipe_agg['Total_Engagement'] * 0.8 + 30, # Size proportional to total engagement
        c=recipe_agg['Avg_Reputation'],
        cmap='plasma',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Reviewer Reputation', fontsize=10)
    plt.title('Plot 8: Recipe Performance Bubble Chart (Size = Total Engagement)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Average Star Rating', fontsize=11)
    plt.ylabel('Total Reviews Received', fontsize=11)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_bubblechart_recipe_engagement.png'))
    plt.close()
    print("Generated Plot 8: Bubble Chart")

    # -------------------------------------------------------------
    # 9. CORRELATION HEATMAP: Pearson Matrix of Numerical Attributes
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8), dpi=300)
    corr_cols = ['stars', 'user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'best_score', 'text_length', 'engagement_total', 'net_thumbs']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Plot 9: Pearson Correlation Matrix Heatmap across Numerical Metrics', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_heatmap_correlation_matrix.png'))
    plt.close()
    print("Generated Plot 9: Correlation Heatmap")

    # -------------------------------------------------------------
    # 10. HEXBIN PLOT: 2D Point Density of Thumbs Up vs. Thumbs Down
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    # Clip for visual density inspection
    plt.hexbin(df['thumbs_up'].clip(upper=25), df['thumbs_down'].clip(upper=25),
               gridsize=25, cmap='Blues', mincnt=1, bins='log')
    cb = plt.colorbar()
    cb.set_label('Log10(Point Count in Hexagon Cell)', fontsize=10)
    plt.title('Plot 10: Hexbin Plot of Thumbs Up vs. Thumbs Down Interaction Density', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Thumbs Up Count (Clipped at 25)', fontsize=11)
    plt.ylabel('Thumbs Down Count (Clipped at 25)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_hexbin_thumbs_density.png'))
    plt.close()
    print("Generated Plot 10: Hexbin Plot")

    # -------------------------------------------------------------
    # 11. WORD CLOUD: High-Frequency Vocabulary in 5-Star Reviews
    # -------------------------------------------------------------
    positive_text = " ".join(df[df['stars'] == 5]['text'].dropna().sample(min(3000, len(df[df['stars'] == 5])), random_state=42))
    stopwords = set(['recipe', 'make', 'made', 'used', 'cook', 'good', 'great', 'delicious', 'taste', 'flavor', 'one', 'add', 'time'])
    wc = WordCloud(width=1000, height=500, background_color='white', stopwords=stopwords, colormap='Dark2', max_words=120).generate(positive_text)
    plt.figure(figsize=(11, 5.5), dpi=300)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Plot 11: Word Cloud of Prominent Culinary Terms in 5-Star Reviews', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_wordcloud_review_text.png'))
    plt.close()
    print("Generated Plot 11: Word Cloud")

    # -------------------------------------------------------------
    # 12. STACKED 100% BAR CHART: Rating Proportion by Engagement Tier
    # -------------------------------------------------------------
    crosstab_pct = pd.crosstab(df['engagement_cat'], df['rating_cat'], normalize='index') * 100
    # Reorder index logically
    order_idx = [x for x in ['Engage_Zero', 'Engage_Low', 'Engage_Medium', 'Engage_High'] if x in crosstab_pct.index]
    crosstab_pct = crosstab_pct.loc[order_idx]

    plt.figure(figsize=(10, 6), dpi=300)
    crosstab_pct.plot(kind='bar', stacked=True, colormap='tab20c', edgecolor='black', figsize=(10, 6), ax=plt.gca())
    plt.title('Plot 12: Stacked 100% Bar Chart of Rating Distributions Across Engagement Tiers', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Engagement Tier', fontsize=11)
    plt.ylabel('Percentage Composition (%)', fontsize=11)
    plt.legend(title='Rating Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '12_stacked_bar_rating_by_engagement.png'))
    plt.close()
    print("Generated Plot 12: Stacked Bar Chart")

    # -------------------------------------------------------------
    # 13. DONUT / PIE CHART: User Reputation Tier Proportions
    # -------------------------------------------------------------
    rep_counts = df['reputation_cat'].value_counts()
    colors = ['#4a90e2', '#50e3c2', '#f5a623']
    plt.figure(figsize=(7, 7), dpi=300)
    plt.pie(rep_counts, labels=rep_counts.index, autopct='%1.1f%%', startangle=140,
            colors=colors, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    plt.title('Plot 13: Donut Chart of Reviewer Community Reputation Proportions', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '13_donut_chart_reputation_segments.png'))
    plt.close()
    print("Generated Plot 13: Donut Chart")

    # -------------------------------------------------------------
    # 14. NETWORK GRAPH: Attribute Association Rule Structure
    # -------------------------------------------------------------
    G = nx.Graph()
    # Define strong item co-occurrences found in transactions
    edges = [
        ('Rating_High_5', 'Rep_Novice', 10),
        ('Rating_High_5', 'Engage_Zero', 8),
        ('Rating_High_5', 'No_Reply', 12),
        ('Rating_High_5', 'Feedback_Neutral', 7),
        ('Rating_High_5', 'Len_Medium', 6),
        ('Rep_Novice', 'Engage_Zero', 9),
        ('Rep_Novice', 'No_Reply', 11),
        ('Engage_Zero', 'No_Reply', 10),
        ('Engage_Zero', 'Feedback_Neutral', 9),
        ('Rating_0_Unrated', 'Feedback_Critical', 4),
        ('Engage_High', 'Has_Reply', 5),
        ('Rep_Veteran', 'Engage_High', 3)
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    plt.figure(figsize=(9, 7), dpi=300)
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=1800, node_color='#76d7c4', edgecolors='#117864', linewidths=2)
    nx.draw_networkx_edges(G, pos, width=[w * 0.4 for w in weights], edge_color='#566573', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif', font_weight='bold')
    plt.title('Plot 14: Co-occurrence Network Graph of Review Behavioral Attributes', fontsize=13, fontweight='bold', pad=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '14_network_graph_association_rules.png'))
    plt.close()
    print("Generated Plot 14: Network Graph")

    print(f"All 14 unique plots saved successfully in {output_dir}/")

if __name__ == '__main__':
    generate_all_visualizations()
