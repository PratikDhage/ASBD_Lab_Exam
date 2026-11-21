"""
Descriptive Statistics Generator for Recipe Reviews and User Feedback Dataset
ASBD Lab Exam Project
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

def compute_descriptive_statistics(cleaned_csv='outputs/cleaned_data/cleaned_dataset.csv', output_dir='outputs/descriptive_statistics'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(cleaned_csv)
    print(f"Loaded dataset with shape {df.shape} for descriptive analytics.")

    # 1. Numerical Feature Analysis
    num_cols = [
        'stars', 'user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down',
        'best_score', 'text_length', 'word_count', 'engagement_total', 'net_thumbs', 'approval_ratio'
    ]

    num_summary = []
    for col in num_cols:
        series = df[col].dropna()
        mean_val = series.mean()
        std_val = series.std()
        var_val = series.var()
        median_val = series.median()
        mode_series = series.mode()
        mode_val = mode_series.iloc[0] if len(mode_series) > 0 else np.nan
        min_val = series.min()
        max_val = series.max()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr_val = q75 - q25
        skew_val = series.skew()
        kurt_val = series.kurtosis()

        num_summary.append({
            'Attribute': col,
            'Count': len(series),
            'Mean': round(mean_val, 4),
            'Median': round(median_val, 4),
            'Mode': round(mode_val, 4),
            'Std_Dev': round(std_val, 4),
            'Variance': round(var_val, 4),
            'Min': round(min_val, 4),
            'Q1_25%': round(q25, 4),
            'Q3_75%': round(q75, 4),
            'IQR': round(iqr_val, 4),
            'Max': round(max_val, 4),
            'Skewness': round(skew_val, 4),
            'Kurtosis': round(kurt_val, 4)
        })

    df_num_summary = pd.DataFrame(num_summary)
    num_csv_path = os.path.join(output_dir, 'numerical_summary_statistics.csv')
    df_num_summary.to_csv(num_csv_path, index=False)
    print(f"Saved numerical statistics to {num_csv_path}")

    # 2. Categorical Analysis
    cat_cols = ['recipe_name', 'rating_cat', 'engagement_cat', 'reputation_cat', 'length_cat', 'feedback_cat', 'reply_cat', 'day_of_week']
    for cat in cat_cols:
        counts = df[cat].value_counts()
        percents = df[cat].value_counts(normalize=True) * 100
        cat_df = pd.DataFrame({
            'Category': counts.index,
            'Frequency': counts.values,
            'Percentage': percents.round(2).values
        })
        cat_df.to_csv(os.path.join(output_dir, f'categorical_{cat}_distribution.csv'), index=False)

    # 3. Correlation & Covariance Matrices
    corr_pearson = df[num_cols].corr(method='pearson')
    corr_spearman = df[num_cols].corr(method='spearman')
    cov_matrix = df[num_cols].cov()

    corr_pearson.to_csv(os.path.join(output_dir, 'correlation_pearson.csv'))
    corr_spearman.to_csv(os.path.join(output_dir, 'correlation_spearman.csv'))
    cov_matrix.to_csv(os.path.join(output_dir, 'covariance_matrix.csv'))
    print("Saved correlation and covariance matrices.")

    # 4. Temporal Analysis
    temporal_summary = df.groupby('year_month').agg(
        Review_Count=('comment_id', 'count'),
        Avg_Rating=('stars', 'mean'),
        Median_Rating=('stars', 'median'),
        Avg_Thumbs_Up=('thumbs_up', 'mean'),
        Avg_Engagement=('engagement_total', 'mean')
    ).reset_index().sort_values('year_month')
    temporal_summary.to_csv(os.path.join(output_dir, 'temporal_monthly_trends.csv'), index=False)

    # 5. Recipe Performance Profile (Top 20 recipes)
    recipe_profile = df.groupby(['recipe_code', 'recipe_name']).agg(
        Review_Count=('comment_id', 'count'),
        Avg_Rating=('stars', 'mean'),
        Perfect_Rating_Ratio=('is_perfect_rating', 'mean'),
        Total_Thumbs_Up=('thumbs_up', 'sum'),
        Avg_Engagement=('engagement_total', 'mean'),
        Avg_Review_Length=('text_length', 'mean')
    ).reset_index().sort_values('Review_Count', ascending=False)
    recipe_profile.to_csv(os.path.join(output_dir, 'recipe_performance_summary.csv'), index=False)
    print("Saved recipe performance summary.")

    # 6. Overall Metadata Profile JSON
    profile = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'unique_recipes': int(df['recipe_name'].nunique()),
        'unique_users': int(df['user_id'].nunique()),
        'unique_comments': int(df['comment_id'].nunique()),
        'date_range_start': str(df['datetime'].min()),
        'date_range_end': str(df['datetime'].max()),
        'stars_breakdown': {str(k): int(v) for k, v in df['stars'].value_counts().items()},
        'mean_rating': float(round(df['stars'].mean(), 4)),
        'median_rating': float(df['stars'].median()),
        'mean_user_reputation': float(round(df['user_reputation'].mean(), 4)),
        'mean_text_length': float(round(df['text_length'].mean(), 4)),
        'total_engagement_events': int(df['engagement_total'].sum())
    }
    with open(os.path.join(output_dir, 'dataset_profile.json'), 'w') as f:
        json.dump(profile, f, indent=4)
    print("Saved dataset_profile.json.")

if __name__ == '__main__':
    compute_descriptive_statistics()
