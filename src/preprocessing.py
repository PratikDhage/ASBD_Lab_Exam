"""
Data Preprocessing Pipeline for Recipe Reviews and User Feedback Dataset
ASBD Lab Exam Project
"""

import os
import re
import html
import numpy as np
import pandas as pd

def clean_text(raw_text):
    """Clean and normalize review text: HTML entities, whitespace, invalid characters."""
    if pd.isna(raw_text) or raw_text is None:
        return ""
    text_str = str(raw_text)
    # Decode HTML entities (e.g. &amp;, &#39;, &quot;)
    text_str = html.unescape(text_str)
    # Remove excessive carriage returns and repeated whitespaces
    text_str = re.sub(r'[\r\n\t]+', ' ', text_str)
    text_str = re.sub(r'\s+', ' ', text_str)
    return text_str.strip()

def preprocess_data(raw_csv_path='dataset/dataset.csv', output_dir='outputs/cleaned_data'):
    """
    Executes full data cleaning, feature engineering, and task-specific dataset generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")

    # 1. Drop irrelevant index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("Dropped 'Unnamed: 0' column.")

    # 2. Check and handle missing review text
    missing_text_count = df['text'].isna().sum()
    print(f"Missing review text count: {missing_text_count}")
    df['text'] = df['text'].apply(clean_text)

    # 3. Handle duplicates
    # Check comment_id uniqueness
    dup_comments = df['comment_id'].duplicated().sum()
    print(f"Duplicate comment_ids: {dup_comments}")
    if dup_comments > 0:
        df = df.drop_duplicates(subset=['comment_id'], keep='first')
        print(f"Removed {dup_comments} duplicate comments.")

    # 4. Handle invalid / negative values in engagement metrics
    # Engagement counts cannot be negative
    engagement_cols = ['reply_count', 'thumbs_up', 'thumbs_down', 'user_reputation', 'best_score', 'stars']
    for col in engagement_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"Found {negative_count} negative values in {col}, replacing with median/0.")
            df[col] = df[col].clip(lower=0)

    # Validate stars range [0, 5]
    invalid_stars = ((df['stars'] < 0) | (df['stars'] > 5)).sum()
    if invalid_stars > 0:
        print(f"Found {invalid_stars} invalid star ratings outside [0, 5], clipping.")
        df['stars'] = df['stars'].clip(0, 5)

    # 5. Temporal Feature Extraction
    # created_at is unix epoch seconds
    df['datetime'] = pd.to_datetime(df['created_at'], unit='s', errors='coerce')
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['year_month'] = df['datetime'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    df['is_weekend'] = df['datetime'].dt.day_of_week.isin([5, 6]).astype(int)

    # 6. Feature Engineering
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(x.split()) if x else 0)
    df['engagement_total'] = df['thumbs_up'] + df['thumbs_down'] + df['reply_count']
    df['net_thumbs'] = df['thumbs_up'] - df['thumbs_down']
    df['approval_ratio'] = (df['thumbs_up']) / (df['thumbs_up'] + df['thumbs_down'] + 1e-5)
    df['has_reply'] = (df['reply_count'] > 0).astype(int)
    df['is_perfect_rating'] = (df['stars'] == 5).astype(int)

    # 7. Discretization / Binning for Rule Mining & Categorical Analysis
    # Rating categories
    def categorize_rating(star):
        if star == 0:
            return 'Rating_0_Unrated'
        elif star in [1, 2]:
            return 'Rating_Low_1_2'
        elif star in [3, 4]:
            return 'Rating_Medium_3_4'
        else:
            return 'Rating_High_5'

    df['rating_cat'] = df['stars'].apply(categorize_rating)

    # Engagement categories based on distribution
    def categorize_engagement(eng):
        if eng == 0:
            return 'Engage_Zero'
        elif eng <= 2:
            return 'Engage_Low'
        elif eng <= 6:
            return 'Engage_Medium'
        else:
            return 'Engage_High'

    df['engagement_cat'] = df['engagement_total'].apply(categorize_engagement)

    # Reputation categories
    def categorize_reputation(rep):
        if rep == 1:
            return 'Rep_Novice'
        elif rep <= 5:
            return 'Rep_Intermediate'
        else:
            return 'Rep_Veteran'

    df['reputation_cat'] = df['user_reputation'].apply(categorize_reputation)

    # Review length categories
    def categorize_length(length):
        if length < 80:
            return 'Len_Short'
        elif length <= 220:
            return 'Len_Medium'
        else:
            return 'Len_Long'

    df['length_cat'] = df['text_length'].apply(categorize_length)

    # Feedback category (Thumbs balance)
    def categorize_feedback(row):
        tu, td = row['thumbs_up'], row['thumbs_down']
        if tu == 0 and td == 0:
            return 'Feedback_Neutral'
        elif tu > td:
            return 'Feedback_Positive'
        elif td > tu:
            return 'Feedback_Critical'
        else:
            return 'Feedback_Mixed'

    df['feedback_cat'] = df.apply(categorize_feedback, axis=1)

    # Reply category
    df['reply_cat'] = df['has_reply'].apply(lambda x: 'Has_Reply' if x == 1 else 'No_Reply')

    # 8. Save Cleaned Dataset
    cleaned_path = os.path.join(output_dir, 'cleaned_dataset.csv')
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_path} with shape {df.shape}")

    # 9. Task-Specific Datasets

    # A) ARM / FP-Growth & ACLOSE Transactions
    # Each row becomes a transaction of meaningful categorical tokens
    transaction_cols = ['rating_cat', 'reputation_cat', 'engagement_cat', 'length_cat', 'feedback_cat', 'reply_cat']
    df_transactions = df[transaction_cols].copy()
    arm_path = os.path.join(output_dir, 'arm_transactions.csv')
    df_transactions.to_csv(arm_path, index=False)
    print(f"Saved ARM transactions to {arm_path}")

    # B) KNN Classification Dataset
    # Exclude IDs, user_name, comment_id, and best_score (to avoid data leakage as best_score is directly computed from ratings/engagement)
    knn_cols = [
        'user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down',
        'text_length', 'word_count', 'hour', 'month', 'is_weekend', 'stars'
    ]
    df_knn = df[knn_cols].copy()
    # Add a 3-class target: Low (0-2), Medium (3-4), High (5)
    def map_target_3class(s):
        if s <= 2:
            return 0  # Low / Critical
        elif s <= 4:
            return 1  # Moderate
        else:
            return 2  # High / Perfect
    df_knn['rating_3class'] = df_knn['stars'].apply(map_target_3class)
    knn_path = os.path.join(output_dir, 'knn_features.csv')
    df_knn.to_csv(knn_path, index=False)
    print(f"Saved KNN features to {knn_path}")

    # C) AGNES Clustering Dataset
    # Selected numerical features describing user engagement and content density
    agnes_cols = [
        'user_reputation', 'thumbs_up', 'thumbs_down', 'reply_count',
        'text_length', 'word_count', 'best_score'
    ]
    df_agnes = df[agnes_cols].copy()
    agnes_path = os.path.join(output_dir, 'agnes_features.csv')
    df_agnes.to_csv(agnes_path, index=False)
    print(f"Saved AGNES features to {agnes_path}")

    return df

if __name__ == '__main__':
    preprocess_data()
