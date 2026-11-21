"""
K-Nearest Neighbors (KNN) Supervised Classification
ASBD Lab Exam Project
"""

import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def run_knn_classification(features_csv='outputs/cleaned_data/knn_features.csv', output_dir='outputs/knn'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} records for KNN classification.")

    # 1. Feature and Target Selection
    # Features exclude IDs, comment text, and exclude 'best_score' to prevent Target Leakage
    feature_cols = ['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'text_length', 'word_count', 'hour', 'month', 'is_weekend']
    X = df[feature_cols].copy()
    
    # Target: 3-class rating classification (0: Critical/Low [0-2], 1: Medium [3-4], 2: High/5-Star)
    y = df['rating_3class'].copy()
    target_names = ['Class 0: Low (0-2 Stars)', 'Class 1: Medium (3-4 Stars)', 'Class 2: High (5 Stars)']

    print(f"Features: {feature_cols}")
    print(f"Class distribution:\n{y.value_counts(normalize=True).round(4) * 100}")

    # 2. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 3. Feature Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Hyperparameter Exploration across K, Metric, and Weighting
    k_values = [3, 5, 7, 9, 11, 15, 21]
    metrics = ['euclidean', 'manhattan']
    weights_list = ['uniform', 'distance']

    tuning_results = []
    best_f1 = -1.0
    best_model = None
    best_params = {}

    print("\n--- Tuning KNN Hyperparameters ---")
    for metric in metrics:
        for weight in weights_list:
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
                knn.fit(X_train_scaled, y_train)
                y_pred = knn.predict(X_test_scaled)

                acc = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
                rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

                tuning_results.append({
                    'K': k,
                    'Metric': metric,
                    'Weight': weight,
                    'Accuracy': round(acc, 4),
                    'Macro_Precision': round(prec_macro, 4),
                    'Macro_Recall': round(rec_macro, 4),
                    'Macro_F1': round(macro_f1, 4),
                    'Weighted_F1': round(weighted_f1, 4)
                })

                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_model = knn
                    best_params = {'K': k, 'Metric': metric, 'Weight': weight}

    df_tuning = pd.DataFrame(tuning_results)
    tuning_csv = os.path.join(output_dir, 'knn_hyperparameter_tuning.csv')
    df_tuning.to_csv(tuning_csv, index=False)
    print(f"Saved hyperparameter tuning results to {tuning_csv}")
    print(f"Best Configuration: {best_params} with Test Macro F1: {best_f1:.4f}")

    # 5. Evaluate Best Model
    y_pred_best = best_model.predict(X_test_scaled)
    acc_best = accuracy_score(y_test, y_pred_best)
    report_dict = classification_report(y_test, y_pred_best, target_names=target_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(output_dir, 'knn_classification_report.csv'))

    cm = confusion_matrix(y_test, y_pred_best)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    df_cm.to_csv(os.path.join(output_dir, 'knn_confusion_matrix.csv'))

    # Save model and scaler
    joblib.dump(best_model, os.path.join(output_dir, 'best_knn_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'knn_scaler.joblib'))
    print("Saved trained model and scaler to disk.")

    # 6. Visualizations
    # A) Accuracy and F1 vs. K curve
    plt.figure(figsize=(10, 5), dpi=300)
    for metric, style, color in [('euclidean', '-', '#1f77b4'), ('manhattan', '--', '#e67e22')]:
        sub = df_tuning[(df_tuning['Metric'] == metric) & (df_tuning['Weight'] == 'distance')]
        plt.plot(sub['K'], sub['Accuracy'], marker='o', linestyle=style, color=color, label=f'Accuracy ({metric.title()})')
        plt.plot(sub['K'], sub['Macro_F1'], marker='s', linestyle=style, alpha=0.7, label=f'Macro F1 ({metric.title()})')

    plt.title('KNN Performance vs. K Value (Distance-Weighted)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Number of Neighbors (K)', fontsize=11)
    plt.ylabel('Score Metric', fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle=':')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'knn_performance_vs_k.png'))
    plt.close()

    # B) Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title(f'Confusion Matrix: Best KNN (K={best_params["K"]}, {best_params["Metric"].title()}, {best_params["Weight"].title()})',
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.ylabel('Actual Class', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'knn_confusion_matrix.png'))
    plt.close()
    print("Saved KNN visualization plots.")

    return best_model, best_params, df_tuning

if __name__ == '__main__':
    run_knn_classification()
