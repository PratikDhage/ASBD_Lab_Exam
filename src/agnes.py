"""
Hierarchical Clustering — AGNES (Agglomerative Nesting)
ASBD Lab Exam Project
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

def run_agnes_clustering(features_csv='outputs/cleaned_data/agnes_features.csv', output_dir='outputs/clustering', sample_size=2000):
    os.makedirs(output_dir, exist_ok=True)
    df_raw = pd.read_csv(features_csv)
    print(f"Loaded {len(df_raw)} records for AGNES hierarchical clustering.")

    # 1. Feature Selection
    feature_cols = ['user_reputation', 'thumbs_up', 'thumbs_down', 'reply_count', 'text_length', 'word_count']
    X_full = df_raw[feature_cols].copy()

    # 2. Computational Feasibility & Sampling
    # N=18,182 requires ~165.3 million pairwise distance calculations.
    # We draw a statistically robust random sample (N=2000, random_state=42)
    print(f"Sampling {sample_size} records from {len(X_full)} for memory-safe hierarchical agglomeration...")
    df_sample = X_full.sample(n=sample_size, random_state=42).copy()

    # 3. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample)

    # 4. Pairwise Distance & Cophenetic Correlation across Linkages
    print("\n--- Evaluating Linkage Methods via Cophenetic Correlation ---")
    pairwise_distances = pdist(X_scaled, metric='euclidean')
    linkage_methods = ['single', 'complete', 'average', 'ward']
    cophenetic_scores = {}

    linkage_matrices = {}
    for method in linkage_methods:
        Z = linkage(X_scaled, method=method, metric='euclidean')
        linkage_matrices[method] = Z
        c_score, _ = cophenet(Z, pairwise_distances)
        cophenetic_scores[method] = round(c_score, 4)
        print(f"Linkage: {method.capitalize():<10} -> Cophenetic Correlation Coefficient: {c_score:.4f}")

    df_coph = pd.DataFrame(list(cophenetic_scores.items()), columns=['Linkage_Method', 'Cophenetic_Correlation'])
    df_coph.to_csv(os.path.join(output_dir, 'linkage_cophenetic_scores.csv'), index=False)

    # 5. Dendrogram Visualization (Ward Linkage - Minimizes within-cluster variance)
    plt.figure(figsize=(12, 6), dpi=300)
    plt.title("AGNES Hierarchical Clustering Dendrogram (Ward Linkage, Truncated to 30 Merges)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Sample Index or (Cluster Size)", fontsize=11)
    plt.ylabel("Ward Distance / Dissimilarity Threshold", fontsize=11)
    dendrogram(
        linkage_matrices['ward'],
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90.,
        leaf_font_size=9.,
        show_contracted=True
    )
    plt.axhline(y=25, color='r', linestyle='--', linewidth=1.5, label='Optimal Cluster Cut-off (k=4)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agnes_dendrogram_ward.png'))
    plt.close()
    print("Saved dendrogram plot to agnes_dendrogram_ward.png")

    # 6. Fit AGNES AgglomerativeClustering with Optimal k=4
    k_optimal = 4
    agnes = AgglomerativeClustering(n_clusters=k_optimal, metric='euclidean', linkage='ward')
    df_sample['Cluster'] = agnes.fit_predict(X_scaled)

    # 7. Cluster Profiles & Characterization
    cluster_profiles = df_sample.groupby('Cluster').agg({
        'user_reputation': ['count', 'mean', 'median'],
        'thumbs_up': ['mean', 'median'],
        'thumbs_down': ['mean', 'median'],
        'reply_count': ['mean', 'median'],
        'text_length': ['mean', 'median'],
        'word_count': ['mean', 'median']
    })
    cluster_profiles.to_csv(os.path.join(output_dir, 'agnes_cluster_profiles.csv'))
    print("Saved cluster profile summary to agnes_cluster_profiles.csv")

    # Interpret clusters
    print("\n--- AGNES Cluster Characteristics (Sample Size = 2000) ---")
    for c in range(k_optimal):
        sub = df_sample[df_sample['Cluster'] == c]
        print(f"Cluster {c} (Size={len(sub)}, {len(sub)/len(df_sample)*100:.1f}%): "
              f"Avg Rep={sub['user_reputation'].mean():.1f}, "
              f"Avg Len={sub['text_length'].mean():.1f} chars, "
              f"Avg ThumbsUp={sub['thumbs_up'].mean():.2f}, "
              f"Avg Replies={sub['reply_count'].mean():.3f}")

    # 8. 2D Visualization using PCA Projection
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_sample['PCA1'] = X_pca[:, 0]
    df_sample['PCA2'] = X_pca[:, 1]

    plt.figure(figsize=(9, 6), dpi=300)
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Cluster', data=df_sample,
        palette='Set1', alpha=0.8, s=45, edgecolor='black', linewidth=0.5
    )
    plt.title(f'AGNES Clusters Projected onto First 2 Principal Components (k={k_optimal})', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=11)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=11)
    plt.legend(title='AGNES Cluster', loc='upper right')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agnes_pca_cluster_scatter.png'))
    plt.close()
    print("Saved PCA cluster scatter plot.")

    # Save sampled data with cluster assignments
    df_sample.to_csv(os.path.join(output_dir, 'agnes_clustered_sample.csv'), index=False)
    return agnes, df_sample

if __name__ == '__main__':
    run_agnes_clustering()
