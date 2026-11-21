"""
Association Rule Mining with FP-Growth
ASBD Lab Exam Project
"""

import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def run_fpgrowth_analysis(transactions_csv='outputs/cleaned_data/arm_transactions.csv', output_dir='outputs/association_rules'):
    os.makedirs(output_dir, exist_ok=True)
    df_raw = pd.read_csv(transactions_csv)
    print(f"Loaded {len(df_raw)} transaction records.")

    # 1. Convert DataFrame rows into lists of items (basket format)
    transactions = df_raw.values.tolist()

    # 2. One-hot encoding via TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    print(f"One-hot encoded transaction matrix shape: {df_encoded.shape} (Items: {len(te.columns_)})")

    # 3. Parameter Experimentation: Multiple Support Thresholds
    support_thresholds = [0.02, 0.03, 0.05, 0.10, 0.15]
    support_benchmarks = []

    print("\n--- Testing Multiple Minimum Support Thresholds ---")
    for min_sup in support_thresholds:
        t0 = time.time()
        fi = fpgrowth(df_encoded, min_support=min_sup, use_colnames=True)
        elapsed = time.time() - t0
        itemset_count = len(fi)
        max_len = fi['itemsets'].apply(len).max() if itemset_count > 0 else 0
        support_benchmarks.append({
            'min_support': min_sup,
            'frequent_itemsets_count': itemset_count,
            'max_itemset_size': max_len,
            'execution_time_sec': round(elapsed, 4)
        })
        print(f"Min Support: {min_sup:.2f} -> {itemset_count} frequent itemsets (max size: {max_len}) in {elapsed:.4f}s")

    df_sup_benchmarks = pd.DataFrame(support_benchmarks)
    df_sup_benchmarks.to_csv(os.path.join(output_dir, 'fpgrowth_support_experiments.csv'), index=False)

    # 4. Mine Frequent Itemsets at Primary Support Threshold (min_support = 0.03 = 3%)
    primary_min_sup = 0.03
    frequent_itemsets = fpgrowth(df_encoded, min_support=primary_min_sup, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets_sorted = frequent_itemsets.sort_values(by=['length', 'support'], ascending=[True, False])

    # Convert frozensets to readable string for CSV
    fi_export = frequent_itemsets_sorted.copy()
    fi_export['itemsets'] = fi_export['itemsets'].apply(lambda s: ", ".join(list(s)))
    fi_export.to_csv(os.path.join(output_dir, 'frequent_itemsets.csv'), index=False)
    print(f"\nSaved {len(fi_export)} frequent itemsets to frequent_itemsets.csv")

    # 5. Generate Association Rules with Multiple Confidence Thresholds
    conf_thresholds = [0.50, 0.60, 0.70]
    rules_dict = {}

    for min_conf in conf_thresholds:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
        rules_dict[min_conf] = rules
        print(f"Min Confidence: {min_conf:.2f} -> {len(rules)} association rules generated.")

    # Primary rules at min_confidence = 0.50
    primary_rules = rules_dict[0.50].copy()
    
    # Calculate additional metrics where possible (e.g., conviction, leverage)
    # mlxtend association_rules already calculates support, confidence, lift, leverage, conviction
    primary_rules['antecedents_str'] = primary_rules['antecedents'].apply(lambda s: ", ".join(list(s)))
    primary_rules['consequents_str'] = primary_rules['consequents'].apply(lambda s: ", ".join(list(s)))
    primary_rules['rule_str'] = primary_rules['antecedents_str'] + " => " + primary_rules['consequents_str']

    # Filter out trivial/redundant rules: keep rules with lift > 1.05 and confidence >= 0.50
    filtered_rules = primary_rules[primary_rules['lift'] >= 1.05].copy()
    print(f"Filtered {len(filtered_rules)} non-trivial rules (Lift >= 1.05).")

    # Sort and save various rankings
    top_by_lift = primary_rules.sort_values(by='lift', ascending=False)
    top_by_conf = primary_rules.sort_values(by='confidence', ascending=False)
    top_by_supp = primary_rules.sort_values(by='support', ascending=False)

    def format_rules_df(df_r):
        df_out = df_r[['rule_str', 'antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'leverage', 'conviction']].copy()
        for col in ['support', 'confidence', 'lift', 'leverage', 'conviction']:
            df_out[col] = df_out[col].round(4)
        return df_out

    format_rules_df(primary_rules).to_csv(os.path.join(output_dir, 'all_association_rules.csv'), index=False)
    format_rules_df(top_by_lift.head(30)).to_csv(os.path.join(output_dir, 'top_rules_by_lift.csv'), index=False)
    format_rules_df(top_by_conf.head(30)).to_csv(os.path.join(output_dir, 'top_rules_by_confidence.csv'), index=False)
    format_rules_df(top_by_supp.head(30)).to_csv(os.path.join(output_dir, 'top_rules_by_support.csv'), index=False)
    print("Saved all association rule tables to CSV.")

    # 6. Visualizations for Association Rules
    plt.figure(figsize=(9, 6), dpi=300)
    sc = plt.scatter(primary_rules['support'], primary_rules['confidence'],
                     c=primary_rules['lift'], cmap='viridis', alpha=0.7, s=40, edgecolors='none')
    cb = plt.colorbar(sc)
    cb.set_label('Lift', fontsize=11)
    plt.title('Association Rules: Support vs. Confidence (Colored by Lift)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Support', fontsize=11)
    plt.ylabel('Confidence', fontsize=11)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'arm_support_vs_confidence.png'))
    plt.close()

    # Top 10 rules bar chart by lift
    top10_lift = top_by_lift.head(10).copy()
    plt.figure(figsize=(11, 6), dpi=300)
    sns.barplot(x='lift', y='rule_str', data=top10_lift, palette='mako')
    plt.title('Top 10 Association Rules by Lift (min_sup=0.03, min_conf=0.50)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Lift Metric', fontsize=11)
    plt.ylabel('Rule (Antecedent => Consequent)', fontsize=10)
    for i, v in enumerate(top10_lift['lift']):
        plt.text(v + 0.02, i, f"Conf: {top10_lift['confidence'].iloc[i]:.2f} | Lift: {v:.2f}", va='center', fontsize=9, fontweight='semibold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'arm_top_rules_lift_barchart.png'))
    plt.close()
    print("Generated and saved ARM visualization plots.")

    return frequent_itemsets, primary_rules

if __name__ == '__main__':
    run_fpgrowth_analysis()
