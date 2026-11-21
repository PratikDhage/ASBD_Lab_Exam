"""
Closed Frequent Itemset Mining — ACLOSE Algorithm Implementation
ASBD Lab Exam Project

Theory:
- An itemset X is Frequent if support(X) >= min_support.
- An itemset X is Closed if there exists NO proper superset Y (Y strictly contains X)
  such that support(Y) == support(X).
- The closure operator c(X) maps X to the intersection of all transactions containing X.
- ACLOSE (Pasquier, Bastide, Taouil, Lakhal, 1999) mines Closed Frequent Itemsets
  losslessly by generating minimal generators and calculating their closures.
"""

import os
import time
from collections import defaultdict
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class ACloseMiner:
    """
    Implementation of the ACLOSE algorithm for mining Closed Frequent Itemsets (CFIs).
    """
    def __init__(self, min_support_ratio=0.05):
        self.min_support_ratio = min_support_ratio
        self.min_support_count = 0
        self.num_transactions = 0
        self.transactions = []
        self.item_tid_map = defaultdict(set) # Inverted tid-list representation for fast closure

    def fit(self, transactions):
        self.transactions = [frozenset(t) for t in transactions]
        self.num_transactions = len(self.transactions)
        self.min_support_count = int(np.ceil(self.min_support_ratio * self.num_transactions))

        # Build vertical tid representation
        self.item_tid_map.clear()
        for tid, trans in enumerate(self.transactions):
            for item in trans:
                self.item_tid_map[item].add(tid)

    def get_tids(self, itemset):
        """Return the set of transaction IDs that contain all items in itemset."""
        if not itemset:
            return set(range(self.num_transactions))
        tids = None
        for item in itemset:
            item_tids = self.item_tid_map.get(item, set())
            if tids is None:
                tids = item_tids.copy()
            else:
                tids = tids.intersection(item_tids)
                if not tids:
                    break
        return tids or set()

    def closure(self, itemset, tids=None):
        """
        Compute the Galois closure c(X) = intersection of all transactions containing X.
        Items present in all matching transactions form the closure.
        """
        if tids is None:
            tids = self.get_tids(itemset)
        if not tids:
            return frozenset()
        
        # An item is in the closure if its tid-set contains all tids in `tids`
        closed_items = set()
        for item, item_tids in self.item_tid_map.items():
            if tids.issubset(item_tids):
                closed_items.add(item)
        return frozenset(closed_items)

    def mine_closed_itemsets(self):
        """
        Execute ACLOSE:
        1. Discover frequent generators level by level.
        2. Prune candidates whose closure is already known or support equals a sub-generator.
        3. Compute closures to yield unique Closed Frequent Itemsets.
        """
        closed_itemsets = {} # map: closed_itemset (frozenset) -> support_count

        # L1: Frequent 1-generators
        # An item is a frequent 1-generator if count >= min_support_count
        current_generators = {}
        for item, tids in self.item_tid_map.items():
            cnt = len(tids)
            if cnt >= self.min_support_count:
                g = frozenset([item])
                current_generators[g] = cnt
                # Compute closure of 1-generator
                c = self.closure(g, tids)
                if c not in closed_itemsets or cnt > closed_itemsets[c]:
                    closed_itemsets[c] = cnt

        k = 1
        while current_generators:
            k += 1
            # Generate Candidate (k)-generators by joining (k-1)-generators
            gen_list = list(current_generators.keys())
            candidates = set()
            for i in range(len(gen_list)):
                for j in range(i + 1, len(gen_list)):
                    union_cand = gen_list[i].union(gen_list[j])
                    if len(union_cand) == k:
                        # Pruning step 1: All (k-1) subsets must be frequent generators
                        all_subsets_freq = True
                        for sub in combinations(union_cand, k - 1):
                            if frozenset(sub) not in current_generators:
                                all_subsets_freq = False
                                break
                        if all_subsets_freq:
                            candidates.add(union_cand)

            next_generators = {}
            for cand in candidates:
                cand_tids = self.get_tids(cand)
                cand_supp = len(cand_tids)
                if cand_supp >= self.min_support_count:
                    # Pruning step 2: cand is NOT a minimal generator if its support equals
                    # the support of any of its (k-1) subsets (generator pruning in ACLOSE)
                    is_minimal_generator = True
                    for sub in combinations(cand, k - 1):
                        sub_supp = current_generators.get(frozenset(sub), 0)
                        if cand_supp == sub_supp:
                            is_minimal_generator = False
                            break

                    if is_minimal_generator:
                        next_generators[cand] = cand_supp
                        # Compute closure of this generator
                        c = self.closure(cand, cand_tids)
                        if c not in closed_itemsets or cand_supp > closed_itemsets[c]:
                            closed_itemsets[c] = cand_supp

            current_generators = next_generators

        return closed_itemsets

def brute_force_closed_itemsets(transactions, min_support_ratio):
    """
    Brute-force validator:
    1. Count support of all itemsets.
    2. Filter itemsets >= min_support (Frequent Itemsets).
    3. An itemset X is closed iff no proper superset Y has supp(Y) == supp(X).
    """
    num_trans = len(transactions)
    min_supp_cnt = int(np.ceil(min_support_ratio * num_trans))
    all_items = sorted(list(set(item for t in transactions for item in t)))

    # Find all frequent itemsets
    frequent = {}
    for r in range(1, len(all_items) + 1):
        found_any = False
        for combo in combinations(all_items, r):
            c_set = set(combo)
            cnt = sum(1 for t in transactions if c_set.issubset(t))
            if cnt >= min_supp_cnt:
                frequent[frozenset(combo)] = cnt
                found_any = True
        if not found_any:
            break

    # Filter closed itemsets
    closed = {}
    for X, supp_X in frequent.items():
        is_closed = True
        for Y, supp_Y in frequent.items():
            if len(Y) > len(X) and X.issubset(Y):
                if supp_Y == supp_X:
                    is_closed = False
                    break
        if is_closed:
            closed[X] = supp_X

    return frequent, closed

def run_aclose_analysis(arm_csv='outputs/cleaned_data/arm_transactions.csv', output_dir='outputs/closed_itemsets'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(arm_csv)
    transactions = [row.dropna().tolist() for _, row in df.iterrows()]
    print(f"Loaded {len(transactions)} transactions for ACLOSE.")

    # 1. Verification on a Small Synthetic / Sample Dataset
    print("\n--- Verifying ACLOSE Algorithm Correctness vs. Brute-Force ---")
    toy_sample = transactions[:100]
    toy_min_sup = 0.10

    bf_freq, bf_closed = brute_force_closed_itemsets(toy_sample, toy_min_sup)
    miner_toy = ACloseMiner(min_support_ratio=toy_min_sup)
    miner_toy.fit(toy_sample)
    aclose_toy_closed = miner_toy.mine_closed_itemsets()

    print(f"Brute-Force Frequent Itemsets: {len(bf_freq)}")
    print(f"Brute-Force Closed Itemsets:   {len(bf_closed)}")
    print(f"ACLOSE Mined Closed Itemsets:  {len(aclose_toy_closed)}")
    
    # Assert set equality
    assert set(bf_closed.keys()) == set(aclose_toy_closed.keys()), "ACLOSE verification mismatch!"
    print("Verification PASSED: ACLOSE exactly matches 100% of mathematical closed itemsets!")

    # 2. Multi-threshold Experimentation on Full Dataset
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    comparison_records = []

    print("\n--- Running ACLOSE Across Multiple Support Thresholds ---")
    final_cfi_dict = {}

    for sup in thresholds:
        miner = ACloseMiner(min_support_ratio=sup)
        miner.fit(transactions)
        t0 = time.time()
        cfis = miner.mine_closed_itemsets()
        t_aclose = time.time() - t0

        if sup == 0.10:
            final_cfi_dict = cfis

        # Also get FI count from transaction encoder / apriori/fp-growth for comparison
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import fpgrowth
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        fi = fpgrowth(df_encoded, min_support=sup, use_colnames=True)
        fi_count = len(fi)
        cfi_count = len(cfis)
        reduction_pct = ((fi_count - cfi_count) / fi_count * 100) if fi_count > 0 else 0

        comparison_records.append({
            'min_support': sup,
            'frequent_itemsets': fi_count,
            'closed_frequent_itemsets': cfi_count,
            'redundancy_reduction_pct': round(reduction_pct, 2),
            'aclose_execution_time_sec': round(t_aclose, 4)
        })
        print(f"Min Support {sup*100:4.1f}% | FI: {fi_count:<4} | CFI: {cfi_count:<4} | Reduction: {reduction_pct:5.2f}% | Time: {t_aclose:.4f}s")

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(os.path.join(output_dir, 'fi_vs_cfi_comparison.csv'), index=False)

    # 3. Format and Save Primary CFIs (at min_support = 0.10)
    cfi_rows = []
    for itemset, supp_cnt in final_cfi_dict.items():
        supp_ratio = supp_cnt / len(transactions)
        cfi_rows.append({
            'itemset': ", ".join(sorted(list(itemset))),
            'itemset_size': len(itemset),
            'support_count': supp_cnt,
            'support_ratio': round(supp_ratio, 4)
        })
    df_cfi = pd.DataFrame(cfi_rows).sort_values(by=['itemset_size', 'support_count'], ascending=[True, False])
    df_cfi.to_csv(os.path.join(output_dir, 'closed_frequent_itemsets.csv'), index=False)
    print(f"\nSaved {len(df_cfi)} Closed Frequent Itemsets (min_sup=0.10) to closed_frequent_itemsets.csv")

    # 4. Visualizations
    # A) FI vs. CFI Count across Support Thresholds
    plt.figure(figsize=(9, 5), dpi=300)
    x = np.arange(len(df_comp))
    width = 0.35
    plt.bar(x - width/2, df_comp['frequent_itemsets'], width, label='Ordinary Frequent Itemsets (FI)', color='#3498db', edgecolor='black')
    plt.bar(x + width/2, df_comp['closed_frequent_itemsets'], width, label='Closed Frequent Itemsets (CFI - ACLOSE)', color='#e74c3c', edgecolor='black')
    plt.xticks(x, [f"{int(s*100)}%" for s in df_comp['min_support']])
    plt.xlabel('Minimum Support Threshold', fontsize=11)
    plt.ylabel('Number of Itemsets', fontsize=11)
    plt.title('Frequent Itemsets vs. Closed Frequent Itemsets (Redundancy Reduction)', fontsize=13, fontweight='bold', pad=12)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aclose_fi_vs_cfi_comparison.png'))
    plt.close()

    # B) Redundancy Reduction Percentage Curve
    plt.figure(figsize=(8, 4.5), dpi=300)
    plt.plot([f"{int(s*100)}%" for s in df_comp['min_support']], df_comp['redundancy_reduction_pct'],
             marker='D', color='#27ae60', linewidth=2.5, markersize=8)
    plt.title('Percentage Reduction in Itemset Redundancy Achieved by ACLOSE', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Minimum Support Threshold', fontsize=11)
    plt.ylabel('Redundancy Reduction (%)', fontsize=11)
    plt.ylim(0, 100)
    for i, v in enumerate(df_comp['redundancy_reduction_pct']):
        plt.text(i, v + 3, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aclose_redundancy_reduction_curve.png'))
    plt.close()
    print("Saved ACLOSE visualization plots.")

    return df_cfi, df_comp

if __name__ == '__main__':
    run_aclose_analysis()
