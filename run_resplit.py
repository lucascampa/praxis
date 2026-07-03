from resplit import RESPLIT
import pandas as pd
import json
import time
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from split._tree import Leaf, Node
import tracemalloc

# --- Patch: skip the eager materialization of the Rashomon set ---
# fit() normally ends with hash_for_indexing(), which reconstructs every one of
# the num_models trees (the full cross-product of subtree choices) into a dict.
# That step is what OOM-kills the process on large sets. The compact prefix-trie
# representation built during the fill loop already contains everything we need:
# the set size sits in num_models, and the exact best tree is recovered below
# by per-socket minimization (socket losses are independent, so the global
# minimum is the sum of per-socket minima — no enumeration required).
RESPLIT.hash_for_indexing = lambda self, prefixes: prefixes

MODELS_CACHE = 'results/resplit_models.pkl'


def get_num_leaves(tree):
    if isinstance(tree, Leaf) or tree is None:
        return 1
    return get_num_leaves(tree.left_child) + get_num_leaves(tree.right_child)


def predict_tree(tree, X):
    """Vectorized prediction for a Node/Leaf tree over binary features.
    RESPLIT convention: feature == 1 goes to the LEFT child."""
    preds = np.zeros(len(X), dtype=int)

    def rec(node, mask):
        if not mask.any():
            return
        if isinstance(node, Leaf):
            preds[mask] = node.prediction
            return
        col = X.iloc[:, int(node.feature)].values.astype(bool)
        rec(node.left_child, mask & col)
        rec(node.right_child, mask & ~col)

    if len(X):
        rec(tree, np.ones(len(X), dtype=bool))
    return preds


def best_tree_in_trie(trie, X, y, lam, n):
    """Exact best tree within one compact prefix trie.

    The prefix routes disjoint sample partitions to each socket (leaf of the
    prefix), so each socket's loss contribution is independent of the choices
    made at other sockets. The best full tree is therefore the best candidate
    at each socket, chosen independently, summed.

    Returns (loss_contribution, tree), or (None, None) if any socket has an
    empty candidate menu (the prefix contributes no valid trees).
    """
    if isinstance(trie, tuple):  # (subtrie, count) wrapper from enumerate step
        trie = trie[0]

    if isinstance(trie, list):  # socket: menu of candidate subtrees
        best_score, best_cand = None, None
        for cand in trie:
            errors = (predict_tree(cand, X) != y.values).sum() if len(X) else 0
            score = errors / n + lam * get_num_leaves(cand)
            if best_score is None or score < best_score:
                best_score, best_cand = score, cand
        return best_score, best_cand

    if isinstance(trie, Leaf):
        errors = (y.values != trie.prediction).sum() if len(X) else 0
        return errors / n + lam, trie

    # interior prefix node
    col = X.iloc[:, int(trie.feature)].values.astype(bool)
    l_score, l_tree = best_tree_in_trie(trie.left_child, X[col], y[col], lam, n)
    if l_score is None:
        return None, None
    r_score, r_tree = best_tree_in_trie(trie.right_child, X[~col], y[~col], lam, n)
    if r_score is None:
        return None, None
    return l_score + r_score, Node(feature=int(trie.feature),
                                   left_child=l_tree, right_child=r_tree)


def print_paths(node, names, path=None):
    if path is None:
        path = []
    if isinstance(node, Leaf):
        conds = ' AND '.join(path) if path else '(always)'
        print(f"IF {conds} -> {'satisfied' if node.prediction else 'not satisfied'}")
        return
    f = names[int(node.feature)]
    print_paths(node.left_child, names, path + [f"[{f}]=YES"])
    print_paths(node.right_child, names, path + [f"[{f}]=NO"])


# Binarized data exported from notebooks/comparison.ipynb (same ThresholdGuessBinarizer
# output PRAXIS uses — RESPLIT requires binary features, see SPLIT paper Appendix A.8)
train_data = pd.read_csv('airline-passenger-satisfaction/train_binarized.csv')
x_train = train_data.drop(columns=['satisfaction'])
y_train = train_data['satisfaction']

test_data = pd.read_csv('airline-passenger-satisfaction/test_binarized.csv')
x_test = test_data.drop(columns=['satisfaction'])
y_test = test_data['satisfaction']

config = {
    "regularization": 0.005,
    "rashomon_bound_multiplier": 0.01,
    "depth_budget": 5,
    "cart_lookahead_depth": 3,
    "verbose": True
}

if os.path.exists(MODELS_CACHE):
    # Reuse the fitted model from a previous run — extraction tweaks don't
    # need to pay for another hour-long fit
    print(f"Loading cached RESPLIT model from {MODELS_CACHE}...")
    with open(MODELS_CACHE, 'rb') as f:
        cache = pickle.load(f)
    models = cache['models']
    num_models = cache['num_models']
    num_models_per_prefix = cache['num_models_per_prefix']
    prefix_count = cache['prefix_count']
    total_runtime = cache['runtime']
    peak_memory_mb = cache['peak_memory_mb']
else:
    print("Training RESPLIT...")
    tracemalloc.start()
    start_time = time.perf_counter()
    model = RESPLIT(config, fill_tree="treefarms")
    model.fit(x_train, y_train)
    total_runtime = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / 1024 / 1024

    models = model.models
    num_models = int(model.num_models)
    num_models_per_prefix = model.num_models_per_prefix
    prefix_count = model.rashomon_set_prefix.get_tree_count()

    with open(MODELS_CACHE, 'wb') as f:
        pickle.dump({
            'models': models,
            'num_models': num_models,
            'num_models_per_prefix': num_models_per_prefix,
            'prefix_count': prefix_count,
            'runtime': total_runtime,
            'peak_memory_mb': peak_memory_mb,
        }, f)
    print(f"✓ Fitted model cached to {MODELS_CACHE}")

dead_prefixes = sum(1 for c in num_models_per_prefix if c == 0)
print(f"\nPrefix trees found by TREEFARMS: {prefix_count}")
print(f"Rashomon set size: {num_models} trees")
print(f"Prefixes with no valid trees: {dead_prefixes} / {len(num_models_per_prefix)}")
print(f"Total runtime: {total_runtime:.3f} seconds")
print(f"Peak memory: {peak_memory_mb:.2f} MB\n")

lambda_reg = config['regularization']
n = len(y_train)

# Exact best tree: per-socket minima summed per prefix, best prefix wins
print("Extracting best tree from compact representation...")
extract_start = time.perf_counter()
min_loss = float('inf')
best_tree = None
best_prefix_idx = -1
for p, trie in enumerate(models):
    if num_models_per_prefix[p] == 0:
        continue
    loss, tree = best_tree_in_trie(trie, x_train, y_train, lambda_reg, n)
    if loss is None:
        continue
    if loss < min_loss:
        min_loss = loss
        best_tree = tree
        best_prefix_idx = p
        print(f"Prefix {p}: new best loss = {loss:.6f}")
    if (p + 1) % 100 == 0:
        print(f"  ...{p + 1}/{len(models)} prefixes scanned "
              f"({time.perf_counter() - extract_start:.0f}s)")
print(f"Extraction took {time.perf_counter() - extract_start:.3f} seconds\n")

best_num_leaves = get_num_leaves(best_tree)

# Sanity check: recompute loss by evaluating the assembled tree directly
train_pred = predict_tree(best_tree, x_train)
recomputed_loss = (train_pred != y_train.values).sum() / n + lambda_reg * best_num_leaves
if abs(recomputed_loss - min_loss) > 1e-6:
    print(f"WARNING: recomputed loss {recomputed_loss:.6f} != trie-derived {min_loss:.6f}")
min_loss = recomputed_loss

print(f"=== RESPLIT BEST TREE (prefix {best_prefix_idx}) ===")
print_paths(best_tree, list(x_train.columns))
print()

# Test metrics for best tree
test_pred = predict_tree(best_tree, x_test)
acc = float(accuracy_score(y_test, test_pred))
prec = float(precision_score(y_test, test_pred))
rec = float(recall_score(y_test, test_pred))
f1 = float(f1_score(y_test, test_pred))

results = [{
    'Model': 'RESPLIT (Best Tree)',
    'Train_Loss': round(min_loss, 6),
    'Test_Accuracy': acc,
    'Test_Precision': prec,
    'Test_Recall': rec,
    'Test_F1': f1,
    'Num_Leaves': best_num_leaves,
    'Rashomon_Set_Size': num_models,
    'Runtime (s)': round(total_runtime, 3),
    'Peak_Memory (MB)': round(peak_memory_mb, 2)
}]

# Print results
results_df = pd.DataFrame(results)
print("=" * 100)
print(results_df.to_string(index=False))
print("=" * 100)

# Save to JSON for comparison notebook
output_file = 'results/results2.json'

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_file}")
