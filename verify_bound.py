"""
Lightweight verification of why RESPLIT's leaf-level enumeration misbehaves,
using socket identities from the depth-6 run's resplit_log.txt — no refit.

Facts from the log needing explanation:
  - healthy socket (features[7]==1 & [25]==0, 25,100 rows): bound 0.15697 -> 5 trees
  - dead socket   (features[27]==0 & [1]==0, 62,153 rows): bound 0.159347 -> 0 trees
    A bound equal to a greedy (feasible) tree's loss can never exclude every
    tree, since the optimum <= greedy. Zero trees means the bound RESPLIT
    computed sits below the partition's true optimum — something is
    inconsistent between the Python-side bound and the C++ solver's objective.

Per socket, this script:
  1. Reproduces the greedy fill and the bound exactly as
     fill_leaves_with_treefarms() computes it; checks it against the log.
     Also reports train_greedy's internal loss (mixed normalization: errors
     divided by full n, regularization scaled to the partition) next to the
     correctly-normalized local loss of the same tree.
  2. Reruns the leaf-level TREEFARMS with that bound; checks tree count
     against the log.
  3. Computes the partition's TRUE optimum with GOSDT (same config RESPLIT's
     fill_tree='optimal' path uses) and compares it to the bound:
       optimum >  bound  -> the greedy reference is scored inconsistently
                            (its claimed loss beats the true optimum);
       optimum <= bound  -> the C++ solver rejects trees the bound admits
                            (units/precision mismatch on the C++ side).
  4. Reruns TREEFARMS with a corrected bound derived from the true optimum
     to show the socket revives.

Run from repo root (WSL, praxis-env):  python verify_bound.py
"""
from resplit import RESPLIT, TREEFARMS
from split import GOSDTClassifier
import pandas as pd
import numpy as np
from split._tree import Leaf

REG = 0.005
REMAINING_DEPTH = 4  # depth_budget=6, cart_lookahead_depth=3 -> 6-3+1


def get_num_leaves(tree):
    if isinstance(tree, Leaf) or tree is None:
        return 1
    return get_num_leaves(tree.left_child) + get_num_leaves(tree.right_child)


def predict_tree(tree, X):
    """Vectorized prediction; RESPLIT convention: feature==1 -> left child."""
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


def local_objective(tree, X, y, reg_local):
    errors = int((predict_tree(tree, X) != y.values).sum())
    return errors / len(y) + reg_local * get_num_leaves(tree), errors


train_data = pd.read_csv('data/airline-passenger-satisfaction/train_binarized.csv')
x_train = train_data.drop(columns=['satisfaction'])
y_train = train_data['satisfaction']
n_full = len(y_train)

# Bare RESPLIT instance — no fit; we borrow train_greedy/extract_tree/_predict_sample
resplit_model = RESPLIT({"regularization": REG, "depth_budget": 6,
                         "cart_lookahead_depth": 3, "verbose": False})
resplit_model.n = n_full  # normally set inside fit()

SOCKETS = [
    # (label, [(feature_idx, value), ...], expected_rows, bound_in_log, trees_in_log)
    ("healthy (7==1, 25==0)", [(7, 1), (25, 0)], 25100, 0.15697, 5),
    ("dead (27==0, 1==0)",    [(27, 0), (1, 0)], 62153, 0.159347, 0),
]

for label, conditions, expected_rows, log_bound, log_trees in SOCKETS:
    mask = pd.Series(True, index=x_train.index)
    for feat, val in conditions:
        mask &= x_train.iloc[:, feat] == val
    X_part = x_train[mask].reset_index(drop=True)
    y_part = y_train[mask].reset_index(drop=True)

    print("=" * 78)
    print(f"SOCKET {label}: {len(X_part)} rows (log: {expected_rows})")
    if len(X_part) != expected_rows:
        print("  !! partition size mismatch — path decoding wrong, skipping")
        continue

    reg_local = min(0.1, REG * n_full / len(y_part))
    print(f"  local regularization: {reg_local:.6f}")

    # --- 1. Reproduce the greedy fill and the bound as the source computes it
    greedy_tree, greedy_internal_loss = resplit_model.train_greedy(
        X_part, y_part, REMAINING_DEPTH, reg_local)
    greedy_local_loss, greedy_errors = local_objective(
        greedy_tree, X_part, y_part, reg_local)
    greedy_leaves = get_num_leaves(greedy_tree)

    # fill_leaves_with_treefarms: bound = misclassification MEAN on the
    # partition + reg_local * leaves
    bound = greedy_errors / len(y_part) + reg_local * greedy_leaves

    print(f"  greedy fill: {greedy_leaves} leaves, {greedy_errors} errors")
    print(f"  train_greedy internal loss (mixed units): {greedy_internal_loss:.6f}")
    print(f"  same tree, correct local units:           {greedy_local_loss:.6f}")
    print(f"  reproduced bound: {bound:.6f}   [log: {log_bound}]")

    # --- 2. Leaf TREEFARMS with the reproduced bound (as the fill does)
    tf = TREEFARMS({'depth_budget': REMAINING_DEPTH,
                    'regularization': reg_local,
                    'rashomon_bound': bound})
    tf.fit(X_part, y_part)
    print(f"  TREEFARMS @ reproduced bound: {tf.get_tree_count()} trees   "
          f"[log: {log_trees}]")

    # --- 3. True optimum of this subproblem via GOSDT (fill_tree='optimal' config)
    leaf_clf = GOSDTClassifier(regularization=reg_local,
                               depth_budget=REMAINING_DEPTH,
                               time_limit=60, similar_support=False,
                               verbose=False, allow_small_reg=False)
    leaf_clf.fit(X_part, y_part)
    optimal_tree = resplit_model.extract_tree(leaf_clf)
    optimum, opt_errors = local_objective(optimal_tree, X_part, y_part, reg_local)
    print(f"  GOSDT true optimum: {optimum:.6f} "
          f"({get_num_leaves(optimal_tree)} leaves, {opt_errors} errors)")
    if optimum > bound + 1e-9:
        print("  -> optimum ABOVE the bound: the greedy reference is scored "
              "inconsistently with the solver's objective")
    else:
        print("  -> optimum WITHIN the bound: the C++ solver is rejecting "
              "trees the bound admits (units/precision mismatch)")

    # --- 4. Corrected bound anchored to the true optimum
    corrected = optimum * 1.01 + 1e-6
    tf2 = TREEFARMS({'depth_budget': REMAINING_DEPTH,
                     'regularization': reg_local,
                     'rashomon_bound': corrected})
    tf2.fit(X_part, y_part)
    print(f"  TREEFARMS @ corrected bound {corrected:.6f}: "
          f"{tf2.get_tree_count()} trees")

print("=" * 78)
