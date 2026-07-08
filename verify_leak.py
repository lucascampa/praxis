"""
Pin down which config key truncates the stump-level TREEFARMS search.

Background (see verify_bound.py and logs/resplit_depth6_noleak.txt):
  - healthy stump (features[7]==1 & [25]==0), bound 0.15697:
      in-pipeline -> 5 trees, standalone -> 24 trees
  - stump config dumps show cart_lookahead_depth: 3 leaking from the top-level
    call; setting look_ahead=false changed nothing (bit-identical run), so the
    boolean is suspected to be a no-op and the integer the real gate.

Four calls on the same stump, same bound, varying only the lookahead keys.
All calls share one process, so this also demonstrates the overlay/leak
behavior directly (call 4 passes no lookahead keys after call 1 set them).

Run from repo root (WSL, praxis-env):
  python verify_leak.py 2>&1 | tee logs/verify_leak.txt
"""
from resplit import RESPLIT, TREEFARMS
import pandas as pd
import numpy as np
from split._tree import Leaf

REG = 0.005
REMAINING_DEPTH = 4


def get_num_leaves(tree):
    if isinstance(tree, Leaf) or tree is None:
        return 1
    return get_num_leaves(tree.left_child) + get_num_leaves(tree.right_child)


train_data = pd.read_csv('data/airline-passenger-satisfaction/train_binarized.csv')
x_train = train_data.drop(columns=['satisfaction'])
y_train = train_data['satisfaction']
n_full = len(y_train)

mask = (x_train.iloc[:, 7] == 1) & (x_train.iloc[:, 25] == 0)
X_part = x_train[mask].reset_index(drop=True)
y_part = y_train[mask].reset_index(drop=True)
assert len(X_part) == 25100, f"expected 25100 rows, got {len(X_part)}"

reg_local = min(0.1, REG * n_full / len(y_part))

# Reproduce the stump's bound exactly as fill_leaves_with_treefarms does
resplit_model = RESPLIT({"regularization": REG, "depth_budget": 6,
                         "cart_lookahead_depth": 3, "verbose": False})
resplit_model.n = n_full
greedy_tree, _ = resplit_model.train_greedy(X_part, y_part, REMAINING_DEPTH, reg_local)
preds = np.array([resplit_model._predict_sample(X_part.values[i, :], greedy_tree)
                  for i in range(len(X_part))])
bound = (y_part != preds).mean() + reg_local * get_num_leaves(greedy_tree)
print(f"stump: {len(X_part)} rows, reg_local {reg_local:.6f}, bound {bound:.6f} "
      f"[log: 0.15697]\n")

CASES = [
    ("pipeline repro: lookahead_depth=3, look_ahead=True ",
     {'cart_lookahead_depth': 3, 'look_ahead': True}),
    ("gate test:      lookahead_depth=4 (=depth_budget)  ",
     {'cart_lookahead_depth': 4, 'look_ahead': True}),
    ("no-op test:     lookahead_depth=3, look_ahead=False",
     {'cart_lookahead_depth': 3, 'look_ahead': False}),
    ("leak test:      no lookahead keys passed at all    ",
     {}),
]

for label, extra in CASES:
    config = {'depth_budget': REMAINING_DEPTH,
              'regularization': reg_local,
              'rashomon_bound': bound}
    config.update(extra)
    tf = TREEFARMS(config)
    tf.fit(X_part, y_part)
    print(f"{label} -> {tf.get_tree_count()} trees")

print("\nReading: 5 = truncated search, 24 = exact search.")
print("If case 2 gives 24, cart_lookahead_depth is the gate and depth_budget")
print("is the safe 'off' value. If case 4 inherits case 3's keys instead of")
print("using defaults, the overlay/leak is demonstrated in-process.")
