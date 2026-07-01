from resplit import RESPLIT
import pandas as pd
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from split._tree import Leaf
import tracemalloc

def get_num_leaves(tree):
    if isinstance(tree, Leaf) or tree is None:
        return 1
    if isinstance(tree, tuple):
        return 1
    return get_num_leaves(tree.left_child) + get_num_leaves(tree.right_child)

SELECTED_FEATURES = [
    'Online_boarding', 'Type_of_Travel_Personal Travel', 'Class_Eco',
    'Inflight_wifi_service', 'On_board_service', 'Customer_Type_disloyal Customer',
    'Inflight_entertainment', 'Checkin_service', 'Leg_room_service'
]

train_data = pd.read_csv('airline-passenger-satisfaction/train_clean_encoded_balanced.csv')
x_train = train_data[SELECTED_FEATURES]
y_train = train_data['satisfaction']

test_data = pd.read_csv('airline-passenger-satisfaction/test_clean_encoded.csv')
x_test = test_data[SELECTED_FEATURES]
y_test = test_data['satisfaction']

config = {
    "regularization": 0.005,
    "rashomon_bound_multiplier": 0.01,
    "depth_budget": 5,
    "cart_lookahead_depth": 3,
    "verbose": True
}

print("Training RESPLIT...")
tracemalloc.start()
start_time = time.perf_counter()
model = RESPLIT(config, fill_tree="treefarms")
model.fit(x_train, y_train)
total_runtime = time.perf_counter() - start_time
_, peak_memory = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_memory_mb = peak_memory / 1024 / 1024

print(f"\nPrefix trees found by TREEFARMS: {model.rashomon_set_prefix.get_tree_count()}")
print(f"Rashomon set size: {len(model)} trees")
print(f"Total runtime: {total_runtime:.3f} seconds")
print(f"Peak memory: {peak_memory_mb:.2f} MB\n")

lambda_reg = config['regularization']

# Find tree with minimum training loss
min_loss = float('inf')
best_tree_idx = -1
best_num_leaves = 0


for i in range(len(model)):
    tree = model[i]
    train_pred = model.predict(x_train, i)
    num_leaves = get_num_leaves(tree)
    train_error_rate = (train_pred != y_train.values).sum() / len(y_train)
    train_loss = train_error_rate + lambda_reg * num_leaves

    if train_loss < min_loss:
        min_loss = train_loss
        best_tree_idx = i
        best_num_leaves = num_leaves

# Calculate test metrics for best tree
test_pred = model.predict(x_test, best_tree_idx)
acc = float(accuracy_score(y_test, test_pred))
prec = float(precision_score(y_test, test_pred))
rec = float(recall_score(y_test, test_pred))
f1 = float(f1_score(y_test, test_pred))

results = [{
    'Model': 'RESPLIT (Best Tree)',
    'Tree_Index': best_tree_idx,
    'Train_Loss': round(min_loss, 6),
    'Test_Accuracy': acc,
    'Test_Precision': prec,
    'Test_Recall': rec,
    'Test_F1': f1,
    'Num_Leaves': best_num_leaves,
    'Rashomon_Set_Size': len(model),
    'Runtime (s)': round(total_runtime, 3),
    'Peak_Memory (MB)': round(peak_memory_mb, 2)
}]

# Print results
results_df = pd.DataFrame(results)
print("="*100)
print(results_df.to_string(index=False))
print("="*100)

# Save to JSON for comparison notebook
output_file = 'results/results2.json'

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_file}")
