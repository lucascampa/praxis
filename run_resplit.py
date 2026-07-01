from resplit import RESPLIT
import pandas as pd
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from split._tree import Leaf

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

train_data = pd.read_csv('airline-passenger-satisfaction/praxis_dataset_balanced.csv')
x_train = train_data[SELECTED_FEATURES]
y_train = train_data['satisfaction']

test_data = pd.read_csv('airline-passenger-satisfaction/test_clean_encoded.csv')
x_test = test_data[SELECTED_FEATURES]
y_test = test_data['satisfaction']

config = {
    "regularization": 0.005,
    "rashomon_bound_multiplier": 0.03,
    "depth_budget": 6,
    "cart_lookahead_depth": 2,
    "verbose": True
}

print("Training RESPLIT...")
start_time = time.perf_counter()
model = RESPLIT(config, fill_tree="treefarms")
model.fit(x_train, y_train)

total_runtime = time.perf_counter() - start_time

print(f"\nRashomon set size: {len(model)} trees")
print(f"Total runtime: {total_runtime:.3f} seconds\n")

results = []

avg_accuracy = 0
avg_precision = 0
avg_recall = 0
avg_f1 = 0

for i, tree in enumerate(model):
    pred = tree.predict(x_test)
    acc = float(accuracy_score(y_test, pred))
    prec = float(precision_score(y_test, pred))
    rec = float(recall_score(y_test, pred))
    f1 = float(f1_score(y_test, pred))

    avg_accuracy += acc
    avg_precision += prec
    avg_recall += rec
    avg_f1 += f1

    results.append({
        'Tree': i,
        'Model': f'RESPLIT (Tree {i})',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Loss': round(loss, 6),
        'Size': f"Tree {i} of {len(model)}",
        'Runtime (s)': round(total_runtime / len(model), 3)
    })

# Add average row and calculate average loss
n = len(model)
avg_loss = sum(r['Loss'] for r in results) / len(results) + config['regularization'] * n
results.append({
    'Tree': 'Avg',
    'Model': f'RESPLIT (avg of {n} trees)',
    'Accuracy': avg_accuracy / n,
    'Precision': avg_precision / n,
    'Recall': avg_recall / n,
    'F1': avg_f1 / n,
    'Loss': round(avg_loss, 6),
    'Size': f"{n} trees",
    'Runtime (s)': round(total_runtime, 3)
})

results_df = pd.DataFrame(results)
print("="*100)
print(results_df.to_string(index=False))
print("="*100)



# Save to JSON for comparison notebook
output_file = 'results2.json'

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_file}")
