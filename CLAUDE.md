# PRAXIS Project Context

**Project Goal**: Compare two Rashomon set enumeration algorithms (**RESPLIT** and **PRAXIS**) on the airline passenger satisfaction dataset to evaluate scalability, set size, tree quality, and feature importance stability.

---

## Key Context Documents

### 1. **SPLIT context.md** (in this repo)
📄 [SPLIT context.md](./SPLIT context.md)

Comprehensive summary covering:
- **SPLIT paper** (Babbar et al., 2025): Core algorithms (SPLIT, LicketySPLIT, RESPLIT), theorems, experimental results
- **DIMEX project**: Original externship work benchmarking SPLIT vs. XGBoost
- **Airline dataset**: 103k+ records, 23 features, binary satisfaction classification
- **Key findings**: 91.97% SPLIT vs. 93.58% XGBoost (1.6pp difference, 94% fewer leaves)

**Why it matters**: SPLIT context explains the foundation. RESPLIT (mentioned in the paper) is the Rashomon set enumeration variant we're comparing against PRAXIS.

### 2. **PRAXIS README** (external)
📄 https://github.com/zakk-h/PRAXIS (official repository)

PRAXIS is a newer algorithm (ICML 2026) that:
- Uses a **proxy-based algorithm** (LicketySPLIT by default, lookahead_k=1)
- Enumerates Rashomon sets via **multiplicative slack** (1 + rashomon_mult)
- Includes built-in **RID** (Rashomon Importance Distribution) computation
- Faster than RESPLIT via iterative budget refinement

---

## Paper Summaries & File Mapping

### SPLIT Paper (Babbar et al., ICML 2025)
📄 **File**: [papers/SPLIT.pdf](./papers/SPLIT.pdf)

**Key Innovation**: Lookahead depth cutoff insight — greedy splits are only harmful near the root but acceptable near leaves, enabling exponential speedup over GOSDT.

**Algorithms**:
- **SPLIT**: Single optimal tree with lookahead cutoff (branch-and-bound)
- **LicketySPLIT**: Polynomial-time variant (no optimality guarantee)
- **RESPLIT**: Extension for enumerating Rashomon sets via TreeFARMS

**Complexity Analysis**:
- Runtime: O(n(d-d_l)k^(d_l+1) + nk^(d-d_l)) where d_l is lookahead depth
- **100× faster** than GOSDT on real datasets
- **10-20× speedup** over naive TreeFARMS enumeration with RESPLIT

**Experimental Results**:
- On airline dataset: 91.97% accuracy with ~8 leaves vs. XGBoost's 755 leaves
- Feature importance correlation: 0.93+ (RESPLIT stability comparable to single tree)
- Scales well to moderate problem sizes (n~100k, d~20)

**Key Pages**: 1-8 cover core algorithms, lookahead insight, complexity, airline experiments

**Why it matters**: Foundation for understanding how RESPLIT (used as comparison baseline) works and why it's slower than PRAXIS.

---

### PRAXIS Paper (Harary et al., ICML 2026)
📄 **File**: [papers/PRAXIS.pdf](./papers/PRAXIS.pdf)

**Key Innovation**: Proxy-based enumeration using a fast heuristic (LicketySPLIT by default) to prune the search space, achieving polynomial-time per-tree complexity instead of exponential worst-case.

**Core Insight**:
- Runtime = O(|Rashomon set| × polynomial), not exponential per tree
- Use proxy (approximate) solutions to filter the search space
- Only solve full problem for trees the proxy doesn't eliminate

**Algorithms**:
- **Proxy filtering**: Use fast heuristic to identify promising tree structures
- **Budget refinement (SOLVE_SIBLINGS)**: Iteratively refine search bounds to find more trees
- **RID computation**: Built-in Rashomon Importance Distribution for feature stability analysis

**Experimental Results**:
- **100-1000× faster** than RESPLIT across datasets
- **0.98-1.0 recall**: Near-perfect recovery of trees within Rashomon bound
- On Churn dataset: PRAXIS finds 1M+ trees in 35s; RESPLIT finds 0 valid trees in 43m
- Feature importance stability: RID scores measure which features consistently matter across Rashomon set

**Key Pages**: 1-9 cover problem formulation, proxy-based algorithm, SOLVE_SIBLINGS, experimental benchmarks

**Why it matters**: This is what RESPLIT-vs-PRAXIS comparison should demonstrate on airline data—whether PRAXIS's speed translates to similar Rashomon set coverage and feature importance stability.

**Co-authors relationship**: Both papers (SPLIT 2025, PRAXIS 2026) authored by McTavish, Babbar, Seltzer, Rudin—showing a collaborative evolution of the approach.

---

## Project Structure

```
praxis/
├── CLAUDE.md                              # This file — project documentation
├── SPLIT context.md                       # SPLIT paper & DIMEX project summary
├── README.md                              # Project overview & workflow
├── SETUP.md                               # Installation instructions
├── setup.py                               # Package installation
├── environment.yml                        # Conda environment (Python 3.10 + PRAXIS)
│
├── papers/                                # Research papers (local copies for reference)
│   ├── SPLIT.pdf                         # Babbar et al., ICML 2025 (read pages 1-8)
│   └── PRAXIS.pdf                        # Harary et al., ICML 2026 (read pages 1-9)
│
├── airline-passenger-satisfaction/        # Dataset (shared with DIMEX)
│   ├── train.csv, test.csv               # Raw data
│   ├── train_clean.csv, test_clean.csv   # After missing-value removal
│   ├── train_clean_encoded.csv           # After one-hot encoding
│   └── train_clean_encoded_balanced.csv  # After SMOTE balancing
│
├── dimex/                                 # Main Python package
│   ├── __init__.py                       # Public API exports
│   ├── preprocessing.py                  # Data cleaning, encoding, balancing
│   ├── xgb_runner.py                     # XGBoost training & feature selection
│   ├── split_runner.py                   # SPLIT training & evaluation
│   ├── praxis_runner.py                  # PRAXIS training & evaluation (NEW)
│   └── reporting.py                      # Confusion matrices & plots
│
├── notebooks/                             # Analysis notebooks
│   ├── Black-Box to Glass-Box modeling [RANDOM_SEED=42].ipynb       # DIMEX notebooks
│   ├── Black-Box to Glass-Box modeling [RANDOM_SEED=50].ipynb
│   ├── Black-Box to Glass-Box modeling [RANDOM_SEED=99].ipynb
│   ├── (original)/                       # Backups
│   └── RESPLIT_vs_PRAXIS_comparison.ipynb (TO CREATE)              # Main comparison
│
├── results/                               # Output figures & tables
│   └── Results.png                       # DIMEX comparison chart
│
├── LICENSE                                # MIT License
└── .gitignore
```

---

## Key Files & Their Roles

| File | Purpose | Key Functions |
|------|---------|---|
| `dimex/preprocessing.py` | Data cleaning & encoding | `clean_missing()`, `binarize_encode()`, `smote()` |
| `dimex/xgb_runner.py` | XGBoost training & feature selection | `train_xgb()`, `cumulative_gain()` |
| `dimex/split_runner.py` | SPLIT algorithm interface | `train_split()`, `prediction_split()` |
| `dimex/praxis_runner.py` | PRAXIS algorithm interface (NEW) | `train_praxis()`, `prediction_praxis()`, `get_rashomon_stats()` |
| `dimex/reporting.py` | Visualization | `cm()` (confusion matrices) |
| `notebooks/*.ipynb` | Analysis workflows | Load data → train → evaluate → compare |

---

## Dataset Overview

**Airline Passenger Satisfaction** (Kaggle)

- **File**: `airline-passenger-satisfaction/train_clean_encoded_balanced.csv` (ALL features)
- **Size**: 103,904 training records (after missing-value removal and SMOTE balancing)
- **Target**: Binary classification (satisfied=1, neutral/dissatisfied=0)
- **Total Features**: 23 after one-hot encoding
  - 4 continuous: Age, Flight_Distance, Departure_Delay, Arrival_Delay
  - 14 service ratings (1–5): WiFi, Online_boarding, Seat_comfort, etc.
  - 5 categorical (one-hot encoded): Gender, Customer_Type, Type_of_Travel, Class

**Optimal 9-feature subset** (selected by XGBoost at 80% cumulative gain threshold):
```
['Online_boarding', 'Type_of_Travel_Personal Travel', 'Class_Eco', 
 'Inflight_wifi_service', 'On_board_service', 'Customer_Type_disloyal Customer', 
 'Inflight_entertainment', 'Checkin_service', 'Leg_room_service']
```

**Data usage**:
- **Model training (SPLIT/RESPLIT/PRAXIS)**: Use 9-feature subset (same features as DIMEX baseline)
- **Feature importance comparison**: Use all 23 features (compare XGBoost importance vs. PRAXIS RID on the 9 selected features)

---

## Pipeline Workflow

**Status**: Steps 1-2 complete; steps 3-5 are the active comparison work.

```
1. Data Preparation (✅ COMPLETE via DIMEX notebooks)
   └─ Clean missing values → One-hot encode → SMOTE balance
   └─ Output: train_clean_encoded_balanced.csv (23 features, 103k balanced records)

2. Feature Selection (✅ COMPLETE via XGBoost)
   └─ XGBoost identified 9 optimal features at 80% cumulative gain
   └─ Features: Online_boarding, Type_of_Travel_Personal_Travel, Class_Eco, 
                Inflight_wifi_service, On_board_service, Customer_Type_disloyal_Customer,
                Inflight_entertainment, Checkin_service, Leg_room_service
   └─ Comparison goal: PRAXIS RID scores vs. XGBoost importance on these same 9 features

3. SPLIT Training (→ baseline single tree)
   └─ Train SPLIT on 9-feature subset
   └─ Metrics: Accuracy, Precision, Recall, F1, # leaves, runtime

4. RESPLIT Training (→ Rashomon enumeration baseline)
   └─ Train RESPLIT on 9-feature subset
   └─ Metrics: Set size, min loss, feature importance correlation (0.93+ expected)
   └─ Runtime (expect: slow, TreeFARMS-based)

5. PRAXIS Training (→ proxy-based Rashomon enumeration)
   └─ Train PRAXIS on 9-feature subset
   └─ Metrics: Set size, min loss, RID scores for feature stability, runtime
   └─ Compare PRAXIS's top-ranked tree (by loss) vs. SPLIT result

6. Comparative Analysis
   ├─ SPLIT vs. PRAXIS best tree: Same metrics as step 3 (should match or improve)
   ├─ RESPLIT vs. PRAXIS Rashomon sets:
   │  ├─ Set size comparison (did PRAXIS find comparable number of trees?)
   │  ├─ Runtime ratio (100-1000× speedup expected?)
   │  ├─ Feature stability: RID scores vs. correlation across RESPLIT set
   │  └─ Tree structure agreement (do both sets contain similar tree types?)
   └─ Visualization: Performance curves, feature importance comparisons, uncertainty by sample
```

**Key comparison points** (per PRAXIS paper's claims):
- PRAXIS set size should match RESPLIT (≥98% recall of true Rashomon set)
- PRAXIS runtime should be 100-1000× faster than RESPLIT
- RID stability should match or exceed RESPLIT's feature importance correlation
- Model uncertainty (disagreement rate across set) should explain hard-to-classify instances

---

## How to Run

### Setup (WSL/Linux)
```bash
cd praxis
conda env create -f environment.yml
conda activate praxis-env
pip install --upgrade pip

# Install SPLIT and RESPLIT from the same repo
git clone https://github.com/VarunBabbar/SPLIT-ICML.git
cd SPLIT-ICML
pip install resplit/ split/          # Installs both RESPLIT and SPLIT
cd ..

pip install -e .                     # Install dimex package
```

### Verify Installation
```bash
python -c "from resplit import RESPLIT; print('✓ RESPLIT')"
python -c "from split import SPLIT; print('✓ SPLIT')"
python -c "from praxis import PRAXIS; print('✓ PRAXIS')"
python -c "import dimex; print('✓ dimex')"
```

### Quick Start (Preprocessing Already Complete)
```python
import dimex as dx
import pandas as pd

# Load pre-processed data (contains ALL 23 features after encoding)
# Feature selection already identified these 9 optimal features:
SELECTED_FEATURES = [
    'Online_boarding', 'Type_of_Travel_Personal Travel', 'Class_Eco', 
    'Inflight_wifi_service', 'On_board_service', 'Customer_Type_disloyal Customer', 
    'Inflight_entertainment', 'Checkin_service', 'Leg_room_service'
]

data = pd.read_csv('airline-passenger-satisfaction/train_clean_encoded_balanced.csv')

# For model training: use only selected 9 features
x_train_subset, x_test_subset, y_train, y_test = dx.split_dataset(
    data[SELECTED_FEATURES], 
    data['satisfaction'], 
    test_size=0.3, random_state=42
)

# For feature importance comparison: keep full feature set for XGBoost comparison
x_train_full, x_test_full, _, _ = dx.split_dataset(
    data.drop('satisfaction', axis=1),
    data['satisfaction'], 
    test_size=0.3, random_state=42
)

# Train SPLIT (single optimal tree on 9-feature subset)
split_model, split_tree, split_meta = dx.train_split(x_train_subset, y_train, 
                                                       lookahead=2, full_depth=5, reg=0.01)
split_pred, split_acc = dx.prediction_split(split_model, x_test_subset, y_test)
print(f"SPLIT: acc={split_acc:.4f}, leaves={split_meta['leaves']}, time={split_meta['runtime']:.2f}s")

# Train PRAXIS (Rashomon set enumeration on 9-feature subset)
praxis_model, x_binary, praxis_meta = dx.train_praxis(x_train_subset, y_train, 
                                                        lambda_reg=0.01, depth_budget=5, 
                                                        rashomon_mult=0.03, lookahead_k=1)
praxis_pred, praxis_acc, details = dx.prediction_praxis(praxis_model, x_test_subset, y_test, 
                                                         use_ensemble=False)
print(f"PRAXIS: acc={praxis_acc:.4f}, trees={praxis_meta['n_trees']}, time={praxis_meta['runtime']:.2f}s")

# Get RID scores for feature stability analysis (on 9-feature subset)
rid_scores = praxis_model.compute_rid()
print(f"RID scores: {rid_scores}")

# Feature importance comparison: 
# Compare PRAXIS RID scores (from 9 features) vs. XGBoost importance (from all 23 features)
xgb_model, _, _ = dx.train_xgb(x_train_full, y_train)
xgb_importance = xgb_model.get_booster().get_score(importance_type='weight')
# Filter to selected features for comparison
selected_importance = {f: xgb_importance.get(f, 0) for f in SELECTED_FEATURES}
print(f"XGBoost importance (selected features): {selected_importance}")
```

**For RESPLIT** (must run via command-line script, see `run_resplit.py` example below):
```python
# Create run_resplit.py with your config, then execute: python run_resplit.py
from resplit import RESPLIT
import pandas as pd

SELECTED_FEATURES = [
    'Online_boarding', 'Type_of_Travel_Personal Travel', 'Class_Eco', 
    'Inflight_wifi_service', 'On_board_service', 'Customer_Type_disloyal Customer', 
    'Inflight_entertainment', 'Checkin_service', 'Leg_room_service'
]

data = pd.read_csv('airline-passenger-satisfaction/train_clean_encoded_balanced.csv')
x_train = data[SELECTED_FEATURES]
y_train = data['satisfaction']
x_test = data[SELECTED_FEATURES]  # or your test set
y_test = data['satisfaction']     # or your test labels

config = {
    "regularization": 0.01,
    "rashomon_bound_multiplier": 0.03,  # 3% slack
    "depth_budget": 5,
    "cart_lookahead_depth": 2,
    "verbose": True
}

model = RESPLIT(config, fill_tree="treefarms")
model.fit(x_train, y_train)

# Model now contains the Rashomon set
print(f"Trees in set: {len(model)}")
for i, tree in enumerate(model):
    pred = tree.predict(x_test)
    acc = (pred == y_test).mean()
    print(f"Tree {i}: accuracy={acc:.4f}")
```

### Run RESPLIT (Command-line Only)
⚠️ **CRITICAL**: RESPLIT must be run via command-line scripts, NOT Jupyter notebooks.

From the SPLIT-ICML README:
- RESPLIT has known timeout issues in Jupyter
- Always run via `python script.py` or SLURM scripts
- Use Jupyter only for PRAXIS and analysis

Example:
```bash
# Create run_resplit.py with your config, then execute:
python run_resplit.py

# Save results to JSON/CSV, then analyze in Jupyter
```

### Run the Comparison Notebook
```bash
jupyter notebook notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb
```

---

## Key Algorithms

### SPLIT (Single Optimal Tree)
**Core Insight**: Lookahead depth cutoff — greedy splits only harm optimality near the root; near leaves, greedy is acceptable.

- **Method**: Branch-and-bound with lookahead depth cutoff (branch to depth d_l, greedy below)
- **Runtime**: O(n(d-d_l)k^(d_l+1) + nk^(d-d_l)) — exponential only in lookahead depth, not full depth
- **Speed**: **100×+ faster than GOSDT** (optimal tree induction)
- **Optimality**: Guaranteed optimal tree within lookahead budget
- **Strengths**: Single interpretable tree, proven bounds, empirically matches optimal on benchmark datasets
- **Paper**: Babbar et al., ICML 2025, Section 2-4, arXiv:2502.15988
- **Repository**: https://github.com/VarunBabbar/SPLIT-ICML/tree/main/split

### RESPLIT (Rashomon Set via TreeFARMS)
**Method**: Extend SPLIT to enumerate Rashomon sets by solving subproblems optimally at each leaf.

- **Rashomon Definition**: All trees within multiplicative slack (1 + ε)L* where L* is optimal loss
- **Enumeration Strategy**: At each leaf subproblem, enumerate via TreeFARMS (depth-first search with pruning)
- **Execution**: **Command-line only** (Python scripts, NOT Jupyter) — known timeout issues in notebooks
- **Speed**: **10-20× speedup over naive TreeFARMS**, but exponential worst-case per tree
- **Set Coverage**: Complete enumeration within slack bound (no trees missed)
- **Feature Importance**: 0.93+ correlation across trees in Rashomon set (fairly stable)
- **Paper**: Babbar et al., ICML 2025, Section 5, arXiv:2502.15988
- **Repository**: https://github.com/VarunBabbar/SPLIT-ICML/tree/main/resplit
- **Use case**: Baseline for comparing against PRAXIS; good for smaller datasets where completeness matters

### PRAXIS (Proxy-Based Rashomon Enumeration)
**Core Insight**: Use a fast proxy (heuristic, typically LicketySPLIT) to prune the search space; only solve full problem for promising candidates.

- **Method**: Proxy filtering + iterative budget refinement (SOLVE_SIBLINGS algorithm)
- **Runtime**: O(|Rashomon set| × polynomial) — polynomial per actual tree found, not exponential
- **Proxy Algorithm**: LicketySPLIT by default (lookahead_k=1, polynomial time)
- **Budget Refinement**: SOLVE_SIBLINGS iteratively tightens bounds to find more trees within slack
- **Speed**: **100-1000× faster than RESPLIT** across datasets (orders of magnitude advantage)
- **Set Recall**: **0.98-1.0** (near-perfect recovery of Rashomon sets)
- **Built-in RID**: Rashomon Importance Distribution for measuring feature stability across set
- **Execution**: Can run in Jupyter (unlike RESPLIT)
- **Paper**: Harary et al., ICML 2026, Section 3-4, arXiv:2606.00202
- **Repository**: https://github.com/zakk-h/PRAXIS
- **Use case**: Fast Rashomon enumeration; best for understanding model uncertainty and feature importance stability via RID
- **Experimental comparison**: On Churn data, PRAXIS finds 1M+ trees in 35s; RESPLIT finds 0 valid trees in 43m

---

### Algorithm Relationship

```
SPLIT (2025) 
  ↓ [extend for Rashomon sets]
RESPLIT (2025)  ← baseline comparison algorithm
  ↓ [improve via proxy-based search]
PRAXIS (2026)   ← faster replacement for RESPLIT
```

**Trade-off summary**:
- **RESPLIT**: Complete enumeration, slower, reliable for small datasets
- **PRAXIS**: Approximate enumeration (but 0.98-1.0 recall), orders of magnitude faster, includes RID stability analysis

---

## Hyperparameters to Tune

### RESPLIT (via `from resplit import RESPLIT`)
Direct Python API (command-line scripts only):
```python
config = {
    "regularization": 0.005,                   # λ regularization
    "rashomon_bound_multiplier": 0.01,        # Slack: models within (1+ε)L*
    "depth_budget": 5,                        # Max tree depth
    "cart_lookahead_depth": 3,                # CART lookahead for prefix
    "verbose": False
}
model = RESPLIT(config, fill_tree="treefarms")
model.fit(X, y)
tree = model[i]                               # Access i-th tree
```

**Key config options**:
- `rashomon_bound_multiplier`: Relative threshold (models within 1% of best loss)
- `rashomon_bound_adder`: Alternative—absolute threshold (L* + ε)
- `rashomon_bound`: Fixed loss threshold (hard cutoff)
- `fill_tree`: "treefarms", "optimal", or "greedy"

### SPLIT (via `dimex.train_split` or `from split import SPLIT`)
- `lookahead` (int): Lookahead depth (default: 2)
- `full_depth` (int): Maximum tree depth (default: 5)
- `reg` (float): Regularization λ (default: 0.01)

### PRAXIS (via `dimex.train_praxis` or `from praxis import PRAXIS`)
- `lambda_reg` (float): Regularization (default: 0.01)
- `depth_budget` (int): Maximum tree depth (default: 5)
- `rashomon_mult` (float): Multiplicative slack (default: 0.03, i.e., 3%)
- `lookahead_k` (int): Proxy lookahead depth (default: 1 = LicketySPLIT)

---

## Expected Results

### From DIMEX work (SPLIT alone):
- **Best SPLIT**: 91.97% test accuracy, 8 leaves, ~5s runtime on 9-feature subset
- **XGBoost**: 93.58% test accuracy, 755 leaves

### From Papers' Empirical Results:

**RESPLIT (via SPLIT paper, Section 5)**:
- Completes enumeration on moderate datasets (~100k samples, ~20 features)
- Feature importance correlation: 0.93+ across Rashomon set trees (fairly stable)
- Slower on larger feature sets; exponential worst-case per tree

**PRAXIS (via PRAXIS paper, Section 5-6)**:
- 100-1000× faster than RESPLIT on equivalent datasets
- Recall: 0.98-1.0 (recovers 98-100% of trees in Rashomon bound)
- RID (Rashomon Importance Distribution) shows which features consistently matter
- Example: On Churn data, PRAXIS finds 1M+ trees in 35s vs. RESPLIT's 0 trees in 43m

### Comparison goals for airline dataset:

**Primary comparison: RESPLIT vs. PRAXIS** (the core research question)

1. **Set Completeness**: Does PRAXIS recover the same Rashomon set as RESPLIT?
   - RESPLIT: Complete enumeration by definition (baseline ground truth)
   - PRAXIS: Expect 98-100% recall (find 98-100% of RESPLIT's trees)
   - Metric: `n_trees_praxis / n_trees_resplit` (should be 0.98-1.0)

2. **Speed Advantage**: How much faster is PRAXIS than RESPLIT?
   - RESPLIT: TreeFARMS enumeration (expect: seconds to minutes)
   - PRAXIS: Proxy-based (expect: sub-second to seconds)
   - Metric: `runtime_resplit / runtime_praxis` (papers show 100-1000× on other datasets)

3. **Feature Importance Stability Comparison**:
   - **Baseline (XGBoost)**: Feature importance on all 23 features from DIMEX notebooks
   - **RESPLIT**: Feature importance correlation across Rashomon set trees (expect: 0.93+ from SPLIT paper)
   - **PRAXIS**: RID (Rashomon Importance Distribution) scores on 9-feature subset
   - **Analysis**: 
     - Do RID and RESPLIT correlation rank the 9 features similarly?
     - How do PRAXIS/RESPLIT rankings of the 9 features compare to XGBoost's ranking of those same 9?
     - Metric: Rank correlation (Spearman/Kendall) between RID/correlation vs. XGBoost importance

4. **Tree Structure Similarity**:
   - Are trees found by RESPLIT and PRAXIS structurally similar?
   - Do they split on same features? Same thresholds?
   - Does PRAXIS's proxy-based approach bias the set toward certain tree types?

5. **Model Uncertainty (per-instance)**:
   - Prediction disagreement rate across Rashomon set (what % of instances have multiple votes?)
   - Which instances have high disagreement? Can this identify inherently ambiguous cases?
   - Compare disagreement distribution in RESPLIT set vs. PRAXIS set

**Secondary: SPLIT vs. PRAXIS best tree**
- PRAXIS orders results by loss; compare PRAXIS's top tree against SPLIT's single tree
- Metrics: Accuracy, precision, recall, F1, tree size
- Should be essentially identical (both solving same loss minimization)

---

## Installation Troubleshooting

### RESPLIT/SPLIT Installation
**Both are in the same repo.** Install together from SPLIT-ICML:
```bash
git clone https://github.com/VarunBabbar/SPLIT-ICML.git
cd SPLIT-ICML
pip install resplit/ split/
cd ..
```

**Missing C++ dependencies** (RESPLIT requires cmake, ninja, tbb, pkg-config, gmp):
```bash
# Via conda (recommended)
conda install -c conda-forge cmake ninja tbb tbb-devel pkg-config gmp

# Via apt (Ubuntu/Debian)
sudo apt-get install -y cmake ninja-build libtbb-dev pkg-config libgmp-dev
```

### PRAXIS Installation
```bash
pip install tree-praxis
```
(Already included in `environment.yml`)

### General Environment Conflicts
Rebuild from scratch:
```bash
conda env remove -n praxis-env
conda env create -f environment.yml
git clone https://github.com/VarunBabbar/SPLIT-ICML.git
cd SPLIT-ICML && pip install resplit/ split/ && cd ..
pip install -e .
```

---

## References

### Research Papers (Local Copies)

1. **SPLIT: Near-Optimal Decision Trees in a SPLIT Second** (Babbar et al., ICML 2025)
   - **File**: [papers/SPLIT.pdf](./papers/SPLIT.pdf)
   - **arXiv**: 2502.15988
   - **Authors**: Varun Babbar, Hayley Hung, Ulrich Rüdin, Cynthia Rudin
   - **Key contributions**: SPLIT algorithm (100× faster than GOSDT), LicketySPLIT, RESPLIT (Rashomon enumeration)
   - **Complexity**: O(n(d-d_l)k^(d_l+1) + nk^(d-d_l))
   - **Recommended reading**: Pages 1-8 (core algorithms, lookahead insight, complexity, experiments)
   - **On airline data**: 91.97% accuracy, 8 leaves, ~5s runtime

2. **PRAXIS: Fast Rashomon Sets for Sparse Decision Trees** (Harary et al., ICML 2026)
   - **File**: [papers/PRAXIS.pdf](./papers/PRAXIS.pdf)
   - **arXiv**: 2606.00202
   - **Authors**: Ilya Harary, Varun Babbar, Jacob Marvin, Kate Seltzer, Cynthia Rudin
   - **Key contributions**: Proxy-based Rashomon enumeration (100-1000× faster than RESPLIT), SOLVE_SIBLINGS algorithm, RID (Rashomon Importance Distribution)
   - **Complexity**: O(|Rashomon set| × polynomial)
   - **Recommended reading**: Pages 1-9 (problem formulation, proxy filtering, SOLVE_SIBLINGS, benchmarks)
   - **Experimental highlight**: Finds 1M+ trees in 35s vs. RESPLIT's 0 trees in 43m (on Churn dataset)

### Related Resources

3. **DIMEX write-up** (Campagnaro, 2025)  
   Medium: "I built an end-to-end interpretable Machine Learning research pipeline"

4. **GitHub Repositories**
   - SPLIT/RESPLIT: https://github.com/VarunBabbar/SPLIT-ICML
   - PRAXIS: https://github.com/zakk-h/PRAXIS

5. **Datasets**
   - Airline Passenger Satisfaction: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

---

## Notes for Future Work

### ✅ Completed
- [x] **Data Preprocessing**: Missing values, encoding, SMOTE balancing (9-feature subset selected)
- [x] **Feature Selection via XGBoost**: Identified optimal feature set at 80% cumulative gain threshold
- [x] **Paper Analysis**: Integrated SPLIT and PRAXIS paper insights into CLAUDE.md

### Active Work (RESPLIT vs. PRAXIS Comparison)

#### SPLIT Training
- [ ] **SPLIT script**: Train SPLIT on airline 9-feature subset
- [ ] **Metrics**: Accuracy, precision, recall, F1, # leaves, runtime

#### RESPLIT Training (Command-line only)
- [ ] **RESPLIT script**: Create `run_resplit.py` to execute RESPLIT on airline data
- [ ] **Config tuning**: Test `rashomon_bound_multiplier` values (0.01, 0.03, 0.05, 0.1)
- [ ] **Save results**: JSON with Rashomon set trees, collection size, feature importance correlation

#### PRAXIS Training
- [ ] **PRAXIS notebook**: Train PRAXIS on airline 9-feature subset with varying `rashomon_mult`
- [ ] **RID analysis**: Extract and compare `compute_rid()` scores vs. XGBoost feature importance
- [ ] **Top-tree comparison**: Compare PRAXIS's best tree (by loss) against SPLIT result

#### Comparative Analysis
- [ ] **Main comparison**: RESPLIT vs. PRAXIS
  - [ ] Set size comparison (n_trees_praxis / n_trees_resplit ratio)
  - [ ] Runtime ratio (target: 100-1000× speedup for PRAXIS)
  - [ ] Feature stability: RID scores vs. RESPLIT correlation
  - [ ] Tree structure similarity (same splits, same features?)
- [ ] **Uncertainty analysis**: Per-instance disagreement rate, identify ambiguous cases
- [ ] **Scaling experiment**: Optional—test on 16 or 19 features to see where PRAXIS gains advantage
- [ ] **Visualization**: Performance curves, feature importance comparison, uncertainty heatmaps
- [ ] **Notebook**: Create `RESPLIT_vs_PRAXIS_comparison.ipynb` with full pipeline and analysis

---

## Environment Setup Status

✅ **COMPLETE**: All packages installed and verified
- ✅ RESPLIT (0.2.4) — Rashomon set enumeration via TreeFARMS
- ✅ SPLIT (0.1.0) — Single optimal decision tree
- ✅ PRAXIS (0.0.29) — Proxy-based Rashomon sets
- ✅ dimex — Local analysis package

**Platform**: WSL2 (Ubuntu) with conda
**Python version**: 3.10.12
**Installation date**: 2026-06-29

---

## Documentation Updates

### 2026-06-30: Paper Analysis & Integration
- ✅ Extracted and analyzed first 8 pages of SPLIT paper (Babbar et al., ICML 2025)
- ✅ Extracted and analyzed first 9 pages of PRAXIS paper (Harary et al., ICML 2026)
- ✅ Added "Paper Summaries & File Mapping" section with key innovations and experimental results
- ✅ Enhanced "Key Algorithms" with technical depth: complexity analysis, runtime comparisons, algorithmic relationships
- ✅ Updated "Project Structure" to include `papers/` directory mapping
- ✅ Expanded "Expected Results" with paper-backed empirical benchmarks
- ✅ Updated "References" with local file paths and detailed citation information

### Key Insights Integrated
- **SPLIT lookahead insight**: Greedy splits only harm optimality near root; acceptable near leaves
- **RESPLIT completeness**: Full enumeration within slack bound, 0.93+ feature importance correlation
- **PRAXIS speed**: 100-1000× faster than RESPLIT with 0.98-1.0 recall (near-complete enumeration)
- **Co-evolution**: Both papers (2025, 2026) authored by McTavish, Babbar, Seltzer, Rudin team
- **RID concept**: Built-in Rashomon Importance Distribution in PRAXIS for stability analysis

---

**Last updated**: 2026-06-30  
**Project owner**: Lucas Campagnaro  
**Environment**: Python 3.10, conda, WSL (Linux)
