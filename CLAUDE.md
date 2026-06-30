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

- **Size**: 103,904 training records (after missing-value removal)
- **Target**: Binary classification (satisfied=1, neutral/dissatisfied=0)
- **Features**: 23 total
  - 4 continuous: Age, Flight_Distance, Departure_Delay, Arrival_Delay
  - 14 service ratings (1–5): WiFi, Online_boarding, Seat_comfort, etc.
  - 5 categorical (one-hot encoded): Gender, Customer_Type, Type_of_Travel, Class

**Best feature set (from XGBoost selection)**: 9 features at 80% cumulative gain threshold

---

## Pipeline Workflow

```
1. Data Preparation (dimex.preprocessing)
   └─ Clean missing values → One-hot encode → SMOTE balance

2. Feature Selection (dimex.xgb_runner)
   └─ Train XGBoost → Rank by feature importance → Select top N%

3. Model Training (dimex.split_runner, dimex.praxis_runner)
   ├─ SPLIT: Single optimal sparse tree
   └─ PRAXIS: Rashomon set of near-optimal trees

4. Evaluation & Comparison
   ├─ Single-tree metrics (accuracy, # leaves, runtime)
   ├─ Rashomon set metrics (set size, min objective, feature importance stability)
   └─ Visualization (plots, tables)
```

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

### Quick Start
```python
import dimex as dx

# Preprocess
data_clean, stats, fname = dx.clean_missing('airline-passenger-satisfaction/train.csv')
data_encoded, fname = dx.binarize_encode(fname, 'satisfaction', 'neutral or dissatisfied')

# Split
x_train, x_test, y_train, y_test = dx.split_dataset(data_encoded, test_size=0.3, random_state=42)

# Balance
x_train_bal, y_train_bal = dx.smote(x_train, y_train)

# Feature selection via XGBoost
xgb, _, _ = dx.train_xgb(x_train_bal, y_train_bal)
x_top, gain_info = dx.cumulative_gain(xgb, x_train_bal, y_train_bal, 0.80)

# Train models
split_model, split_tree, split_meta = dx.train_split(x_top, y_train_bal, lookahead=2, full_depth=5, reg=0.01)
praxis_model, x_binary, praxis_meta = dx.train_praxis(x_top, y_train_bal, lambda_reg=0.01, depth_budget=5, rashomon_mult=0.03, lookahead_k=1)

# Evaluate
split_pred, split_acc = dx.prediction_split(split_model, x_test, y_test)
praxis_pred, praxis_acc, details = dx.prediction_praxis(praxis_model, x_test, y_test, use_ensemble=False)

print(f"SPLIT: {split_acc:.4f}, leaves={split_meta['leaves']}, time={split_meta['runtime']:.2f}s")
print(f"PRAXIS: {praxis_acc:.4f}, trees={praxis_meta['n_trees']}, time={praxis_meta['runtime']:.2f}s")
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

### RESPLIT (Rashomon Set via TreeFARMS)
- **Method**: Enumeration of Rashomon sets using TreeFARMS at leaf subproblems
- **Rashomon enumeration**: Via multiplicative slack (`rashomon_bound_multiplier` = relative threshold)
- **Execution**: Command-line only (Python scripts, NOT Jupyter)
- **Speed**: TreeFARMS-based enumeration (slower than PRAXIS on most datasets)
- **Strengths**: Direct enumeration, complete Rashomon set within slack bound
- **Paper**: Babbar et al., ICML 2025, arXiv:2502.15988
- **Repository**: https://github.com/VarunBabbar/SPLIT-ICML/tree/main/resplit

### SPLIT (Single Optimal Tree)
- **Method**: Branch-and-bound with lookahead depth cutoff, post-process with GOSDT
- **Speed**: 100×+ faster than GOSDT
- **Strengths**: Single interpretable tree, proven optimality bounds
- **Paper**: Babbar et al., ICML 2025, arXiv:2502.15988

### PRAXIS (Proxy-Based Rashomon Sets)
- **Method**: Proxy-based algorithm (LicketySPLIT) with iterative budget refinement
- **Rashomon enumeration**: Multiplicative slack (1 + rashomon_mult)
- **Execution**: Can run in Jupyter (unlike RESPLIT)
- **Speed**: Faster than RESPLIT via iterative refinement
- **Strengths**: Speed, built-in RID computation for feature importance stability
- **Paper**: Harary et al., ICML 2026, arXiv:2606.00202
- **Repository**: https://github.com/zakk-h/PRAXIS

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

From DIMEX work (SPLIT alone):
- **Best SPLIT**: 91.97% test accuracy, 8 leaves, ~5s runtime on 9-feature subset
- **XGBoost**: 93.58% test accuracy, 755 leaves

Comparison goals:
- How many trees in RESPLIT's vs. PRAXIS's Rashomon set?
- Do PRAXIS trees have better or worse feature importance stability?
- Runtime comparison: PRAXIS vs. RESPLIT?
- Can Rashomon sets explain model uncertainty on the airline data?

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

1. **SPLIT paper** (Babbar et al., 2025)  
   Title: "Near-Optimal Decision Trees in a SPLIT Second"  
   arXiv: 2502.15988

2. **PRAXIS paper** (Harary et al., 2026)  
   Title: "Fast Rashomon Sets for Sparse Decision Trees"  
   arXiv: 2606.00202

3. **DIMEX write-up** (Campagnaro, 2025)  
   Medium: "I built an end-to-end interpretable Machine Learning research pipeline"

4. **Original datasets**
   - Airline Satisfaction: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

---

## Notes for Future Work

### RESPLIT-Specific
- [ ] **RESPLIT script**: Create `run_resplit.py` to execute RESPLIT on airline data (cmd-line only)
- [ ] **RESPLIT config tuning**: Test `rashomon_bound_multiplier` values (0.01, 0.03, 0.05, 0.1)
- [ ] **RESPLIT results**: Save Rashomon set to JSON, parse tree collection size

### PRAXIS-Specific
- [ ] **PRAXIS notebook**: Create cells for PRAXIS training with varying `rashomon_mult`
- [ ] **RID analysis**: Use PRAXIS's built-in `compute_rid()` for feature importance stability
- [ ] **Ensemble voting**: Compare single-tree vs. ensemble predictions on test set

### Comparative Analysis
- [ ] **Notebook**: Create `RESPLIT_vs_PRAXIS_comparison.ipynb` with full pipeline
- [ ] **Benchmarking**: Runtime, Rashomon set size, tree quality across multiple random seeds
- [ ] **Disagreement**: Visualize per-sample prediction variance across trees in both sets
- [ ] **Feature importance**: Compare RESPLIT enumeration stability vs. PRAXIS RID
- [ ] **Scaling**: Test on larger feature sets (16, 19 features) to see where PRAXIS wins
- [ ] **Documentation**: Add examples to README for RESPLIT and PRAXIS workflows

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

**Last updated**: 2026-06-29  
**Project owner**: Lucas Campagnaro  
**Environment**: Python 3.10, conda, WSL (Linux)
