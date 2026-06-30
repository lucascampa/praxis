# PRAXIS
This project extends the original [DIMEX](../dimex/) work by comparing two Rashomon set enumeration algorithms: **RESPLIT** (from the SPLIT paper) and **PRAXIS** (an evolution with a proxy-based approach).

## Project Goals
Build on the DIMACS externship work demonstrating interpretable ML can match black-box performance. This phase compares two methods for exploring sets of near-optimal sparse decision trees:
- **RESPLIT**: Enumerates Rashomon sets using TreeFARMS at leaf subproblems
- **PRAXIS** ([paper](https://arxiv.org/abs/2606.00202), [repo](https://github.com/zakk-h/PRAXIS)): Enumerates Rashomon sets using a proxy-based algorithm with iterative refinement

Both are applied to the same airline satisfaction dataset to evaluate scalability, Rashomon set size, tree quality, and feature importance stability.

See the original [write-up](https://medium.com/@lucascampagnaro/i-built-an-end-to-end-interpretable-machine-learning-research-pipeline-0ba67d0ba700) for DIMEX context.

## Features & Workflow
- **Preprocessing**: Missing-value cleaning, label binarization, categorical encoding, SMOTE, and undersampling
- **XGBoost toolkit**: Training, model metadata, feature selection, and predictions
- **SPLIT toolkit**: Training, model metadata, predictions, and RESPLIT Rashomon set enumeration
- **PRAXIS toolkit**: Training, Rashomon set enumeration, disagreement analysis, and RID (Rashomon Importance Distribution)

1. Preprocess the data (same as DIMEX)
2. Feature selection with XGBoost (same as DIMEX)
3. Train RESPLIT and PRAXIS on selected features
4. Compare Rashomon sets, tree counts, runtimes, and feature importance

## Project Structure
```
├── airline-passenger-satisfaction/ # Dataset used
├── dimex/
├── notebooks/ # Notebooks running dimex with different seed values
├── results/ # Comparison table of the different results from running XGBoost and SPLIT using different parameters and seed values
├── environment.yml # Conda environment file (instead of requirements.txt)
├── .gitignore
├── LICENSE
├── README.md
├── SETUP.md # Setup guide
└── setup.py # Run this to install dimex
```

## Installation
See [SETUP.md](SETUP.md) for full installation instructions.

## Quick Start

```python
import dimex as dx

# Preprocess: clean, encode, balance
data_clean, stats, fname = dx.clean_missing('airline-passenger-satisfaction/train.csv')
data_encoded, fname = dx.binarize_encode(fname, 'satisfied', 'neutral or dissatisfied')
x_train, x_test, y_train, y_test = dx.split_dataset(data_encoded, test_size=0.3, random_state=42)
x_train_bal, y_train_bal = dx.smote(x_train, y_train)

# Feature selection with XGBoost
xgb_model, _, _ = dx.train_xgb(x_train_bal, y_train_bal)
x_selected, gain_info = dx.cumulative_gain(xgb_model, x_train_bal, y_train_bal, 0.80)

# Train SPLIT (single optimal tree)
split_model, split_tree, split_meta = dx.train_split(x_selected, y_train_bal, 
                                                      lookahead=2, full_depth=5, reg=0.01)
split_pred, split_acc = dx.prediction_split(split_model, x_test, y_test)

# Train PRAXIS (Rashomon set)
praxis_model, x_binary, praxis_meta = dx.train_praxis(x_selected, y_train_bal,
                                                       lambda_reg=0.01, depth_budget=5, 
                                                       rashomon_mult=0.03, lookahead_k=1)
praxis_pred, praxis_acc, details = dx.prediction_praxis(praxis_model, x_test, y_test, use_ensemble=False)

print(f"SPLIT:  {split_acc:.4f} accuracy, {split_meta['leaves']} leaves, {split_meta['runtime']:.2f}s")
print(f"PRAXIS: {praxis_acc:.4f} accuracy, {praxis_meta['n_trees']} trees, {praxis_meta['runtime']:.2f}s")

# Visualize
dx.cm(y_test, split_pred[0])
```

**For detailed workflows**, see notebooks:
- `notebooks/Black-Box to Glass-Box modeling [RANDOM_SEED=42].ipynb` — DIMEX baseline
- `notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb` — Full RESPLIT vs. PRAXIS comparison

## Expected Results

**From DIMEX (SPLIT baseline):**
- SPLIT: 91.97% accuracy, 8 leaves, ~5s on 9-feature subset
- XGBoost: 93.58% accuracy, 755 leaves
- **Tradeoff**: 1.6% accuracy loss for 94% fewer leaves (interpretability win)

**PRAXIS Comparison Targets:**
- How large is the PRAXIS Rashomon set vs. RESPLIT's?
- Do PRAXIS trees have better feature importance stability (RID)?
- Runtime: PRAXIS vs. RESPLIT?
- Uncertainty quantification: Can Rashomon disagreement improve model calibration?

Results will be in `notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb` and `results/`.

## References

**Algorithms:**
- Babbar, V., McTavish, H., Rudin, C., Seltzer, M. (2025). Near-Optimal Decision Trees in a SPLIT Second. *arXiv preprint* arXiv:2502.15988. [[Link](https://arxiv.org/abs/2502.15988)]
- Harary, Z., et al. (2026). Fast Rashomon Sets for Sparse Decision Trees. *ICML 2026*. [[Link](https://arxiv.org/abs/2606.00202)]

**Original DIMEX Work:**
- Campagnaro, L. (2025). _I built an end-to-end interpretable Machine Learning research pipeline_. Medium. [[Link](https://medium.com/@lucascampagnaro/i-built-an-end-to-end-interpretable-machine-learning-research-pipeline-0ba67d0ba700)]

**Dataset:**
- Airline Passenger Satisfaction. Kaggle. [[Link](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)]

## Documentation

For complete context, see [CLAUDE.md](CLAUDE.md) and [SPLIT context.md](SPLIT%20context.md).