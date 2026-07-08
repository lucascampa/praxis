# PRAXIS
This project extends the original [DIMEX](../dimex/) work by comparing two Rashomon set enumeration algorithms: **RESPLIT** (from the SPLIT paper) and **PRAXIS** (an evolution with a proxy-based approach).

## Project Goals
Build on the DIMACS externship work demonstrating interpretable ML can match black-box performance. This phase compares two methods for exploring sets of near-optimal sparse decision trees:
- **RESPLIT**: Enumerates Rashomon sets using TreeFARMS at leaf subproblems
- **PRAXIS** ([paper](https://arxiv.org/abs/2606.00202), [repo](https://github.com/zakk-h/PRAXIS)): Enumerates Rashomon sets using a proxy-based algorithm with iterative refinement

Both are applied to the same airline satisfaction dataset to evaluate scalability, Rashomon set size, tree quality, and feature importance stability.

See the original [write-up](https://medium.com/@lucascampagnaro/i-built-an-end-to-end-interpretable-machine-learning-research-pipeline-0ba67d0ba700) for DIMEX context.

## Features & Workflow
- **Preprocessing** (`dimex`): Missing-value cleaning, label binarization, categorical encoding, SMOTE, and undersampling
- **XGBoost toolkit** (`dimex`): Training, model metadata, feature selection, and predictions
- **SPLIT toolkit** (`dimex`): Training, model metadata, and predictions
- **RESPLIT runner** (`run_resplit.py`): Command-line Rashomon set enumeration, with workarounds for three defects in the released RESPLIT code (see [RESPLIT investigation.md](RESPLIT%20investigation.md))
- **Rashomon set analysis** (`prxs`): Export PRAXIS Rashomon sets to JSON, rebuild trees from their path representation, and draw labeled tree diagrams

1. Preprocess the data (same as DIMEX)
2. Feature selection with XGBoost (same as DIMEX)
3. Train SPLIT, RESPLIT, PRAXIS, and STreeD on the selected features ([notebooks/comparison.ipynb](notebooks/comparison.ipynb) + `run_resplit.py`)
4. Compare Rashomon sets, tree counts, runtimes, and feature importance
5. Analyze Rashomon set *structure*: tree diagrams, split-variable usage, root stability ([notebooks/tree_structure.ipynb](notebooks/tree_structure.ipynb))

## Project Structure
```
├── data/airline-passenger-satisfaction/ # Dataset (raw, cleaned, encoded, balanced, binarized)
├── dimex/ # DIMACS Externship package (old project: SPLIT vs XGBoost)
├── prxs/ # Current-project package: Rashomon set structure analysis
├── run_resplit.py # RESPLIT runner (command-line only)
├── verify_bound.py, verify_leak.py # Standalone repros of RESPLIT defects
├── notebooks/ # DIMEX seed notebooks + comparison.ipynb + tree_structure.ipynb
├── results/ # Comparison tables (results*.json), Rashomon set export (praxis_trees.json), cached RESPLIT models
├── environment.yml # Conda environment file (instead of requirements.txt)
├── .gitignore
├── LICENSE
├── README.md
├── SETUP.md # Setup guide
└── setup.py # Run this to install the dimex and prxs packages
```

## Installation
See [SETUP.md](SETUP.md) for full installation instructions.

## Quick Start

**1. Model comparison** — run [notebooks/comparison.ipynb](notebooks/comparison.ipynb)
(XGBoost, SPLIT, PRAXIS, STreeD), with `python run_resplit.py` from a terminal for the
RESPLIT part (it cannot run in Jupyter). The notebook builds the five-model table and
exports the PRAXIS Rashomon set to `results/praxis_trees.json`.

**2. Rashomon set structure analysis** — run
[notebooks/tree_structure.ipynb](notebooks/tree_structure.ipynb). It works entirely
from the JSON export (no re-fitting) via the `prxs` package:

```python
from prxs import load_trees, draw_tree

payload = load_trees('results/praxis_trees.json')
trees = payload['trees']                    # ranked by objective (best first)
print(payload['n_trees_total'], 'trees')

# Diagram any tree: feature-named nodes, yes/no edges, class-colored leaves
draw_tree(trees[0], payload['feature_names'])
```

Node labels can be translated to plain English through the hand-written
`feature_labels` dict in the notebook (e.g. `Customer_Type_disloyal Customer <= 0.5`
→ "Loyal customer?").

## Results (airline dataset, 9-feature subset, λ=0.005, depth 5, 1% Rashomon slack)

| Model | Test accuracy | Train loss | Size | Runtime | Rashomon set |
|---|---|---|---|---|---|
| XGBoost | 92.9% | 0.082 | 100 trees, 741 leaves | 0.4s | — |
| SPLIT | 91.9% | 0.136 | 8 leaves | 6.2s | — |
| RESPLIT (best tree) | 88.9% | 0.149 | 6 leaves | 3,388s | 1.46×10⁹ trees* |
| PRAXIS (best tree) | 91.9% | 0.136 | 8 leaves | **0.7s** | 140 trees |
| STreeD | 91.9% | 0.136 | 8 leaves | 147s | — |

\* RESPLIT's count is inflated by its compact-trie representation and its best tree
provably misses the optimum — see [RESPLIT investigation.md](RESPLIT%20investigation.md).

**Key findings so far:**
- SPLIT, PRAXIS, and STreeD find the *identical* optimal tree (100% prediction agreement)
- PRAXIS is ~4,800× faster than RESPLIT on this dataset and actually contains the optimal tree
- All 140 Rashomon set trees share the same root split (`Inflight_wifi_service <= 0.5`,
  i.e. "WiFi not rated" → satisfied); five splits are universal across the set, and
  structural diversity lives in the lower levels (7–10 leaves, depth uniformly 5)

## References

**Algorithms:**
- Babbar, V., McTavish, H., Rudin, C., Seltzer, M. (2025). Near-Optimal Decision Trees in a SPLIT Second. *arXiv preprint* arXiv:2502.15988. [[Link](https://arxiv.org/abs/2502.15988)]
- Harary, Z., et al. (2026). Fast Rashomon Sets for Sparse Decision Trees. *ICML 2026*. [[Link](https://arxiv.org/abs/2606.00202)]

**Original DIMEX Work:**
- Campagnaro, L. (2025). _I built an end-to-end interpretable Machine Learning research pipeline_. Medium. [[Link](https://medium.com/@lucascampagnaro/i-built-an-end-to-end-interpretable-machine-learning-research-pipeline-0ba67d0ba700)]

**Dataset:**
- Airline Passenger Satisfaction. Kaggle. [[Link](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)]

## Documentation

- [CLAUDE.md](CLAUDE.md) — operational context: how to run everything, API traps, findings log
- [papers/PAPERS.md](papers/PAPERS.md) — paper summaries (SPLIT 2025, PRAXIS 2026)
- [data/DATA.md](data/DATA.md) — dataset documentation and feature lists
- [RESPLIT investigation.md](RESPLIT%20investigation.md) — why RESPLIT underperforms here (defect record)
- [SPLIT context.md](SPLIT%20context.md) — DIMEX (original externship) background