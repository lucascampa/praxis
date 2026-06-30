# SPLIT Project Context

## The SPLIT Paper

**"Near-Optimal Decision Trees in a SPLIT Second"** — Babbar, McTavish, Rudin, Seltzer (2025, arXiv:2502.15988)

### Core Idea

Decision tree optimization sits between two extremes:
- **Greedy methods** (CART, C4.5): fast, but provably suboptimal — pick the locally best split at each node
- **Optimal methods** (GOSDT, MurTree): globally optimal, but exponentially slow with depth and features

SPLIT exploits a structural property of near-optimal trees: **splits closer to the leaves tend to be greedy anyway**. The proportion of greedy splits in near-optimal trees increases monotonically as you go deeper. This means the exponential search near the leaves yields only marginal gains over just being greedy there.

### The Algorithm Family

**SPLIT** — performs branch-and-bound (dynamic programming) up to a user-chosen *lookahead depth* `d_l`, then switches to greedy splitting. Post-processes by replacing greedy subtrees with fully optimal GOSDT subtrees.

**LicketySPLIT** — polynomial-time variant. Recursively applies SPLIT with lookahead depth 1, then replaces leaves with LicketySPLIT subtrees instead of GOSDT. Runtime: O(nk²d²).

**RESPLIT** — extends SPLIT to approximate the Rashomon set (the set of all near-optimal trees). Calls ModifiedTreeFARMS at the lookahead depth, then at each leaf calls TreeFARMS to find all near-optimal subtrees. Enables scalable variable importance via the Rashomon Importance Distribution (RID).

### Objective Being Optimized

Minimizes regularized loss over the training set:

```
L*(D, d, λ) = min_{T} [ (1/N) Σ 1[T(xᵢ) ≠ yᵢ] + λ·S(T) ]  subject to depth(T) ≤ d
```

where `S(T)` is the number of leaves and `λ` is a sparsity penalty. This trades off accuracy vs. tree size.

### Key Theoretical Results

- **Theorem 5.1**: SPLIT runtime is `O(n(d-d_l)k^(d_l+1) + nk^(d-d_l))` — exponentially faster than full optimal search `O((2k)^d)`
- **Corollary 5.2**: Optimal lookahead depth minimizing runtime is `d_l = (d-1)/2` for large `k`
- **Corollary 5.3**: SPLIT saves a factor of `O(k^((d-1)/2) · (d/2)!)` in runtime vs. GOSDT
- **Theorem 5.4**: LicketySPLIT runs in polynomial time `O(nk²d²)`
- **Theorem 5.5**: SPLIT can be arbitrarily more accurate than greedy methods (proven via Tribes construction)

### Experimental Results

Tested on 9 datasets (HELOC, COMPAS, Netherlands, Covertype, Adult, Bike, Hypothyroid, Spambase, Bank, HIV), depth budget 5, lookahead depth 2.

- **Speed**: SPLIT and LicketySPLIT are **100×+ faster** than GOSDT for most λ values
- **Accuracy/Sparsity**: Consistently on the Pareto frontier of test loss vs. number of leaves
- **RESPLIT**: 10–20× faster Rashomon set computation; near-perfect correlation (τ ≈ 0.93–1.0) with full RID variable importances

### Implementation Notes

SPLIT is built on top of GOSDT by modifying the `get_bounds` function. When the search depth equals the lookahead depth, instead of generic branch-and-bound bounds, both `lb` and `ub` are fixed to the loss of a greedy subtree trained at that subproblem — collapsing the subproblem to a solved state and preventing further search below that depth.

---

## The DIMEX Project

**DIMEX** is a research pipeline built for the Rutgers DIMACS Summer 2025 externship. The goal: benchmark interpretable ML (SPLIT) against black-box ML (XGBoost) on a real-world tabular dataset, and demonstrate how interpretable models can approach black-box performance.

### Dataset: Airline Passenger Satisfaction

Source: Kaggle public dataset.

- **Training**: 103,904 records (310 missing values removed → 103,594 clean)
- **Testing**: 25,976 records (83 missing values removed → 25,893 clean)
- **Target**: Binary — `satisfied` (1) vs. `neutral or dissatisfied` (0)
- **Class balance**: ~43% satisfied, 57% dissatisfied (imbalanced → addressed with SMOTE)

**Features after preprocessing (23 total)**:
- 4 continuous: Age, Flight_Distance, Departure_Delay, Arrival_Delay
- 14 service ratings (1–5 scale): WiFi, Online_boarding, Seat_comfort, Inflight_entertainment, etc.
- 5 one-hot encoded categoricals: Gender, Customer_Type, Type_of_Travel, Class

### Repository Structure

```
dimex/
├── airline-passenger-satisfaction/   # Raw and processed CSV files
├── dimex/                            # Python package
│   ├── preprocessing.py              # Cleaning, encoding, balancing
│   ├── xgb_runner.py                 # XGBoost training & feature selection
│   ├── split_runner.py               # SPLIT interface
│   └── reporting.py                  # Confusion matrix plotting
├── notebooks/                        # Three notebook runs (seeds 42, 50, 99)
├── results/Results.png               # Comparison chart
├── README.md / SETUP.md              # Documentation
└── environment.yml                   # Conda env (Python 3.10)
```

### Pipeline Workflow

1. **Clean** — drop rows with missing values
2. **Encode** — one-hot encode categoricals, binarize target label
3. **Split** — stratified 70/30 train/test, with 30% further split into validation/test
4. **Balance** — SMOTE on training data → 50/50 class distribution
5. **Baseline SPLIT** — train on all 23 features (lookahead=2, depth=5, λ=0.007)
6. **XGBoost feature selection** — train XGBoost (100 trees, max_depth=3), rank by cumulative gain, select top features at 80%/90%/95%/97.5%/99% thresholds → 9/13/16/18/19 features
7. **Refined SPLIT** — retrain on reduced feature sets (depth=6, λ=0.005)
8. **Evaluate** — compare train/val/test accuracy and model complexity across configurations

### Key Results (RANDOM_SEED=42)

| Model | Features | Leaves | Test Accuracy | Runtime |
|---|---|---|---|---|
| Baseline SPLIT | 23 | 6 | 88.87% | < 1s |
| Best XGBoost | 16 | 755 | 93.58% | — |
| **Best SPLIT** | **9** | **8** | **91.97%** | **~5s** |

Across all three seeds (42, 50, 99) the pattern held: SPLIT with XGBoost-selected features achieved ~91–92% accuracy with 6–10 leaves, versus XGBoost's ~93% with ~755 leaves.

**The 9 most important features** (selected by XGBoost 80% cumulative gain threshold):
1. Online_boarding
2. Type_of_Travel_Personal Travel
3. Class_Eco
4. Inflight_wifi_service
5. On_board_service
6. Customer_Type_disloyal Customer
7. Inflight_entertainment
8. Checkin_service
9. Leg_room_service

**Notable finding**: Passengers who rated Inflight WiFi as 0 showed near-universal satisfaction — likely because 0 means "no WiFi available" rather than "lowest quality," suggesting some passengers prefer no WiFi to poor WiFi.

### Key Takeaway

XGBoost achieves 93.58% accuracy with a 755-leaf ensemble (uninterpretable). SPLIT achieves 91.97% with an 8-leaf tree (fully interpretable) — only 1.6 percentage points lower accuracy at 94% fewer model components. The decision logic can be read directly from the tree.

### Dependencies

- Python 3.10, pandas, scikit-learn, xgboost, imbalanced-learn (SMOTE), matplotlib
- SPLIT library: https://github.com/VarunBabbar/SPLIT-ICML/ (must be installed separately)
- Install project: `pip install -e .` from repo root

### External Write-up

Medium article: "I built an end-to-end interpretable Machine Learning research pipeline" by Lucas Campagnaro (2025).
