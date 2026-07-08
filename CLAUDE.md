# PRAXIS Project Context

**Project Goal**: Compare two Rashomon set enumeration algorithms (**RESPLIT** and **PRAXIS**) on the airline passenger satisfaction dataset to evaluate scalability, set size, tree quality, and feature importance stability.

**Status (2026-07-07)**: The five-model comparison is done (see README results table and `results/results5.json`). Active work is Rashomon set *structure* analysis per the July 7 externship guidance: tree diagrams, split-variable drift, TGB parameter sweeps, RID.

---

## Context Documents — what to read, and when

| Document | Read when |
|---|---|
| [RESPLIT investigation.md](./RESPLIT investigation.md) | **Before touching RESPLIT in any way.** Full defect record: three verified bugs in the released code and the monkey-patches in `run_resplit.py`. |
| [papers/PAPERS.md](./papers/PAPERS.md) | Before writing or reasoning about the SPLIT/RESPLIT/PRAXIS algorithms, their complexity, or paper-claimed benchmarks. Summaries of both PDFs live there. |
| [data/DATA.md](./data/DATA.md) | Before touching the dataset: file inventory, feature lists (incl. the 9-feature subset), preprocessing lineage, usage rules. |
| [SPLIT context.md](./SPLIT context.md) | For DIMEX (old project) background: SPLIT-vs-XGBoost results, original externship framing. |
| https://github.com/zakk-h/PRAXIS | PRAXIS official repo/README (API surface, RID). |

Key RESPLIT conclusions (kept inline because they shape all comparisons): RESPLIT's
stump-level truncation is **load-bearing** — with it, RESPLIT finishes in ~1h but
provably misses the optimal tree (best 0.148839 vs true 0.136402); without it, exact
enumeration is intractable (projected days). PRAXIS does both in <1s. RESPLIT
**requires binarized input** (`data/airline-passenger-satisfaction/train_binarized.csv`
/ `test_binarized.csv`, exported by comparison.ipynb from the same
ThresholdGuessBinarizer PRAXIS uses) — it silently produces garbage on raw features.

---

## Project Structure

```
praxis/
├── CLAUDE.md                              # This file — operational context & traps
├── README.md                              # Project overview, workflow, results table
├── SETUP.md                               # Installation instructions
├── SPLIT context.md                       # DIMEX (old project) background
├── RESPLIT investigation.md               # RESPLIT defect record — read before touching RESPLIT
├── setup.py                               # Installs the dimex and prxs packages
├── environment.yml                        # Conda environment (Python 3.10 + PRAXIS)
│
├── papers/
│   ├── PAPERS.md                         # Summaries of both papers — read before algorithm work
│   ├── SPLIT.pdf                         # Babbar et al., ICML 2025 (pages 1-8)
│   └── PRAXIS.pdf                        # Harary et al., ICML 2026 (pages 1-9)
│
├── data/
│   ├── DATA.md                           # Dataset docs — read before touching the data
│   └── airline-passenger-satisfaction/   # raw / clean / encoded / balanced / binarized csvs
│
├── dimex/                                 # DIMEX package (DIMACS Externship: SPLIT vs XGBoost — old project)
├── prxs/                                  # Current-project package: Rashomon set structure analysis
│   └── rashomon_trees.py                 # Export/load PRAXIS trees, rebuild from paths, draw diagrams
│
├── run_resplit.py                         # RESPLIT runner (command-line only; embeds bug workarounds)
├── verify_bound.py, verify_leak.py        # Standalone repros of RESPLIT defects
│
├── notebooks/
│   ├── comparison.ipynb                  # XGBoost/SPLIT/RESPLIT/PRAXIS/STreeD comparison (maintained)
│   ├── tree_structure.ipynb              # Rashomon set diagrams & structure analysis (July 7 guidance)
│   └── Black-Box to Glass-Box modeling [RANDOM_SEED=42|50|99].ipynb  # DIMEX (old project)
│
└── results/
    ├── results1..5.json                  # Cumulative model-comparison tables (built by comparison.ipynb)
    ├── praxis_trees.json                 # Full PRAXIS Rashomon set export (prxs schema, 140 trees)
    └── resplit_models_*.pkl              # Cached fitted RESPLIT models (fit costs ~1h)
```

## Key Files & Their Roles

| File | Purpose | Key Functions |
|------|---------|---|
| `dimex/preprocessing.py` | Data cleaning & encoding | `clean_missing()`, `binarize_encode()`, `smote()` |
| `dimex/xgb_runner.py` | XGBoost training & feature selection | `train_xgb()`, `cumulative_gain()` |
| `dimex/split_runner.py` | SPLIT algorithm interface | `train_split()`, `prediction_split()`, `binarized_features()` |
| `dimex/reporting.py` | Visualization | `cm()` (confusion matrices) |
| `prxs/rashomon_trees.py` | Rashomon set tree export & diagrams | `export_praxis_trees()`, `load_trees()`, `build_tree()`, `draw_tree()`, `sample_tree_indices()` |
| `run_resplit.py` | RESPLIT execution (command-line only) | Embeds all RESPLIT workarounds — never rewrite from scratch |
| `notebooks/comparison.ipynb` | Five-model comparison + Rashomon set export | XGBoost / SPLIT / RESPLIT / PRAXIS / STreeD |
| `notebooks/tree_structure.ipynb` | Rashomon set structure analysis | Diagrams, split-usage tables, hand-written `feature_labels` |

⚠️ **`dimex` preprocessing functions write csvs as a side effect** (`clean_missing`,
`binarize_encode`, `smote` all save into `data/airline-passenger-satisfaction/`). SMOTE
output is **not byte-reproducible across library versions** — never regenerate the
csvs outside the WSL `praxis-env`; if they get clobbered, restore with
`git checkout -- data/airline-passenger-satisfaction/`.

---

## How to Run

### Setup (WSL/Linux)
```bash
conda env create -f environment.yml && conda activate praxis-env
# SPLIT + RESPLIT come from one repo (C++ deps: cmake ninja tbb tbb-devel pkg-config gmp via conda-forge)
git clone https://github.com/VarunBabbar/SPLIT-ICML.git
pip install SPLIT-ICML/resplit/ SPLIT-ICML/split/
pip install -e .        # installs dimex + prxs
```
Verify: `python -c "from resplit import RESPLIT; from split import SPLIT; from praxis import PRAXIS; import dimex, prxs; print('ok')"`

`tree-praxis` also installs on **Windows** (pip) — handy for quick local tests without
WSL (SPLIT/RESPLIT do not).

### Part 1: XGBoost + SPLIT (Jupyter)

Maintained in [notebooks/comparison.ipynb](./notebooks/comparison.ipynb) Part 1.
Trains on the 9-feature subset (list in [data/DATA.md](./data/DATA.md)) from
`train_clean_encoded_balanced.csv`, evaluates on `test_clean_encoded.csv`;
SPLIT config: `lookahead=2, full_depth=6, reg=0.005`.

### Part 2: RESPLIT (Command-line Script Only)

⚠️ **Do not write a fresh RESPLIT script from scratch — use the existing
[run_resplit.py](./run_resplit.py).** It embeds hard-won fixes documented in
[RESPLIT investigation.md](./RESPLIT investigation.md):
- Loads **binarized** input (exported by the comparison.ipynb binarized-export cell)
- Monkey-patches `hash_for_indexing` (OOM on large sets), adds a float-tie tolerance
  to stump bounds, and pins `cart_lookahead_depth` on stump calls (C++ config leak)
- Extracts the exact best tree from the compact prefix-trie representation
  (per-stump minimization) instead of materializing ~10⁹ trees
- Caches the fitted model to `results/resplit_models_*.pkl` (fit costs ~1h)

```bash
python run_resplit.py 2>&1 | tee logs/resplit_<runname>.txt
```
Saves unified-schema results to `results/results2.json`, which the comparison.ipynb
Part-2 cell merges into the running results table.

### Part 3: PRAXIS + full comparison (Jupyter)

Maintained in [notebooks/comparison.ipynb](./notebooks/comparison.ipynb) Part 3 onward:
PRAXIS training, binarized-data export, Rashomon set export, tree printing, and the
five-model table. **Key API facts learned the hard way:**

- PRAXIS hyperparameters go into **`fit()`**, not the constructor:
  `PRAXIS().fit(x_bin, y, lambda_reg=0.005, depth_budget=5, rashomon_mult=0.01, lookahead_k=1)`
- PRAXIS requires **binary input** — `ThresholdGuessBinarizer().fit_transform(x, y)`
- `get_tree_objective(i)` returns a **tuple** `(N × regularized_loss, regularized_loss)`
  — e.g. `(27, 0.09)`. Verified 2026-07-07: `objective[1]` equals the externally
  computed `(preds != y_train).sum()/N + λ·num_leaves` exactly. Sorting by the tuple
  works (lexicographic); never call `int()` on it
- `get_tree_paths(i)` returns signed ids **±(column_index+1)**; decode with
  `names[abs(s)-1]`, `s > 0` = condition true
- All models' table losses are computed externally from raw predictions on TRAIN
  (same formula for every model), never taken from a model's internal objective
- **SPLIT tree printouts use per-model encoder indices** — `feature: 4` in one model
  and `feature: 0` in another can be the same feature. Always decode through that
  model's own `dx.binarized_features(model)`. SPLIT's printed leaf losses are
  per-leaf error rates in float32 (misclassified/leaf_size), not global-N losses

### Part 4: Rashomon set structure analysis (July 7 guidance)

Two-stage workflow, decoupled so the analysis never re-fits PRAXIS:

1. **Export** (comparison.ipynb, the cell right after the binarized-data export):
   `prxs.export_praxis_trees()` writes `results/praxis_trees.json` — all trees ranked
   by objective, with signed-id paths, leaf predictions, per-tree regularized loss,
   leaves/depth, and split features.
2. **Analyze** ([notebooks/tree_structure.ipynb](./notebooks/tree_structure.ipynb)):
   loads the JSON via `prxs.load_trees()`; shows the feature-index→name mapping,
   structure summary, split-variable usage and root-split tallies, and diagrams
   beginning/middle/end trees with `prxs.draw_tree()`.

`draw_tree` conventions: internal nodes = binarized feature, edges = yes/no
(yes ⇔ condition true), leaves colored blue=satisfied / yellow=not satisfied
(CVD-safe pair) with the class written as text. Human-readable node labels come
from the **hand-written** `feature_labels` dict in tree_structure.ipynb — when
editing it, phrase labels so "yes" still means the raw `<=` condition is TRUE
(e.g. `Customer_Type_disloyal Customer <= 0.5` → "Loyal customer?").

---

## Hyperparameters

### RESPLIT (`from resplit import RESPLIT`; command-line scripts only)
```python
config = {
    "regularization": 0.005,            # λ
    "rashomon_bound_multiplier": 0.01,  # slack: models within (1+ε)L*
    "depth_budget": 5,
    "cart_lookahead_depth": 3,
    "verbose": False,
}
model = RESPLIT(config, fill_tree="treefarms")   # also: "optimal", "greedy"
```
Alternatives to `rashomon_bound_multiplier`: `rashomon_bound_adder` (absolute L*+ε),
`rashomon_bound` (fixed cutoff).

### SPLIT (`dimex.train_split`)
`lookahead=2`, `full_depth=6`, `reg=0.005` (project standard).

### PRAXIS (`PRAXIS().fit(...)`)
`lambda_reg=0.005`, `depth_budget=5`, `rashomon_mult=0.01`, `lookahead_k=1`
(project standard; paper defaults are `lambda_reg=0.01`, `rashomon_mult=0.03`).

Algorithm background, complexity, and paper-claimed benchmarks: see
[papers/PAPERS.md](./papers/PAPERS.md). Actual measured results: README table
(from `results/results5.json`).

---

## Notes for Future Work

### ✅ Completed (DIMEX Work)
- [x] **Data Preprocessing**: Missing values, encoding, SMOTE balancing
- [x] **Feature Selection via XGBoost**: Identified 9 optimal features at 80% cumulative gain
- [x] **SPLIT vs XGBoost Comparison**: Across 3 random seeds (42, 50, 99) in Black-Box notebooks

### ✅ Completed (RESPLIT vs PRAXIS Comparison)
- [x] **RESPLIT script**: `run_resplit.py` (command-line only; see RESPLIT investigation.md for the three upstream defects it works around)
- [x] **Five-model comparison**: XGBoost / SPLIT / RESPLIT / PRAXIS / STreeD in comparison.ipynb → `results/results5.json`
- [x] **Rashomon set export**: `prxs.export_praxis_trees()` → `results/praxis_trees.json` (140 trees)
- [x] **Structure analysis scaffolding**: `notebooks/tree_structure.ipynb` (mapping, summary, usage, root tallies, diagrams)
- [x] **Human-readable labels**: `feature_labels` dict hand-translated (2026-07-07)

### Remaining Work (July 7 guidance + open items)
- [ ] **Takeaways**: Fill in the takeaways cell of tree_structure.ipynb after inspecting the sampled diagrams
- [ ] **TGB vs RID importance ranking**: Table/diagram comparing ThresholdGuessBinarizer variable ranking vs Rashomon median importance; track RID memory & runtime
- [ ] **TGB parameter sweeps**: Repeat with different TGB depth/iteration settings; is the small-setting variable set contained in the large-setting one?
- [ ] **Leaf-partition distributions**: Per-tree data distribution over leaves; compare across the set and across TGB/PRAXIS parameters
- [ ] **Config tuning**: Test `rashomon_mult` values (0.01, 0.03, 0.05, 0.1)
- [ ] **Multi-seed robustness**: Test across multiple random seeds (42, 50, 99)
- [ ] **Scaling experiment**: Test on 16 or 19 features to identify PRAXIS advantage threshold

---

## Findings Log

### 2026-07-07: prxs package, tree diagrams, and the "wifi root" resolution
- **New `prxs` package** (separate from `dimex`, the old DIMACS Externship project):
  export/load/reconstruct/diagram PRAXIS Rashomon set trees
- **New `notebooks/tree_structure.ipynb`**: July 7 guidance analysis — run the export
  cell in comparison.ipynb first, then this notebook (no PRAXIS re-fit needed)
- **Corrected API fact**: PRAXIS `get_tree_objective(i)` returns a *tuple*
  `(N × loss, loss)`, not an integer; `objective[1]` is the exact normalized
  regularized loss

**First structural results (rashomon_mult=0.01, 140 trees)**: loss 0.136402–0.143071,
7–10 leaves, depth uniformly 5. Five splits are universal (in all 140 trees):
`Inflight_wifi_service <= 0.5 / 4.5`, `Type_of_Travel_Personal Travel <= 0.5`,
`Customer_Type_disloyal Customer <= 0.5`, `Inflight_entertainment <= 3.5`.
**Every tree roots at `Inflight_wifi_service <= 0.5 → satisfied`** — a rating of 0
means "not rated / N/A" (not "terrible"), and that split isolates 3,887 rows at
99.8% purity (8 errors), so no near-optimal tree can give it up. Diversity lives in
the lower levels.

**Resolved (index-numbering illusion)**: the old DIMEX SPLIT trees seemed to root
elsewhere, but SPLIT printouts use per-model encoder indices — `feature: 0` in one
model and `feature: 4` in another are both wifi ≤ 0.5. Confirmed three ways:
(a) current split_model's `binarized_features` map decodes its root to wifi ≤ 0.5,
(b) the current split_tree printout is character-identical to old model_8's,
(c) old tree_0's root-true leaf loss `0.002058142563328147` = `float32(8/3887)`,
uniquely the wifi==0 subset.

**⚠️ SMOTE artifact in tree leaves ("disloyal → satisfied")**: several Rashomon trees
(e.g. ranks 1–2) contain leaves predicting *disloyal* customers as satisfied,
which contradicts domain intuition. The pattern is real in the **balanced** training
data (personal travel & disloyal & wifi rated: 449 rows, 69% satisfied) but does
**not exist organically**: pre-SMOTE that cell is 163 rows at 15.3% satisfied
(wifi 1–4 subcell: 148 rows, 6.8%; only the tiny wifi=5 subcell, 11 rows, is 100%
satisfied). SMOTE tripled the sparse cell and flipped its majority class. Rule of
thumb: **sanity-check any small-subset leaf against `train_clean_encoded.csv`
(pre-SMOTE)** — sparse regions distort most. The wifi==0 root is organic (99.7%
satisfied pre-SMOTE, n=3,096).

**Retrain on organic data (tree_structure.ipynb, new section; preview numbers from a
local dry run)**: same hyperparameters on pre-SMOTE `train_clean_encoded.csv` →
**21,087 trees** at the same 1% slack (vs 140 balanced; the organic loss landscape is
far flatter), loss 0.1214–0.1330, 6–11 leaves. Four universal splits survive
(wifi 0.5, travel type, loyalty, entertainment 3.5); root is wifi in 100% of trees
but splits between thresholds (~71% `<= 0.5`, ~29% `<= 4.5`). **Key framing**: the
wrong-cell "personal & disloyal → satisfied" leaf sits at ranks 1–2 of the SMOTE set
but first appears ~rank 250/21k organically — a Rashomon set's *tail* always contains
cheap wrong-small-cell flips (that's what slack is); SMOTE's damage was promoting a
tail pattern to the top of the set. Judge a pattern by where in the set it appears,
not just whether it appears.

### 2026-07-03 (see RESPLIT investigation.md)
SPLIT/PRAXIS/STreeD find the **identical** optimal tree — earlier apparent structural
diversity was a notebook printer bug (signed-id decoding).

### 2026-06-30
Paper analysis integrated; summaries now maintained in [papers/PAPERS.md](./papers/PAPERS.md).

---

## Environment

WSL2 (Ubuntu) + conda `praxis-env`, Python 3.10.12 (installed 2026-06-29):
RESPLIT 0.2.4, SPLIT 0.1.0, PRAXIS 0.0.29, dimex + prxs (editable).
`tree-praxis` also runs on Windows for quick local tests.

---

**Last updated**: 2026-07-07
**Project owner**: Lucas Campagnaro