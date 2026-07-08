# Setup Complete ✓

The `praxis` project has been fully configured and is ready to run. Below is a checklist of everything that's been prepared.

## Project Structure

### Documentation
- ✅ **CLAUDE.md** — Master project context (read this first!)
- ✅ **SPLIT context.md** — SPLIT paper + DIMEX background
- ✅ **README.md** — Project overview, quick start, expected results
- ✅ **SETUP.md** — Installation instructions for SPLIT and PRAXIS
- ✅ **SETUP_COMPLETE.md** — This file

### Code Package (`dimex/`)
- ✅ **__init__.py** — Updated to export PRAXIS functions
- ✅ **preprocessing.py** — Data cleaning, encoding, balancing (unchanged)
- ✅ **xgb_runner.py** — XGBoost training & feature selection (unchanged)
- ✅ **split_runner.py** — SPLIT training & evaluation (unchanged)
- ✅ **praxis_runner.py** — **NEW** PRAXIS training & evaluation wrapper
- ✅ **reporting.py** — Confusion matrices & plots (unchanged)

### Configuration
- ✅ **setup.py** — Updated package name to `praxis`
- ✅ **environment.yml** — Updated with Python 3.10, PRAXIS dependencies
- ✅ **.gitignore** — Project gitignore (unchanged)

### Notebooks
- ✅ **notebooks/Black-Box to Glass-Box modeling [RANDOM_SEED=42].ipynb** — DIMEX baseline (unchanged)
- ✅ **notebooks/Black-Box to Glass-Box modeling [RANDOM_SEED=50].ipynb** — DIMEX variant (unchanged)
- ✅ **notebooks/Black-Box to Glass-Box modeling [RANDOM_SEED=99].ipynb** — DIMEX variant (unchanged)
- ✅ **notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb** — **NEW** Main comparison notebook

### Data
- ✅ **data/airline-passenger-satisfaction/** — Dataset shared with DIMEX (unchanged)

### Results
- ✅ **results/Results.png** — DIMEX comparison chart (from original work)

---

## What's New in This Project

### 1. PRAXIS Integration
- `dimex/praxis_runner.py`: Wrapper around PRAXIS API with functions matching `split_runner.py` pattern
  - `train_praxis()` — Train Rashomon set
  - `prediction_praxis()` — Generate predictions (single tree or ensemble)
  - `get_rashomon_stats()` — Extract Rashomon set statistics

### 2. Comparison Notebook
- `notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb`: Full pipeline comparing RESPLIT vs PRAXIS
  1. Data preparation (cleaning, encoding, balancing)
  2. Feature selection via XGBoost (top 9 features at 80% cumulative gain)
  3. Train SPLIT (single optimal tree)
  4. Train PRAXIS (Rashomon set)
  5. Evaluate both on test set
  6. Create comparison tables and visualizations
  7. Summarize findings

### 3. Updated Documentation
- **CLAUDE.md**: Master documentation tying everything together
- **README.md**: Updated with PRAXIS goals and quick-start code
- **SETUP.md**: Step-by-step installation for both SPLIT and PRAXIS

---

## Quick Start

### 1. Install Environment
```bash
conda env create -f environment.yml
conda activate praxis-env
```

### 2. Install Dependencies (in WSL)
```bash
pip install --upgrade pip
git clone https://github.com/VarunBabbar/SPLIT-ICML.git
cd SPLIT-ICML && pip install resplit/ split/ && cd ..
pip install -e .  # Install dimex package
```

### 3. Verify Installation
```bash
python -c "from resplit import RESPLIT; print('✓ RESPLIT')"
python -c "from split import SPLIT; print('✓ SPLIT')"
python -c "from praxis import PRAXIS; print('✓ PRAXIS')"
python -c "import dimex; print('✓ dimex')"
```

### ⚠️ Important: RESPLIT Execution Constraint
**RESPLIT must be run via command-line scripts (Windows cmd), NOT Jupyter notebooks.**

The SPLIT-ICML README specifically notes:
- Run RESPLIT via `python3 script.py` or SLURM, not in Jupyter
- Jupyter has timeout issues with RESPLIT
- Use the notebook (`RESPLIT_vs_PRAXIS_comparison.ipynb`) only for PRAXIS comparisons

Example: Create `run_resplit.py` and execute via:
```bash
# From Windows Command Prompt (cmd.exe)
cd C:\path\to\praxis
python run_resplit.py
```

### 4. Run the Comparison
```bash
jupyter notebook notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb
```

---

## File Map

| File | Purpose |
|------|---------|
| CLAUDE.md | **START HERE** — Master context, all references, full workflow |
| SPLIT context.md | Deep dive into SPLIT paper & original DIMEX work |
| README.md | Quick project overview & quick-start code |
| SETUP.md | Step-by-step installation instructions |
| dimex/praxis_runner.py | PRAXIS API wrapper (NEW) |
| notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb | Main analysis (NEW) |

---

## What Each Algorithm Does

### SPLIT (RESPLIT variant)
- Single optimal sparse decision tree
- Branch-and-bound with lookahead cutoff
- Post-processing with GOSDT
- Output: 1 tree with ~8 leaves
- Runtime: ~5 seconds

### PRAXIS
- Enumerates Rashomon set (multiple near-optimal trees)
- Proxy-based algorithm (LicketySPLIT lookahead)
- Iterative budget refinement
- Output: Multiple trees within multiplicative slack
- Runtime: Comparable to SPLIT
- Built-in RID for feature importance

---

## Expected Results

**From original DIMEX work:**
- SPLIT: 91.97% accuracy, 8 leaves
- XGBoost: 93.58% accuracy, 755 leaves
- **Tradeoff**: 1.6% accuracy loss → 94% fewer leaves

**PRAXIS additions:**
- Rashomon set with multiple near-optimal trees
- Ensemble voting can improve accuracy
- Feature importance stability analysis via RID
- Per-sample disagreement quantification

---

## Next Steps

Once you open this folder and run the notebook, you can:

1. **Analyze Rashomon set** — How large is it? How do trees differ?
2. **Compute RID** — Use PRAXIS's `compute_rid()` for feature importance
3. **Study disagreement** — Analyze per-sample prediction variance
4. **Tune hyperparameters** — Sweep rashomon_mult, lookahead_k
5. **Compare scalability** — Test on larger feature sets
6. **Visualize trees** — Use PRAXIS's `plot_tree()` method

---

## References

- **SPLIT Paper** (Babbar et al., 2025): arXiv:2502.15988
- **PRAXIS Paper** (Harary et al., 2026): arXiv:2606.00202
- **PRAXIS Repository**: https://github.com/zakk-h/PRAXIS
- **SPLIT Repository**: https://github.com/VarunBabbar/SPLIT-ICML
- **Original DIMEX Write-up**: https://medium.com/@lucascampagnaro/i-built-an-end-to-end-interpretable-machine-learning-research-pipeline-0ba67d0ba700

---

## All Set!

Everything is ready. Open the notebook and start comparing!

Questions? Check **CLAUDE.md** for full project context.
