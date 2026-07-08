# RESPLIT Investigation: Why It Underperforms on the Airline Dataset

**Dates**: 2026-07-01 → 2026-07-03 (ongoing)
**Context**: In the DIMEX/PRAXIS comparison, SPLIT, PRAXIS, and STreeD all independently
converge to the same optimal tree (train loss **0.136402**, 8 leaves, identical
predictions on train and test). RESPLIT — which should enumerate the Rashomon set
around that same optimum — does not reach it. This file records every hypothesis,
test, and result, in order.

**Reference numbers** (loss = train error rate + 0.005 × leaves):

| Model | Train Loss | Test Acc | Size | Runtime | Rashomon Set |
|---|---|---|---|---|---|
| XGBoost | 0.082193 | 92.9% | 741 leaves | 2.3s | — |
| SPLIT | 0.136402 | 91.9% | 8 leaves | ~6s | — |
| STreeD | 0.136402 | 91.9% | 8 leaves | ~160s | — |
| PRAXIS | 0.136402 | 91.9% | 8 leaves | 0.73s | **140 trees** |
| RESPLIT (best so far) | **0.148839** | 88.9% | 6 leaves | ~57min | 1.46B trees* |

*None of RESPLIT's ~1.5B trees fall inside the true Rashomon band
[0.136402, 0.137766] (= (1+0.01)·L*). The set is huge **and** entirely out-of-band.

---

## Timeline of hypotheses

### H1: Lookahead depth mismatch — REJECTED
`cart_lookahead_depth=3` vs SPLIT's `lookahead=2`. Changed 3→2: loss got **worse**
(0.5023 vs 0.3316). Reverted.

### H2: TREEFARMS leaf-filling at fault — REJECTED
Swapped `fill_tree="treefarms"` → `"optimal"`: same bad result. The corruption was
upstream of the fill strategy.

### H3: Stale package version — REJECTED
Merged upstream `VarunBabbar/SPLIT-ICML`, reinstalled `resplit/`. Identical results.

### H4: Non-binary input — **CONFIRMED** (root cause #1)
RESPLIT's Python layer assumes binary {0,1} features: `dict_to_tree` partitions with
`X.iloc[:, f] == 1`, `_predict_sample` branches on truthiness. We fed raw 0–5 ratings.
PRAXIS refuses non-binary input with a ValueError; RESPLIT silently produces garbage
(prefix tree train loss 0.5146 ≈ coin flip). Both papers binarize as preprocessing via
threshold guessing (SPLIT paper App. A.8 lists RESPLIT among algorithms *requiring*
binarization).

**Fix**: notebook exports the exact `ThresholdGuessBinarizer` output PRAXIS uses →
`data/airline-passenger-satisfaction/train_binarized.csv` / `test_binarized.csv`;
`run_resplit.py` loads those. Result: 1 prefix → **861 prefixes**, 6 trees → **1.65B
trees**. Best tree improved 0.3316 → 0.148839. Still not 0.136402.

### Engineering detour: OOM at materialization
`fit()` ends by reconstructing all trees into a dict (`hash_for_indexing`) — OOM-killed
at 1.65B trees (7.6 GB WSL). **Patch** (in `run_resplit.py`): no-op the materialization;
set size comes from `num_models`; the exact best tree is recovered from the compact
prefix-trie form by per-stump minimization — stump losses are independent (disjoint
data partitions), so the global min is the sum of per-stump minima: cost is the *sum*
of menu sizes, not their *product*. Fitted model cached to pickle so extraction tweaks
don't repay the ~1h fit.

### H5: Depth semantics mismatch — REJECTED (as the binding constraint)
GOSDT-family `depth_budget` counts node levels (budget 5 = 4 splits); PRAXIS/STreeD
count splits (5 = 5 splits); SPLIT ran at `full_depth=6`. RESPLIT at depth_budget=5
caps paths at 4 splits while the optimal tree needs 5. Bumped RESPLIT to 6 (stump
fills: `remaining_depth = 6−3+1 = 4`): per-prefix best losses **bit-identical**,
best still 0.148839. Depth was a real inconsistency (now fixed for fairness) but not
what blocked the optimum.

### H6: Bound tie-loss at the Python→C++ boundary — **CONFIRMED** (root cause #2)
Each stump's menu is bounded by its greedy fill's loss. `verify_bound.py` reproduced
two stumps from the logs exactly (bounds 0.15697 / 0.159347 to 6 decimals):

- **Dead stump** (62,153 rows, 0 trees in log): greedy fill **is** the GOSDT-optimal
  subtree (same 3 leaves, 8,143 errors). Bound == optimum exactly → C++ returns **0
  trees** (strict comparison / float rounding across the boundary). Bound × 1.01 → 2
  trees. So stumps die precisely when greedy plays perfectly; each dead stump silently
  kills its whole prefix.
- Conceptually no epsilon should be needed: "loss ≤ greedy's loss" always contains at
  least the greedy tree itself. This is a silent floating-point boundary defect.

**Patch**: stump bounds inflated by ×(1+1e-4)+1e-6. Dead prefixes 205 → 121, set →
1.46B. Best **unchanged** at 0.148839.

- Side discovery: `train_greedy`'s internal split criterion mixes units (errors ÷ full
  n, but regularization scaled to the partition) — documented, but the bound itself is
  computed in correct local units.

### H7: C++ global-config leakage — **CONFIRMED** (root cause #3)
`verify_bound.py` anomaly: healthy stump gave **24 trees standalone vs 5 in-pipeline**,
identical config. Log's stump-level config dumps show `cart_lookahead_depth: 3` —
a key the stump code never passes. The C++ `configure()` **overlays** onto its previous
global state instead of resetting, so stump calls inherit the top-level lookahead
settings and enumerate truncated menus instead of exact ones — the best local subtrees
never enter the menus at all.

`verify_leak.py` (4 calls, same stump, same bound):

| lookahead_depth | look_ahead | trees |
|---|---|---|
| 3 | True | 5 (truncated) |
| **4 (= depth_budget)** | True | **24 (exact)** |
| 3 | False | 5 → the boolean is a **no-op** |
| *(none passed)* | *(none)* | 5 → leak reproduced in-process |

**Patch**: stump calls set `cart_lookahead_depth = depth_budget` (the integer is the
gate; equal to search depth = no truncation).

**Outcome — INTRACTABLE, and that is the finding.** With exact stump enumeration
(`logs/resplit_depth6_exactstumps.txt`): 1% progress after ~77 min of CPU; single
stumps produced model sets of 16,835 nodes (vs ~18 truncated); projected multi-day
runtime with likely OOM. Run aborted. Conclusion: **the truncation is load-bearing**
— RESPLIT on this dataset is either fast (truncated stump menus, ~1h, provably
misses the optimum: best 0.148839, zero trees in the true band) or exact
(complete menus, computationally intractable). It cannot be both. PRAXIS is both
in 0.73s / 6 MB.

---

## Current state of `run_resplit.py`

Three monkey-patches, applied without touching the installed library:
1. `hash_for_indexing` → no-op (prevents OOM; compact-form extraction instead)
2. stump `rashomon_bound` ×(1+1e-4)+1e-6 (float-tie tolerance)
3. stump `cart_lookahead_depth = depth_budget` (stops config leak / truncation)

Caches per configuration: `results/resplit_models_*.pkl`. Logs in `logs/`.

## Open questions

1. ~~Does the exact-stumps run reach 0.136402?~~ Aborted as intractable (see H7
   outcome). Cheap remaining check, not yet run: verify from the cached pickle that
   a prefix matching the optimal tree's top splits (wifi≤0.5, travel) exists among
   the 861, and GOSDT-solve just its stumps — loss decomposes over stumps, so their
   optima summing to 0.136402 would prove exact menus *would* contain the optimum.
2. Set-size comparability: RESPLIT's ~1.5B counts everything under loose greedy-anchored
   *local* bounds (a superset, by design); PRAXIS's 140 is the exact *global* ε-band.
   These numbers must not be compared as if they measure the same thing.
3. The three defects (silent non-binary acceptance, boundary tie-loss, config leakage)
   are all in the released code, all verified by minimal standalone reproductions —
   worth reporting upstream.

## Postscript: bugs in our own tooling (for honesty and future reference)

- **"Three different optimal trees" was an artifact.** Our notebook tree-printer
  (comparison.ipynb cell-12) had two bugs: SPLIT's paths were printed with YES/NO
  inverted (SPLIT's *left* child = condition TRUE), and PRAXIS feature labels were
  shifted by one. **PRAXIS `get_tree_paths` API convention**: splits are signed ints
  ±(column_index + 1) — the +1 exists because ±0 can't carry a sign; decode with
  `names[abs(s) - 1]`, `s > 0` = condition true. After fixing, SPLIT, PRAXIS, and
  STreeD found the **identical** 8-leaf tree (root wifi≤0.5, travel, wifi≤3.5,
  ent≤3.5, boarding≤4.5, disloyal, wifi≤4.5) — expected, since three exact solvers
  minimizing the same objective must agree when the optimum is unique. The 100%
  prediction agreement was always computed from live models and was always real.
- **Environment breakage (2026-07-03)**: `pip install --force-reinstall resplit/`
  (H3 test) also reinstalled dependencies, silently bumping scikit-learn to 1.7.2
  and stranding imbalanced-learn 0.13.0 (`_safe_tags` ImportError on `import dimex`).
  Fixed by upgrading imbalanced-learn to 0.14.2. Lesson: use `--no-deps` when
  force-reinstalling local packages.

## Verification scripts

- `verify_bound.py` — reproduces two stumps' bounds/counts from the logs; GOSDT
  optimum comparison; corrected-bound revival. (~3 min)
- `verify_leak.py` — 4-case lookahead/leak experiment on one stump. (~2 min)

Both run standalone against the binarized CSVs; neither needs a RESPLIT refit.
