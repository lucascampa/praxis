"""
Package for PRAXIS Rashomon-set analysis (current externship project).

Distinct from `dimex` (the DIMACS Externship package for the earlier
SPLIT-vs-XGBoost comparison work). This package covers structural analysis of
the Rashomon set: exporting trees from a fitted PRAXIS model, reconstructing
them from their path representation, and rendering tree diagrams.
"""

__author__ = "Lucas Campagnaro"

from .rashomon_trees import sample_tree_indices, export_praxis_trees, load_trees, build_tree, draw_tree

# Public API: expose core functions for external use
__all__ = ["sample_tree_indices", "export_praxis_trees", "load_trees", "build_tree", "draw_tree"]
