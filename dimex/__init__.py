"""
Package for PRAXIS research project: preprocessing, model runners, and reporting.

This module aggregates:
- Data preprocessing functions (cleaning, encoding, balancing, splitting).
- Model training and evaluation routines for XGBoost, SPLIT, and PRAXIS algorithms.
- Reporting tools for visualization of model performance.
"""

__author__ = "Lucas Campagnaro"

from .preprocessing import midrange_undersample, smote, clean_missing, split_dataset, balance_stats, binarize_labels, str_to_num, binarize_encode
from .xgb_runner import train_xgb, sort_by_gain, cumulative_gain, size, feature_importance, prediction_xgb
from .split_runner import train_split, n_leaves, prediction_split, binarized_features
from .praxis_runner import train_praxis, prediction_praxis, get_rashomon_stats
from .reporting import cm

# Public API: expose core functions for external use
__all__ = ["midrange_undersample", "smote", "clean_missing", "split_dataset", "balance_stats", "binarize_labels", "str_to_num", "binarize_encode",
           "train_xgb", "sort_by_gain", "cumulative_gain", "size", "feature_importance", "prediction_xgb",
           "train_split", "n_leaves", "prediction_split", "binarized_features",
           "train_praxis", "prediction_praxis", "get_rashomon_stats",
           "cm"]