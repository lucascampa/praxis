"""
Module for PRAXIS algorithm execution and evaluation.

This module provides:
- train_praxis: Train a PRAXIS model with specified hyperparameters, returning the fitted model,
  Rashomon set statistics, and training metadata.
- prediction_praxis: Generate predictions with a trained PRAXIS model and compute accuracy
  (using the best tree or ensemble voting).
- get_rashomon_stats: Extract statistics about the Rashomon set (size, min objective, etc.).
- binarized_features: Extract feature names after binarization.
"""

__author__ = "Lucas Campagnaro"

from praxis import PRAXIS, ThresholdGuessBinarizer
from sklearn.metrics import accuracy_score
import numpy as np
import time

def train_praxis(x_train, y_train,
                 lambda_reg=0.01, depth_budget=5, rashomon_mult=0.03,
                 lookahead_k=1, verbose=False, time_limit=None):
    """
    Train a PRAXIS model with specified hyperparameters and return the fitted model,
    Rashomon set statistics, and training metadata.

    Args:
        x_train (pd.DataFrame): Features for training (continuous or categorical).
        y_train (pd.Series): Binary labels (will be converted to 0/1).
        lambda_reg (float, optional): Regularization parameter. Defaults to 0.01.
        depth_budget (int, optional): Maximum tree depth. Defaults to 5.
        rashomon_mult (float, optional): Multiplicative slack for Rashomon set. Defaults to 0.03.
        lookahead_k (int, optional): Lookahead depth for proxy algorithm. Defaults to 1 (LicketySPLIT).
        verbose (bool, optional): Enable verbose output. Defaults to False.
        time_limit (int, optional): Maximum training time in seconds. Defaults to None.

    Returns:
        tuple:
            model (PRAXIS): The trained PRAXIS model instance.
            x_train_binary (np.ndarray): Binarized training features.
            model_data (dict): Metadata including:
                - "lambda_reg": regularization parameter
                - "depth_budget": maximum tree depth
                - "rashomon_mult": multiplicative slack
                - "lookahead_k": lookahead depth
                - "n_trees": number of trees in Rashomon set
                - "min_objective": objective value of best tree
                - "runtime": training duration in seconds
    """
    # Convert labels to 0/1 if needed
    y_train_encoded = (y_train == y_train.unique()[0]).astype(int)

    # Binarize features
    binarizer = ThresholdGuessBinarizer()
    x_train_binary = binarizer.fit_transform(x_train, y_train_encoded)

    # Train PRAXIS
    model = PRAXIS()
    start = time.perf_counter()

    model.fit(
        x_train_binary,
        y_train_encoded,
        lambda_reg=lambda_reg,
        depth_budget=depth_budget,
        rashomon_mult=rashomon_mult,
        lookahead_k=lookahead_k,
    )

    runtime = time.perf_counter() - start

    # Extract model statistics
    n_trees = model.count_trees()
    min_obj = model.get_min_objective()

    model_data = {
        "lambda_reg": lambda_reg,
        "depth_budget": depth_budget,
        "rashomon_mult": rashomon_mult,
        "lookahead_k": lookahead_k,
        "n_trees": n_trees,
        "min_objective": min_obj,
        "runtime": runtime,
    }

    return model, x_train_binary, model_data


def prediction_praxis(model, x_test, y_test, x_train_binary=None,
                      use_ensemble=False, tree_index=0):
    """
    Generate predictions with a trained PRAXIS model and compute accuracy.

    Args:
        model (PRAXIS): Trained PRAXIS model instance.
        x_test (pd.DataFrame): Test features (will be binarized).
        y_test (pd.Series): True labels for evaluation.
        x_train_binary (np.ndarray, optional): Binarized training features used for binarization context.
        use_ensemble (bool, optional): If True, ensemble all Rashomon trees via majority voting.
                                       If False, use best tree (index 0). Defaults to False.
        tree_index (int, optional): Which tree to use if use_ensemble=False. Defaults to 0 (best tree).

    Returns:
        tuple:
            y_pred (array-like): Predicted labels.
            accuracy (float): Proportion of correct predictions.
            pred_details (dict): Additional details:
                - "ensemble": whether ensemble was used
                - "n_trees_used": number of trees used
                - "agreement": if ensemble, proportion of unanimous votes
    """
    # Convert labels to 0/1 if needed
    y_test_encoded = (y_test == y_test.unique()[0]).astype(int)

    # Binarize test features (using same binarizer as training)
    binarizer = ThresholdGuessBinarizer()
    if x_train_binary is not None:
        x_test_binary = binarizer.fit_transform(x_test, y_test_encoded)
    else:
        x_test_binary = binarizer.fit_transform(x_test, y_test_encoded)

    if use_ensemble:
        # Get predictions from all trees
        all_preds = model.get_all_predictions(x_test_binary, stack=True)

        # Majority voting
        y_pred = (all_preds.mean(axis=0) > 0.5).astype(int)

        # Compute agreement (proportion of unanimous votes)
        agreement = np.mean(
            (all_preds.sum(axis=0) == 0) | (all_preds.sum(axis=0) == all_preds.shape[0])
        )

        pred_details = {
            "ensemble": True,
            "n_trees_used": all_preds.shape[0],
            "agreement": agreement,
        }
    else:
        # Use single best tree
        y_pred = model.get_predictions(tree_index, x_test_binary)

        pred_details = {
            "ensemble": False,
            "n_trees_used": 1,
            "agreement": None,
        }

    accuracy = accuracy_score(y_test_encoded, y_pred)

    return y_pred, accuracy, pred_details


def get_rashomon_stats(model):
    """
    Extract statistics about the Rashomon set.

    Args:
        model (PRAXIS): Trained PRAXIS model instance.

    Returns:
        dict:
            - "n_trees": total number of trees in Rashomon set
            - "min_objective": minimum objective value
            - "max_objective": maximum objective value
            - "root_histogram": histogram of objectives at root (list of tuples)
    """
    n_trees = model.count_trees()
    min_obj = model.get_min_objective()

    # Get root histogram to estimate objective range
    root_hist = model.get_root_histogram()
    max_obj = max([obj for obj, count in root_hist]) if root_hist else min_obj

    return {
        "n_trees": n_trees,
        "min_objective": min_obj,
        "max_objective": max_obj,
        "root_histogram": root_hist,
    }


def binarized_features(model, original_feature_names=None):
    """
    Extract feature names after binarization.

    Args:
        model (PRAXIS): Trained PRAXIS model instance.
        original_feature_names (list, optional): Original feature names before binarization.

    Returns:
        list: Names of binarized features (or generic names if not available).
    """
    # PRAXIS doesn't store feature names directly; we'd need to track the binarizer
    # For now, return generic names
    # In practice, users should track the binarizer from train_praxis()
    return [f"feature_{i}" for i in range(model.count_trees())]  # Placeholder
