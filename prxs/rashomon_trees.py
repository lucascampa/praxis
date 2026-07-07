"""
Module for exporting, reconstructing, and diagramming trees from a PRAXIS Rashomon set.

This module provides:
- sample_tree_indices: Pick tree indices from the beginning, middle, and end of the
  Rashomon set (ranked by objective), per the July 7 externship guidance.
- export_praxis_trees: Serialize selected trees (paths, predictions, objective, loss,
  structure stats) to a JSON file, so analysis notebooks never need the fitted model.
- load_trees: Read an exported JSON file back.
- build_tree: Reconstruct a nested node structure from PRAXIS's flat signed-id paths.
- draw_tree: Render a tree record as a matplotlib diagram (feature-labeled internal
  nodes, yes/no edges, class-colored and class-labeled leaves).

PRAXIS path convention (see RESPLIT investigation.md): get_tree_paths(i) returns one
signed-id sequence per leaf; each id s maps to column abs(s)-1 of the binarized input,
and s > 0 means the split condition is TRUE on that branch.
"""

__author__ = "Lucas Campagnaro"

import json
import matplotlib.pyplot as plt

# CVD-safe pair (blue/yellow, ΔE 123.7 protan) — class is also written as text in the
# leaf, so color is never the only encoding
LEAF_STYLE = {
    0: {"face": "#eda100", "text": "#1a1a19"},  # not satisfied
    1: {"face": "#2a78d6", "text": "#ffffff"},  # satisfied
}
NODE_FACE = "#f0efeb"
NODE_EDGE = "#8a887d"
NODE_TEXT = "#1a1a19"


def sample_tree_indices(model, n_each=5):
    """
    Select tree indices from the beginning, middle, and end of the Rashomon set,
    ranked by increasing objective.

    Args:
        model (PRAXIS): Fitted PRAXIS model.
        n_each (int, optional): Trees to take from each of the three regions.
            Defaults to 5.

    Returns:
        list[tuple]: (rank, tree_index) pairs, deduplicated, in rank order.
            rank is the position after sorting all trees by objective (0 = best).
    """
    n = model.count_trees()
    order = sorted(range(n), key=model.get_tree_objective)
    mid_start = max(0, (n - n_each) // 2)
    ranks = list(range(min(n_each, n)))
    ranks += list(range(mid_start, min(mid_start + n_each, n)))
    ranks += list(range(max(0, n - n_each), n))
    seen = set()
    picks = []
    for r in ranks:
        if r not in seen:
            seen.add(r)
            picks.append((r, order[r]))
    return picks


def export_praxis_trees(model, feature_names, out_path,
                        indices=None, x_train=None, y_train=None, lambda_reg=None):
    """
    Serialize selected Rashomon-set trees to JSON for downstream analysis.

    Args:
        model (PRAXIS): Fitted PRAXIS model.
        feature_names (list[str]): Binarized feature names (from
            binarizer.get_feature_names_out()), indexed by column.
        out_path (str): Destination JSON file path.
        indices (list[tuple], optional): (rank, tree_index) pairs as returned by
            sample_tree_indices. Defaults to all trees ranked by objective.
        x_train (pd.DataFrame, optional): Binarized training features; with y_train
            and lambda_reg, enables per-tree regularized loss computation.
        y_train (pd.Series, optional): Training labels.
        lambda_reg (float, optional): Regularization λ used in the fit.

    Returns:
        dict: The exported payload (also written to out_path).
    """
    if indices is None:
        n = model.count_trees()
        order = sorted(range(n), key=model.get_tree_objective)
        indices = list(enumerate(order))

    trees = []
    for rank, idx in indices:
        paths, leaf_preds = model.get_tree_paths(idx)
        paths = [[int(s) for s in path] for path in paths]
        leaf_preds = [int(p) for p in leaf_preds]
        # get_tree_objective returns a (misclassification_count, normalized_loss)
        # tuple — keep both elements, unwrapping numpy scalars for JSON
        objective = model.get_tree_objective(idx)
        if isinstance(objective, (tuple, list)):
            objective = [v.item() if hasattr(v, "item") else v for v in objective]
        elif hasattr(objective, "item"):
            objective = objective.item()
        record = {
            "rank": rank,
            "index": int(idx),
            "objective": objective,
            "paths": paths,
            "leaf_predictions": leaf_preds,
            "num_leaves": len(paths),
            "depth": max(len(p) for p in paths),
            "split_features": sorted({feature_names[abs(s) - 1] for p in paths for s in p}),
        }
        if x_train is not None and y_train is not None and lambda_reg is not None:
            preds = model.get_predictions(idx, x_train)
            misclass = (preds != y_train.values).sum() / len(y_train)
            record["loss"] = float(misclass + lambda_reg * len(paths))
        trees.append(record)

    payload = {
        "feature_names": list(feature_names),
        "lambda_reg": lambda_reg,
        "n_trees_total": int(model.count_trees()),
        "trees": trees,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Exported {len(trees)} of {payload['n_trees_total']} trees to {out_path}")
    return payload


def load_trees(path):
    """
    Load a JSON file produced by export_praxis_trees.

    Args:
        path (str): Path to the exported JSON file.

    Returns:
        dict: Payload with "feature_names", "lambda_reg", "n_trees_total", "trees".
    """
    with open(path) as f:
        return json.load(f)


def build_tree(paths, leaf_predictions):
    """
    Reconstruct a nested tree from PRAXIS's flat per-leaf signed-id paths.

    All paths from one tree share prefixes exactly, so grouping by the first
    condition and recursing recovers the structure unambiguously.

    Args:
        paths (list[list[int]]): Signed feature ids per leaf, ±(column_index+1).
        leaf_predictions (list[int]): Predicted class per leaf, aligned with paths.

    Returns:
        dict: Either {"leaf": prediction} or
            {"feature": column_index, "true": subtree, "false": subtree}.
    """
    items = list(zip(paths, leaf_predictions))
    if len(items) == 1 and not items[0][0]:
        return {"leaf": items[0][1]}
    features = {abs(p[0]) for p, _ in items}
    if len(features) != 1:
        raise ValueError(f"Sibling paths disagree on split feature: {features}")
    feature = features.pop() - 1
    true_items = [(p[1:], y) for p, y in items if p[0] > 0]
    false_items = [(p[1:], y) for p, y in items if p[0] < 0]
    return {
        "feature": feature,
        "true": build_tree(*zip(*true_items)),
        "false": build_tree(*zip(*false_items)),
    }


def _layout(node, positions, depth, next_x):
    # Post-order: leaves take consecutive x slots, parents center over children
    if "leaf" in node:
        x = next_x[0]
        next_x[0] += 1
    else:
        x_true = _layout(node["true"], positions, depth + 1, next_x)
        x_false = _layout(node["false"], positions, depth + 1, next_x)
        x = (x_true + x_false) / 2
    positions[id(node)] = (x, -depth)
    return x


def draw_tree(tree_record, feature_names, class_names=("not satisfied", "satisfied"),
              feature_labels=None, title=None, ax=None):
    """
    Render one exported tree record as a matplotlib diagram.

    Internal nodes show the binarized feature name; edges are labeled yes/no
    (condition true/false); leaves are colored and labeled by predicted class.

    Args:
        tree_record (dict): One entry of payload["trees"] from load_trees.
        feature_names (list[str]): payload["feature_names"].
        class_names (tuple, optional): Display names for classes (0, 1).
        feature_labels (dict, optional): Raw binarized name → human-readable node
            label (may contain newlines). IMPORTANT: phrase labels so that the
            "yes" edge still means the raw condition is TRUE — e.g.
            "Customer_Type_disloyal Customer <= 0.5" → "Loyal customer?".
            Unlisted features fall back to the raw name.
        title (str, optional): Figure title. Defaults to a rank/loss/leaves summary.
        ax (matplotlib.axes.Axes, optional): Axes to draw on; created if omitted.

    Returns:
        matplotlib.axes.Axes: The axes containing the diagram.
    """
    root = build_tree(tree_record["paths"], tree_record["leaf_predictions"])
    positions = {}
    _layout(root, positions, 0, [0])

    if ax is None:
        width = max(6, 1.8 * tree_record["num_leaves"])
        height = max(3, 1.4 * tree_record["depth"])
        _, ax = plt.subplots(figsize=(width, height))

    def draw_node(node):
        x, y = positions[id(node)]
        if "leaf" in node:
            style = LEAF_STYLE[node["leaf"]]
            ax.text(x, y, class_names[node["leaf"]], ha="center", va="center",
                    fontsize=9, color=style["text"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=style["face"],
                              edgecolor="none"))
            return
        name = feature_names[node["feature"]]
        if feature_labels and name in feature_labels:
            label = feature_labels[name]
        else:
            label = name.replace(" <= ", "\n<= ")
        ax.text(x, y, label,
                ha="center", va="center", fontsize=9, color=NODE_TEXT,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=NODE_FACE,
                          edgecolor=NODE_EDGE))
        for branch, label in (("true", "yes"), ("false", "no")):
            cx, cy = positions[id(node[branch])]
            ax.plot([x, cx], [y - 0.12, cy + 0.12], color=NODE_EDGE,
                    linewidth=1.2, zorder=0)
            ax.text((x + cx) / 2, (y + cy) / 2, label, ha="center", va="center",
                    fontsize=8, color="#5f5d54",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none"))
            draw_node(node[branch])

    draw_node(root)
    if title is None:
        loss = tree_record.get("loss")
        loss_txt = f" | loss {loss:.6f}" if loss is not None else ""
        title = (f"Rank {tree_record['rank']}{loss_txt} | "
                 f"{tree_record['num_leaves']} leaves, depth {tree_record['depth']}")
    ax.set_title(title, fontsize=10)
    ax.margins(0.1)
    ax.axis("off")
    return ax
