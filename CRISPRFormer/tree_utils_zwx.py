from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import torch


@dataclass
class TreeNodeZwx:
    # Store a Newick tree node and the branch length to its parent.

    name: str = ""
    length: float = 0.0
    children: List["TreeNodeZwx"] = field(default_factory=list)


def parse_newick_zwx(newick: str) -> TreeNodeZwx:
    # Parse simple Newick strings from the simulated data.

    text = newick.strip().rstrip(";")
    index = 0

    def parse_node() -> TreeNodeZwx:
        nonlocal index
        children: List[TreeNodeZwx] = []
        name = ""
        if index < len(text) and text[index] == "(":
            index += 1
            while True:
                children.append(parse_node())
                if index >= len(text):
                    break
                if text[index] == ",":
                    index += 1
                    continue
                if text[index] == ")":
                    index += 1
                    break
        start = index
        while index < len(text) and text[index] not in ":,()":
            index += 1
        name = text[start:index].strip()
        length = 0.0
        if index < len(text) and text[index] == ":":
            index += 1
            start = index
            while index < len(text) and text[index] not in ",()":
                index += 1
            raw_length = text[start:index].strip()
            length = float(raw_length) if raw_length else 0.0
        return TreeNodeZwx(name=name, length=length, children=children)

    return parse_node()


def collect_leaf_names_zwx(node: TreeNodeZwx) -> List[str]:
    # Collect leaf names in tree traversal order.

    if not node.children:
        return [node.name]
    names: List[str] = []
    for child in node.children:
        names.extend(collect_leaf_names_zwx(child))
    return names


def _collect_paths_zwx(
    node: TreeNodeZwx,
    path: List[int],
    distance: float,
    paths: Dict[str, List[Tuple[int, float]]],
    next_id: List[int],
) -> None:
    # Recursively record the root-to-leaf node path and cumulative distances for each leaf.

    node_id = next_id[0]
    next_id[0] += 1
    current_path = path + [node_id]
    current_distance = distance + node.length
    if not node.children:
        paths[node.name] = [(node_id, current_distance)]
        for ancestor_id, ancestor_distance in zip(current_path[:-1], [0.0] * (len(current_path) - 1)):
            del ancestor_id, ancestor_distance
        return
    child_paths: Dict[str, List[Tuple[int, float]]] = {}
    for child in node.children:
        _collect_paths_zwx(child, current_path, current_distance, child_paths, next_id)
    for leaf, child_path in child_paths.items():
        paths[leaf] = [(node_id, current_distance)] + child_path


def pairwise_tree_distance_matrix_zwx(newick: str, labels: Sequence[str]) -> torch.Tensor:
    # Convert the ground-truth tree into a pairwise tree distance matrix in array order.

    root = parse_newick_zwx(newick)
    paths: Dict[str, List[Tuple[int, float]]] = {}
    _collect_paths_zwx(root, [], -root.length, paths, [0])
    size = len(labels)
    matrix = torch.zeros(size, size, dtype=torch.float32)
    for i, left in enumerate(labels):
        left_path = paths[str(left)]
        left_dist = left_path[-1][1]
        left_ancestors = {node_id: dist for node_id, dist in left_path}
        for j, right in enumerate(labels):
            if i == j:
                continue
            right_path = paths[str(right)]
            right_dist = right_path[-1][1]
            common_dist = 0.0
            for node_id, dist in right_path:
                if node_id in left_ancestors:
                    common_dist = dist
            matrix[i, j] = float(left_dist + right_dist - 2.0 * common_dist)
    max_value = matrix.max().clamp_min(1e-6)
    return matrix / max_value


def _leaf_set_zwx(node: TreeNodeZwx, splits: Set[frozenset[str]], all_leaves: Set[str]) -> Set[str]:
    # Recursively extract the nontrivial split corresponding to each internal node.

    if not node.children:
        return {node.name}
    leaves: Set[str] = set()
    for child in node.children:
        leaves.update(_leaf_set_zwx(child, splits, all_leaves))
    if 1 < len(leaves) < len(all_leaves) - 1:
        other = all_leaves - leaves
        canonical = leaves if len(leaves) <= len(other) else other
        splits.add(frozenset(canonical))
    return leaves


def tree_splits_zwx(newick: str) -> Set[frozenset[str]]:
    # Extract the set of nontrivial unrooted splits from a Newick tree for RF distance calculation.

    root = parse_newick_zwx(newick)
    all_leaves = set(collect_leaf_names_zwx(root))
    splits: Set[frozenset[str]] = set()
    _leaf_set_zwx(root, splits, all_leaves)
    return splits


def topology_distribution_matrix_zwx(
    newick: str,
    labels: Sequence[str],
    temperature: float = 0.5,
) -> torch.Tensor:
    # Convert the tree topology into a distribution of pairwise leaf co-clustering probabilities for KL supervision.

    labels = [str(label) for label in labels]
    splits = tree_splits_zwx(newick)
    distance = pairwise_tree_distance_matrix_zwx(newick, labels)
    scores = -distance / max(float(temperature), 1e-6)
    for split in splits:
        indices = [index for index, label in enumerate(labels) if label in split]
        for i in indices:
            for j in indices:
                if i != j:
                    scores[i, j] += 1.0
    diagonal = torch.eye(len(labels), dtype=torch.bool)
    scores = scores.masked_fill(diagonal, -math.inf)
    return torch.softmax(scores.flatten(), dim=0).view(len(labels), len(labels)).masked_fill(diagonal, 0.0)


def agglomerative_splits_from_distance_zwx(
    labels: Sequence[str],
    distance_matrix: torch.Tensor,
) -> Set[frozenset[str]]:
    # Approximately recover tree splits from predicted distances using average-linkage clustering.

    label_set = set(str(label) for label in labels)
    clusters: List[Set[str]] = [{str(label)} for label in labels]
    label_to_index = {str(label): index for index, label in enumerate(labels)}
    splits: Set[frozenset[str]] = set()
    while len(clusters) > 1:
        best_pair = (0, 1)
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                values = [
                    float(distance_matrix[label_to_index[a], label_to_index[b]])
                    for a in clusters[i]
                    for b in clusters[j]
                ]
                mean_distance = sum(values) / max(1, len(values))
                if mean_distance < best_distance:
                    best_distance = mean_distance
                    best_pair = (i, j)
        i, j = best_pair
        merged = clusters[i] | clusters[j]
        for index in sorted(best_pair, reverse=True):
            clusters.pop(index)
        clusters.append(merged)
        if 1 < len(merged) < len(label_set) - 1:
            other = label_set - merged
            canonical = merged if len(merged) <= len(other) else other
            splits.add(frozenset(canonical))
    return splits


def normalized_rf_distance_zwx(
    true_newick: str,
    labels: Sequence[str],
    predicted_distance_matrix: torch.Tensor,
) -> float:
    # Compute the normalized RF distance between ground-truth tree splits and predicted clustering splits.

    true_splits = tree_splits_zwx(true_newick)
    predicted_splits = agglomerative_splits_from_distance_zwx(labels, predicted_distance_matrix.detach().cpu())
    denominator = max(1, len(true_splits) + len(predicted_splits))
    return len(true_splits.symmetric_difference(predicted_splits)) / denominator


__all__ = [
    "TreeNodeZwx",
    "parse_newick_zwx",
    "pairwise_tree_distance_matrix_zwx",
    "topology_distribution_matrix_zwx",
    "tree_splits_zwx",
    "agglomerative_splits_from_distance_zwx",
    "normalized_rf_distance_zwx",
]
