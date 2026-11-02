from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from graph_msa_dataset import convert_strains_to_array_paths, group_array_paths_by_repeat_family
from path_alignment import AlignmentWeights, build_pairwise_distance_matrix
from project_config import TRAIN_CONFIG
from simulated_data import load_split_strains
from spacer_graph import build_repeat_family_graph, build_spacer_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic graph MSA array-level tree baseline.")
    parser.add_argument("--output-dir", default="graph_msa_4_8_seed42_benchmark_new_50")
    parser.add_argument("--split", default="test", choices=["training", "validation", "test"])
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG["simulation_seed"]))
    parser.add_argument("--simulation-cache-file", default="", help="Path to an existing simulated cache JSON. If provided, load it directly without regenerating data.",)
    parser.add_argument("--max-levenshtein-distance", type=int, default=0)
    parser.add_argument("--tree-method", default="nj")
    parser.add_argument("--simulation-num-strains", type=int, default=0)
    parser.add_argument("--simulation-num-groups", type=int, default=0)
    return parser.parse_args()

# Support CSV input for real strain data.
def read_spacers_from_csv(
    csv_path: str,
    require_repeat_columns: bool = False,
) -> list[dict]:
    del require_repeat_columns
    grouped = defaultdict(lambda: defaultdict(lambda: {"spacers": [], "repeat": "", "repeat_family": ""}))
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strain_id = row["Folder"].strip()
            array_id = row["Seq_Name"].strip()
            orientation = row["Orientation"].strip()
            spacer = (
                row["Spacer_Sequence"].strip()
                if orientation == "+"
                else row["Reverse_Complement_Sequence"].strip()
            )
            repeat = (
                row["DR_Consensus"].strip()
                if orientation == "+"
                else row["Reverse_Complement_DR"].strip()
            )
            normalized_repeat = repeat.strip().upper()
            payload = grouped[strain_id][array_id]
            payload["spacers"].append(spacer)
            if normalized_repeat and not payload["repeat"]:
                payload["repeat"] = normalized_repeat
                payload["repeat_family"] = normalized_repeat

    strains = []
    for strain_id, arrays in grouped.items():
        array_rows = []
        for array_id, payload in arrays.items():
            array_rows.append(
                {
                    "array_id": array_id,
                    "array_uid": f"{strain_id}::{array_id}",
                    "repeat": payload["repeat"],
                    "repeat_family": payload["repeat_family"],
                    "spacers": list(payload["spacers"]),
                    "candidate_group_id": f"real_group::{strain_id}::{array_id}",
                }
            )
        strains.append({"strain_id": strain_id, "arrays": array_rows})
    return strains


def _tree_to_newick(node, labels: list[str], parent_distance: float, leaf_names: dict[int, str]) -> str:
    if node.is_leaf():
        branch_length = max(parent_distance - node.dist, 0.0)
        return f"{leaf_names[node.id]}:{branch_length:.6f}"
    left = _tree_to_newick(node.left, labels, node.dist, leaf_names)
    right = _tree_to_newick(node.right, labels, node.dist, leaf_names)
    branch_length = max(parent_distance - node.dist, 0.0)
    if branch_length == 0.0:
        return f"({left},{right})"
    return f"({left},{right}):{branch_length:.6f}"


def linkage_to_newick(linkage_matrix: np.ndarray, labels: list[str]) -> str:
    root = to_tree(linkage_matrix, rd=False)
    leaf_names = {index: label for index, label in enumerate(labels)}
    return _tree_to_newick(root, labels, root.dist, leaf_names) + ";"


def _to_biopython_distance_matrix(distance_matrix: np.ndarray, labels: list[str]) -> DistanceMatrix:
    lower_triangle = []
    for row_index in range(len(labels)):
        row = []
        for col_index in range(row_index + 1):
            if row_index == col_index:
                row.append(0.0)
                continue
            row.append(float(distance_matrix[row_index, col_index]))
        lower_triangle.append(row)
    return DistanceMatrix(names=list(labels), matrix=lower_triangle)


def distance_matrix_to_newick(
    distance_matrix: np.ndarray,
    labels: list[str],
    tree_method: str = "nj",
) -> str:
    method = str(tree_method).strip().lower()
    if method == "nj":
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(_to_biopython_distance_matrix(distance_matrix, labels))
        # Clip NJ branch lengths after construction, preserving topology and input distances.
        for clade in tree.find_clades():
            if clade.branch_length is not None and clade.branch_length < 0.0:
                clade.branch_length = 0.0
        handle = StringIO()
        Phylo.write(tree, handle, "newick")
        return handle.getvalue().strip()

    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method=method)
    return linkage_to_newick(linkage_matrix, labels)


# def build_runtime_config(args: argparse.Namespace, output_dir: Path) -> dict:
#     config = dict(TRAIN_CONFIG)
#     config["data_source"] = "simulated"
#     config["simulation_seed"] = args.seed
#     config["simulation_cache_dir"] = str(output_dir / "sim_cache")

#     split_key = {
#         "training": "train",
#         "validation": "valid",
#         "test": "test",
#     }[args.split]
#     if args.simulation_num_strains > 0:
#         config[f"simulation_{split_key}_num_strains"] = args.simulation_num_strains
#     if args.simulation_num_groups > 0:
#         config[f"simulation_{split_key}_num_groups"] = args.simulation_num_groups
#     return config
def build_runtime_config(args: argparse.Namespace, output_dir: Path) -> dict:
    config = dict(TRAIN_CONFIG)
    config["data_source"] = "simulated"
    config["simulation_seed"] = args.seed
    config["simulation_cache_dir"] = str(output_dir / "sim_cache")

    if args.simulation_cache_file:
        config["simulation_cache_file"] = str(Path(args.simulation_cache_file))
        config["simulation_use_cache"] = True

    split_key = {
        "training": "train",
        "validation": "valid",
        "test": "test",
    }[args.split]
    if args.simulation_num_strains > 0:
        config[f"simulation_{split_key}_num_strains"] = args.simulation_num_strains
    if args.simulation_num_groups > 0:
        config[f"simulation_{split_key}_num_groups"] = args.simulation_num_groups
    return config

def _pairwise_distances_from_tree(newick: str, labels: list[str]) -> np.ndarray:
    tree = Phylo.read(StringIO(newick.strip()), "newick")
    terminals = {str(terminal.name): terminal for terminal in tree.get_terminals()}
    distance_matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for left_index, left_label in enumerate(labels):
        for right_index, right_label in enumerate(labels[left_index + 1 :], start=left_index + 1):
            distance = float(tree.distance(terminals[left_label], terminals[right_label]))
            distance_matrix[left_index, right_index] = distance
            distance_matrix[right_index, left_index] = distance
    return distance_matrix


def _distance_upper_triangle(distance_matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices(len(distance_matrix), k=1)
    return distance_matrix[rows, cols]


def _tree_splits(newick: str) -> set[frozenset[str]]:
    tree = Phylo.read(StringIO(newick.strip()), "newick")
    all_leaves = {str(terminal.name) for terminal in tree.get_terminals()}
    splits: set[frozenset[str]] = set()
    for clade in tree.find_clades(order="postorder"):
        leaves = {str(terminal.name) for terminal in clade.get_terminals()}
        if len(leaves) == 0 or len(leaves) == len(all_leaves):
            continue
        smaller_side = leaves if len(leaves) <= len(all_leaves) - len(leaves) else (all_leaves - leaves)
        if 1 < len(smaller_side) < len(all_leaves) - 1:
            splits.add(frozenset(smaller_side))
    return splits

# Compute the RF distance.
def robinson_foulds_distance(true_newick: str, predicted_newick: str) -> tuple[int, float]:
    true_splits = _tree_splits(true_newick)
    predicted_splits = _tree_splits(predicted_newick)
    rf_distance = len(true_splits - predicted_splits) + len(predicted_splits - true_splits)
    denominator = len(true_splits) + len(predicted_splits)
    normalized_rf = rf_distance / denominator if denominator else 0.0
    return rf_distance, normalized_rf


def save_distance_matrix_csv(distance_matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["array_uid", *labels])
        for label, row in zip(labels, distance_matrix):
            writer.writerow([label, *[f"{value:.6f}" for value in row]])


def save_family_graph_summary(family_graph, spacer_classes: dict, output_dir: Path) -> None:
    with open(output_dir / "nodes.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "spacer_class_id",
                "canonical_sequence",
                "support_count",
                "member_sequence_count",
                "predecessor_count",
                "successor_count",
            ],
        )
        writer.writeheader()
        for spacer_class_id, spacer_class in spacer_classes.items():
            node = family_graph.nodes[spacer_class_id]
            writer.writerow(
                {
                    "spacer_class_id": spacer_class_id,
                    "canonical_sequence": spacer_class.canonical_sequence,
                    "support_count": spacer_class.support_count,
                    "member_sequence_count": len(spacer_class.member_sequences),
                    "predecessor_count": len(node["predecessors"]),
                    "successor_count": len(node["successors"]),
                }
            )

    with open(output_dir / "edges.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "target", "support_count"],
        )
        writer.writeheader()
        for edge_payload in family_graph.edges.values():
            writer.writerow(
                {
                    "source": edge_payload["source"],
                    "target": edge_payload["target"],
                    "support_count": edge_payload["support_count"],
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = build_runtime_config(args, output_dir)
    strains_data, metadata = load_split_strains(args.split, runtime_config, read_spacers_from_csv)
    array_paths = convert_strains_to_array_paths(strains_data)
    grouped_paths = group_array_paths_by_repeat_family(array_paths)
    weights = AlignmentWeights()

    family_results: dict[str, dict] = {}
    family_summary_rows: list[dict] = []

    for family_index, (repeat_family, family_paths) in enumerate(sorted(grouped_paths.items()), start=1):
        family_output_dir = output_dir / f"family_{family_index:04d}"
        family_output_dir.mkdir(parents=True, exist_ok=True)

        annotated_paths, spacer_classes = build_spacer_classes(
            family_paths,
            max_levenshtein_distance=args.max_levenshtein_distance,
        )
        family_graph = build_repeat_family_graph(annotated_paths, spacer_classes)
        labels, distance_matrix, alignment_results = build_pairwise_distance_matrix(
            annotated_paths,
            family_graph,
            weights,
            max_levenshtein_distance=args.max_levenshtein_distance,
        )

        family_results[repeat_family] = {
            "paths": annotated_paths,
            "labels": labels,
            "distance_matrix": distance_matrix,
            "family_graph": family_graph,
            "alignment_results": alignment_results,
        }

        np.save(family_output_dir / "distance_matrix.npy", distance_matrix)
        save_distance_matrix_csv(distance_matrix, labels, family_output_dir / "distance_matrix.csv")
        save_family_graph_summary(family_graph, spacer_classes, family_output_dir)

        if len(labels) >= 2:
            newick = distance_matrix_to_newick(distance_matrix, labels, tree_method=args.tree_method)
            with open(family_output_dir / "tree.nwk", "w", encoding="utf-8") as handle:
                handle.write(newick + "\n")

        family_summary_rows.append(
            {
                "repeat_family": repeat_family,
                "num_arrays": len(annotated_paths),
                "num_spacer_classes": len(spacer_classes),
                "num_edges": len(family_graph.edges),
            }
        )

    group_metric_rows: list[dict] = []
    for group in metadata["groups"]:
        repeat_family = str(group["repeat_family"])
        family_result = family_results[repeat_family]
        family_labels = family_result["labels"]
        family_index_lookup = {label: index for index, label in enumerate(family_labels)}
        group_array_uids = [str(value) for value in group["array_leaf_labels"]]
        group_strain_ids = [str(value) for value in group["strain_ids"]]

        predicted_distance_matrix = np.zeros((len(group_array_uids), len(group_array_uids)), dtype=np.float64)
        family_distance_matrix = family_result["distance_matrix"]
        for left_index, left_uid in enumerate(group_array_uids):
            for right_index, right_uid in enumerate(group_array_uids):
                predicted_distance_matrix[left_index, right_index] = family_distance_matrix[
                    family_index_lookup[left_uid],
                    family_index_lookup[right_uid],
                ]

        predicted_newick = distance_matrix_to_newick(
            predicted_distance_matrix,
            group_strain_ids,
            tree_method=args.tree_method,
        )
        true_newick = str(group["newick"]).strip()
        true_distance_matrix = _pairwise_distances_from_tree(true_newick, group_strain_ids)

        predicted_pairwise = _distance_upper_triangle(predicted_distance_matrix)
        true_pairwise = _distance_upper_triangle(true_distance_matrix)
        
        # Values closer to 1 are better; values closer to 0 are worse, and negative values indicate reversed rankings.
        if np.std(predicted_pairwise)==0:
            spearman_value = 0
            group_id = str(group["candidate_group_id"])
            print(f"Group {group_id}: Predicted pairwise distances are constant.")
        else:
            spearman_value = float(spearmanr(predicted_pairwise, true_pairwise).statistic)
        
        rf_distance, normalized_rf = robinson_foulds_distance(true_newick, predicted_newick)

        group_output_dir = output_dir / "group_trees" / str(group["candidate_group_id"])
        group_output_dir.mkdir(parents=True, exist_ok=True)
        np.save(group_output_dir / "predicted_distance_matrix.npy", predicted_distance_matrix)
        np.save(group_output_dir / "true_distance_matrix.npy", true_distance_matrix)
        with open(group_output_dir / "predicted_tree.nwk", "w", encoding="utf-8") as handle:
            handle.write(predicted_newick + "\n")
        with open(group_output_dir / "true_tree.nwk", "w", encoding="utf-8") as handle:
            handle.write(true_newick + "\n")

        group_metric_rows.append(
            {
                "candidate_group_id": str(group["candidate_group_id"]),
                "repeat_family": repeat_family,
                "num_arrays": len(group_array_uids),
                "spearman_correlation": spearman_value,
                "rf_distance": rf_distance,
                "normalized_rf": normalized_rf,
            }
        )

    with open(output_dir / "family_summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["repeat_family", "num_arrays", "num_spacer_classes", "num_edges"],
        )
        writer.writeheader()
        writer.writerows(family_summary_rows)

    with open(output_dir / "group_metrics.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_group_id",
                "repeat_family",
                "num_arrays",
                "spearman_correlation",
                "rf_distance",
                "normalized_rf",
            ],
        )
        writer.writeheader()
        writer.writerows(group_metric_rows)

    mean_spearman = float(np.mean([row["spearman_correlation"] for row in group_metric_rows]))
    mean_normalized_rf = float(np.mean([row["normalized_rf"] for row in group_metric_rows]))
    print(f"split={args.split}")
    print(f"num_strains={len(strains_data)}")
    print(f"num_arrays={len(array_paths)}")
    print(f"num_repeat_families={len(grouped_paths)}")
    print(f"mean_spearman={mean_spearman:.6f}")
    print(f"mean_normalized_rf={mean_normalized_rf:.6f}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":

    start = time.perf_counter()
    main()
    end = time.perf_counter()

    print(f"Runtime: {end-start:.6f} seconds")
