import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from run_graph_msa_stage1 import (
    _pairwise_distances_from_tree,
    robinson_foulds_distance,
)


def read_distance_matrix(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    labels = rows[0][1:]
    matrix = np.asarray(
        [[float(value) for value in row[1:]] for row in rows[1:]],
        dtype=float,
    )
    return labels, matrix


def evaluate_tree(true_newick, predicted_newick, labels, predicted_matrix):
    true_matrix = _pairwise_distances_from_tree(
        true_newick,
        labels,
    )

    upper = np.triu_indices(len(labels), k=1)
    true_pairwise = true_matrix[upper]
    predicted_pairwise = predicted_matrix[upper]

    if np.std(predicted_pairwise) == 0:
        spearman_value = 0.0
    else:
        spearman_value = float(
            spearmanr(
                predicted_pairwise,
                true_pairwise,
            ).statistic
        )

    rf_distance, normalized_rf = robinson_foulds_distance(
        true_newick,
        predicted_newick,
    )

    return {
        "spearman": spearman_value,
        "rf": rf_distance,
        "normalized_rf": normalized_rf,
    }


cache_path = Path(
    # "simulated_global_single_1_3/sim_cache/"
    # "test_global_tree_single_group_seed_42.json"
    "simulated_global_groups_1_3/sim_cache/"
    "test_global_tree_global_groups_seed_42.json"
)
prediction_dir = Path("crisprformer_global_groups_1_3_strain_tree")

payload = json.loads(cache_path.read_text(encoding="utf-8"))
true_global_newick = payload["metadata"]["global_newick"]

# for method_name in [
#     "method_1_distance_fusion",
#     "method_2_spacer_profile",
# ]:
for method_name in [
    "method_1_distance_fusion",
    "method_2_embedding_fusion",
]:
    method_dir = prediction_dir / method_name

    labels, predicted_matrix = read_distance_matrix(
        method_dir / "strain_distance_matrix.csv"
    )

    predicted_newick = (
        method_dir / "strain_tree.nwk"
    ).read_text(encoding="utf-8")

    metrics = evaluate_tree(
        true_global_newick,
        predicted_newick,
        labels,
        predicted_matrix,
    )

    print(method_name)
    print(metrics)