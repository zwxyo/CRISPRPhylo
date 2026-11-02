from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from crispr_data_pipeline_zwx import (
    build_dataset_from_csv_zwx, 
    build_dataset_from_simulated_dir_zwx, 
    collate_crispr_tree_samples_zwx, 
    spacer_vector_dim_zwx
)
from crispr_model_zwx import CRISPRRepeatAxialModelZwx
from tree_utils_zwx import tree_splits_zwx


def parse_args_zwx() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer CRISPR trees with the current ZWX model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="simulated_data_theta_1_3_seed42")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "training", "validation"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kmer-size", type=int, default=None)
    parser.add_argument("--alignment-threshold", type=float, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--max-repeat-distance", type=int, default=0)
    parser.add_argument("--min-arrays", type=int, default=3)
    parser.add_argument("--min-strains", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    return parser.parse_args()


def checkpoint_args_zwx(checkpoint: dict) -> dict:
    # Read the training parameters saved in the checkpoint.

    return dict(checkpoint.get("args", {}))


def safe_label_zwx(label: str) -> str:
    # Convert leaf names to a Newick-safe format.

    return re.sub(r"[\s,:;()]+", "_", str(label))


def move_batch_to_device_zwx(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    # Move tensors in the batch to the specified device.

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def upper_values_zwx(matrix: torch.Tensor, size: int) -> torch.Tensor:
    # Extract upper-triangular distances from the valid region of the distance matrix.

    row, col = torch.triu_indices(size, size, offset=1, device=matrix.device)
    return matrix[:size, :size][row, col]


def rank_zwx(values: torch.Tensor) -> torch.Tensor:
    # Compute simple ranks for a one-dimensional vector.

    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def spearman_zwx(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Compute the Spearman correlation between predicted and ground-truth distances.

    left = rank_zwx(pred.float())
    right = rank_zwx(target.float())
    left = left - left.mean()
    right = right - right.mean()
    denom = left.norm() * right.norm()
    if denom.item() <= 1e-8:
        return 0.0
    return float((left * right).sum() / denom)


def rf_zwx(true_newick: str, pred_newick: str) -> Tuple[int, float]:
    # Compute the RF distance between the ground-truth and predicted trees.

    true_splits = tree_splits_zwx(true_newick)
    pred_splits = tree_splits_zwx(pred_newick)
    distance = len(true_splits.symmetric_difference(pred_splits))
    total = max(1, len(true_splits) + len(pred_splits))
    return distance, distance / total


def matrix_value_zwx(matrix: torch.Tensor, i: int, j: int) -> float:
    # Read a scalar value from the distance matrix.

    return float(matrix[i, j].detach().cpu())


def distance_dict_zwx(labels: Sequence[str], matrix: torch.Tensor) -> Dict[Tuple[str, str], float]:
    # Convert the distance matrix into a dictionary of distances for unordered pairs.

    return {
        (left, right): matrix_value_zwx(matrix, i, j)
        for i, left in enumerate(labels)
        for j, right in enumerate(labels)
        if i < j
    }


def key_zwx(left: str, right: str) -> Tuple[str, str]:
    # Generate a canonical key for the dictionary of unordered-pair distances.

    return (left, right) if left < right else (right, left)


def neighbor_joining_zwx(labels: Sequence[str], matrix: torch.Tensor) -> str:
    # Build a Newick tree from the distance matrix using the Neighbor-Joining algorithm.

    active = [safe_label_zwx(label) for label in labels]
    nodes = {label: label for label in active}
    dist = distance_dict_zwx(active, matrix)
    index = 1
    while len(active) > 2:
        n = len(active)
        total = {label: sum(dist[key_zwx(label, other)] for other in active if other != label) for label in active}
        left, right = min(
            ((a, b) for i, a in enumerate(active) for b in active[i + 1:]),
            key=lambda pair: (n - 2) * dist[key_zwx(*pair)] - total[pair[0]] - total[pair[1]],
        )
        pair_dist = dist[key_zwx(left, right)]
        left_len = max(0.0, 0.5 * pair_dist + (total[left] - total[right]) / (2.0 * (n - 2)))
        right_len = max(0.0, pair_dist - left_len)
        merged = f"inner_{index}"
        index += 1
        nodes[merged] = f"({nodes[left]}:{left_len:.6f},{nodes[right]}:{right_len:.6f})"
        rest = [label for label in active if label not in {left, right}]
        for other in rest:
            value = 0.5 * (dist[key_zwx(left, other)] + dist[key_zwx(right, other)] - pair_dist)
            dist[key_zwx(merged, other)] = max(0.0, value)
        active = rest + [merged]
    left, right = active
    length = max(0.0, dist[key_zwx(left, right)] / 2.0)
    return f"({nodes[left]}:{length:.6f},{nodes[right]}:{length:.6f});"


def write_matrix_zwx(path: Path, labels: Sequence[str], matrix: torch.Tensor) -> None:
    # Write the distance matrix to a CSV file.

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *labels])
        for i, label in enumerate(labels):
            writer.writerow([label, *[f"{matrix_value_zwx(matrix, i, j):.8f}" for j in range(len(labels))]])


def write_metrics_zwx(path: Path, rows: Sequence[dict]) -> None:
    # Write tree reconstruction metrics for each group.

    fields = ["candidate_group_id", "num_arrays", "spearman_correlation", "rf_distance", "normalized_rf"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_zwx(rows: Sequence[dict], key: str) -> float:
    # Compute the mean of the specified metric.

    values = [float(row[key]) for row in rows]
    return sum(values) / max(1, len(values))


@torch.no_grad()
# def infer_trees_zwx(model: CRISPRRepeatAxialModelZwx, loader: DataLoader, device: torch.device, output_dir: Path) -> Dict[str, float]:
#     # Predict distance matrices, build Newick trees, and save results for each group.

#     model.eval()
#     tree_dir = output_dir / "group_trees"
#     tree_dir.mkdir(parents=True, exist_ok=True)
#     rows = []
#     for batch in loader:
#         batch = move_batch_to_device_zwx(batch, device)
#         pred = model(batch)["pairwise_distance"].detach().cpu()
#         target = batch["target_distance"].detach().cpu()
#         row_mask = batch["row_mask"].detach().cpu()
#         for batch_index in range(pred.size(0)):
#             size = int(row_mask[batch_index].sum().item())
#             labels = batch["strain_labels"][batch_index][:size]
#             group_id = str(batch["group_id"][batch_index])
#             true_newick = str(batch["newick"][batch_index]).strip()
#             pred_matrix = pred[batch_index, :size, :size]
#             pred_matrix = 0.5 * (pred_matrix + pred_matrix.transpose(0, 1))
#             pred_matrix.fill_diagonal_(0.0)
#             true_matrix = target[batch_index, :size, :size]
#             pred_newick = neighbor_joining_zwx(labels, pred_matrix)
#             rf_distance, normalized_rf = rf_zwx(true_newick, pred_newick)
#             group_dir = tree_dir / group_id
#             group_dir.mkdir(parents=True, exist_ok=True)
#             write_matrix_zwx(group_dir / "predicted_distance_matrix.csv", labels, pred_matrix)
#             write_matrix_zwx(group_dir / "true_distance_matrix.csv", labels, true_matrix)
#             (group_dir / "predicted_tree.nwk").write_text(pred_newick + "\n", encoding="utf-8")
#             (group_dir / "true_tree.nwk").write_text(true_newick + "\n", encoding="utf-8")
#             rows.append({
#                 "candidate_group_id": group_id,
#                 "num_arrays": size,
#                 "spearman_correlation": spearman_zwx(upper_values_zwx(pred_matrix, size), upper_values_zwx(true_matrix, size)),
#                 "rf_distance": rf_distance,
#                 "normalized_rf": normalized_rf,
#             })
#     write_metrics_zwx(output_dir / "group_metrics.csv", rows)
#     return {
#         "num_groups": float(len(rows)),
#         "mean_spearman": mean_zwx(rows, "spearman_correlation"),
#         "mean_rf_distance": mean_zwx(rows, "rf_distance"),
#         "mean_normalized_rf": mean_zwx(rows, "normalized_rf"),
#     }

# @torch.no_grad()
# def infer_trees_zwx(
#     model: CRISPRRepeatAxialModelZwx,
#     loader: DataLoader,
#     device: torch.device,
#     output_dir: Path,
# ) -> Dict[str, float]:
#     # Predict distance matrices, build Newick trees, and save results for each group.

#     model.eval()
#     tree_dir = output_dir / "group_trees"
#     tree_dir.mkdir(parents=True, exist_ok=True)
#     rows = []
#     metric_rows = []

#     for batch in loader:
#         batch = move_batch_to_device_zwx(batch, device)
#         pred = model(batch)["pairwise_distance"].detach().cpu()
#         target = batch["target_distance"].detach().cpu()
#         row_mask = batch["row_mask"].detach().cpu()

#         for batch_index in range(pred.size(0)):
#             size = int(row_mask[batch_index].sum().item())
#             array_labels = batch["labels"][batch_index][:size]
#             strain_labels = batch["strain_labels"][batch_index][:size]
#             group_id = str(batch["group_id"][batch_index])
#             true_newick = str(batch["newick"][batch_index]).strip()

#             # Continue evaluating simulated data with strain labels as before; output an array tree for real-data CSV input when no ground-truth tree is available.
#             labels = strain_labels if true_newick else array_labels

#             pred_matrix = pred[batch_index, :size, :size]
#             pred_matrix = 0.5 * (pred_matrix + pred_matrix.transpose(0, 1))
#             pred_matrix.fill_diagonal_(0.0)
#             pred_newick = neighbor_joining_zwx(labels, pred_matrix)

#             group_dir = tree_dir / group_id
#             group_dir.mkdir(parents=True, exist_ok=True)
#             write_matrix_zwx(group_dir / "predicted_distance_matrix.csv", labels, pred_matrix)
#             (group_dir / "predicted_tree.nwk").write_text(pred_newick + "\n", encoding="utf-8")

#             row = {
#                 "candidate_group_id": group_id,
#                 "num_arrays": size,
#                 "spearman_correlation": "",
#                 "rf_distance": "",
#                 "normalized_rf": "",
#             }

#             if true_newick:
#                 true_matrix = target[batch_index, :size, :size]
#                 rf_distance, normalized_rf = rf_zwx(true_newick, pred_newick)
#                 write_matrix_zwx(group_dir / "true_distance_matrix.csv", labels, true_matrix)
#                 (group_dir / "true_tree.nwk").write_text(true_newick + "\n", encoding="utf-8")

#                 row["spearman_correlation"] = spearman_zwx(
#                     upper_values_zwx(pred_matrix, size),
#                     upper_values_zwx(true_matrix, size),
#                 )
#                 row["rf_distance"] = rf_distance
#                 row["normalized_rf"] = normalized_rf
#                 metric_rows.append(row)

#             rows.append(row)

#     write_metrics_zwx(output_dir / "group_metrics.csv", rows)

#     if metric_rows:
#         return {
#             "num_groups": float(len(rows)),
#             "evaluated_groups": float(len(metric_rows)),
#             "mean_spearman": mean_zwx(metric_rows, "spearman_correlation"),
#             "mean_rf_distance": mean_zwx(metric_rows, "rf_distance"),
#             "mean_normalized_rf": mean_zwx(metric_rows, "normalized_rf"),
#         }

#     return {
#         "num_groups": float(len(rows)),
#         "evaluated_groups": 0.0,
#         "mean_spearman": float("nan"),
#         "mean_rf_distance": float("nan"),
#         "mean_normalized_rf": float("nan"),
#     }
@torch.no_grad()
def infer_trees_zwx(model: CRISPRRepeatAxialModelZwx, loader: DataLoader, device: torch.device, output_dir: Path) -> Dict[str, object]:
    # Predict distance matrices, build Newick trees, and save results for each group.

    model.eval()
    tree_dir = output_dir / "group_trees"
    tree_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    num_groups = 0

    for batch in loader:
        batch = move_batch_to_device_zwx(batch, device)
        outputs = model(batch)
        pred = outputs["pairwise_distance"].detach().cpu()
        target = batch["target_distance"].detach().cpu()
        row_mask = batch["row_mask"].detach().cpu()

        for batch_index in range(pred.size(0)):
            num_groups += 1
            size = int(row_mask[batch_index].sum().item())
            labels = batch["strain_labels"][batch_index][:size]
            group_id = str(batch["group_id"][batch_index])
            true_newick = str(batch["newick"][batch_index]).strip()
            has_truth = bool(true_newick)

            pred_matrix = pred[batch_index, :size, :size]
            pred_matrix = 0.5 * (pred_matrix + pred_matrix.transpose(0, 1))
            pred_matrix.fill_diagonal_(0.0)
            pred_newick = neighbor_joining_zwx(labels, pred_matrix)

            group_dir = tree_dir / group_id
            group_dir.mkdir(parents=True, exist_ok=True)
            write_matrix_zwx(group_dir / "predicted_distance_matrix.csv", labels, pred_matrix)
            (group_dir / "predicted_tree.nwk").write_text(pred_newick + "\n", encoding="utf-8")

            if has_truth:
                true_matrix = target[batch_index, :size, :size]
                rf_distance, normalized_rf = rf_zwx(true_newick, pred_newick)
                write_matrix_zwx(group_dir / "true_distance_matrix.csv", labels, true_matrix)
                (group_dir / "true_tree.nwk").write_text(true_newick + "\n", encoding="utf-8")
                rows.append({
                    "candidate_group_id": group_id,
                    "num_arrays": size,
                    "spearman_correlation": spearman_zwx(upper_values_zwx(pred_matrix, size), upper_values_zwx(true_matrix, size)),
                    "rf_distance": rf_distance,
                    "normalized_rf": normalized_rf,
                })

    if rows:
        write_metrics_zwx(output_dir / "group_metrics.csv", rows)

    return {
        "num_groups": float(num_groups),
        "num_groups_with_truth": float(len(rows)),
        "mean_spearman": mean_zwx(rows, "spearman_correlation") if rows else None,
        "mean_rf_distance": mean_zwx(rows, "rf_distance") if rows else None,
        "mean_normalized_rf": mean_zwx(rows, "normalized_rf") if rows else None,
    }

    
def build_model_zwx(checkpoint: dict, args: argparse.Namespace) -> CRISPRRepeatAxialModelZwx:
    # Rebuild the current project model from the checkpoint.

    saved_args = checkpoint_args_zwx(checkpoint)
    kmer_size = int(args.kmer_size or saved_args.get("kmer_size", 4))
    return CRISPRRepeatAxialModelZwx(
        spacer_dim=spacer_vector_dim_zwx(kmer_size),
        model_dim=int(saved_args.get("model_dim", 192)),
        num_heads=int(saved_args.get("num_heads", 6)),
        num_layers=int(saved_args.get("num_layers", 4)),
        dropout=float(saved_args.get("dropout", 0.1)),
    )


# def build_loader_zwx(args: argparse.Namespace, checkpoint: dict) -> DataLoader:
#     # Build an inference DataLoader for the current project dataset.

#     saved_args = checkpoint_args_zwx(checkpoint)
#     dataset = build_dataset_from_simulated_dir_zwx(
#         data_dir=args.data_dir,
#         split=args.split,
#         kmer_size=int(args.kmer_size or saved_args.get("kmer_size", 4)),
#         alignment_threshold=float(args.alignment_threshold or saved_args.get("alignment_threshold", 0.18)),
#         max_groups=args.max_groups,
#     )
#     return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_crispr_tree_samples_zwx)

# def build_loader_zwx(args: argparse.Namespace, checkpoint: dict) -> DataLoader:
#     saved_args = checkpoint_args_zwx(checkpoint)
#     common = {
#         "kmer_size": int(args.kmer_size or saved_args.get("kmer_size", 4)),
#         "alignment_threshold": float(args.alignment_threshold or saved_args.get("alignment_threshold", 0.18)),
#         "max_groups": args.max_groups,
#     }
#     if args.csv_path:
#         dataset = build_dataset_from_csv_zwx(
#             csv_path=args.csv_path,
#             max_repeat_distance=args.max_repeat_distance,
#             min_arrays=args.min_arrays,
#             min_strains=args.min_strains,
#             **common,
#         )
#     else:
#         dataset = build_dataset_from_simulated_dir_zwx(
#             data_dir=args.data_dir,
#             split=args.split,
#             **common,
#         )
#     return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_crispr_tree_samples_zwx)

def build_loader_zwx(args: argparse.Namespace, checkpoint: dict) -> DataLoader:
    # Build an inference DataLoader for the current project dataset.

    saved_args = checkpoint_args_zwx(checkpoint)
    common_kwargs = dict(
        kmer_size=int(args.kmer_size or saved_args.get("kmer_size", 4)),
        alignment_threshold=float(args.alignment_threshold or saved_args.get("alignment_threshold", 0.18)),
        max_groups=args.max_groups,
    )

    if args.csv_path:
        dataset = build_dataset_from_csv_zwx(
            csv_path=args.csv_path,
            max_repeat_distance=args.max_repeat_distance,
            **common_kwargs,
        )
    else:
        dataset = build_dataset_from_simulated_dir_zwx(
            data_dir=args.data_dir,
            split=args.split,
            **common_kwargs,
        )

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_crispr_tree_samples_zwx)


def main_zwx() -> None:
    # Load the checkpoint and run inference for tree reconstruction.

    args = parse_args_zwx()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loader = build_loader_zwx(args, checkpoint)
    model = build_model_zwx(checkpoint, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    metrics = infer_trees_zwx(model.to(device), loader, device, Path(args.output_dir))
    # print(f"split={args.split}")
    # print(f"num_groups={int(metrics['num_groups'])}")
    # print(f"mean_spearman={metrics['mean_spearman']:.6f}")
    # print(f"mean_rf_distance={metrics['mean_rf_distance']:.6f}")
    # print(f"mean_normalized_rf={metrics['mean_normalized_rf']:.6f}")
    print(f"input_mode={'csv' if args.csv_path else args.split}")
    print(f"num_groups={int(metrics['num_groups'])}")
    if metrics["num_groups_with_truth"]:
        print(f"mean_spearman={metrics['mean_spearman']:.6f}")
        print(f"mean_rf_distance={metrics['mean_rf_distance']:.6f}")
        print(f"mean_normalized_rf={metrics['mean_normalized_rf']:.6f}")
    else:
        print("ground_truth_metrics=not_available_for_csv_input")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    start = time.perf_counter()
    main_zwx()
    end = time.perf_counter()

    print(f"Runtime: {end-start:.6f} seconds")
