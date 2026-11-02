import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crispr_data_pipeline_zwx import (
    build_dataset_from_simulated_dir_zwx,
    collate_crispr_tree_samples_zwx,
    spacer_vector_dim_zwx,
    build_dataset_from_csv_zwx,
)
from crispr_model_zwx import CRISPRRepeatAxialModelZwx
from infer_tree_zwx import move_batch_to_device_zwx, neighbor_joining_zwx, write_matrix_zwx


def parse_args_zwx() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strain-level CRISPR trees with two aggregation methods.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="simulated_data_theta_1_3_final")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "training", "validation"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kmer-size", type=int, default=None)
    parser.add_argument("--alignment-threshold", type=float, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--within-group-agg", default="mean", choices=["mean", "min"])
    parser.add_argument("--group-weight", default="strain_count", choices=["uniform", "strain_count"])
    parser.add_argument("--embedding-distance", default="euclidean", choices=["cosine", "euclidean"])
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--max-repeat-distance", type=int, default=0)
    parser.add_argument("--min-arrays", type=int, default=3)
    parser.add_argument("--min-strains", type=int, default=2)
    return parser.parse_args()


def checkpoint_args_zwx(checkpoint: dict) -> dict:
    # Read the training parameters saved in the checkpoint.

    return dict(checkpoint.get("args", {}))


def build_model_zwx(checkpoint: dict, args: argparse.Namespace) -> CRISPRRepeatAxialModelZwx:
    # Rebuild the current CRISPR array model using the checkpoint parameters.

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
#     # Build a CRISPR group inference DataLoader for the specified data split.

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
    # Build a CRISPR group inference DataLoader for the specified data split.

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


def pair_key_zwx(left: str, right: str) -> Tuple[str, str]:
    # Generate a canonical key for an unordered strain pair.

    return (left, right) if left < right else (right, left)


def aggregate_values_zwx(values: Sequence[float], mode: str) -> float:
    # Aggregate multiple array distances for the same strain pair within a group using the specified method.

    if mode == "min":
        return min(values)
    return sum(values) / len(values)


def group_weight_zwx(mode: str, unique_strain_count: int) -> float:
    # Compute a group's weight for fusion across groups.

    if mode == "strain_count":
        return float(unique_strain_count)
    return 1.0


def normalize_distance_matrix_zwx(matrix: torch.Tensor) -> torch.Tensor:
    # Scale the distance matrix to a maximum of 1 while keeping the diagonal at 0.

    matrix = matrix.clone()
    matrix.fill_diagonal_(0.0)
    max_value = matrix.max().clamp_min(1e-6)
    return matrix / max_value


def build_distance_fusion_matrix_zwx(
    labels: Sequence[str],
    distance_sums: Dict[Tuple[str, str], float],
    weight_sums: Dict[Tuple[str, str], float],
) -> torch.Tensor:
    # Build the full strain distance matrix from distances weighted across groups.

    observed = [distance_sums[key] / weight_sums[key] for key in distance_sums]
    fill_value = max(observed)
    matrix = torch.zeros(len(labels), len(labels), dtype=torch.float32)
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if i >= j:
                continue
            key = pair_key_zwx(left, right)
            value = distance_sums[key] / weight_sums[key] if key in distance_sums else fill_value
            matrix[i, j] = matrix[j, i] = float(value)
    return normalize_distance_matrix_zwx(matrix)


def build_evidence_matrix_zwx(labels: Sequence[str], evidence_counts: Dict[Tuple[str, str], int]) -> torch.Tensor:
    # Build a matrix of shared-group evidence counts for each strain pair.

    matrix = torch.zeros(len(labels), len(labels), dtype=torch.float32)
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if i >= j:
                continue
            value = float(evidence_counts.get(pair_key_zwx(left, right), 0))
            matrix[i, j] = matrix[j, i] = value
    return matrix


def build_embedding_matrix_zwx(
    labels: Sequence[str],
    strain_embeddings: Dict[str, List[torch.Tensor]],
    distance_mode: str,
) -> torch.Tensor:
    # Aggregate array embeddings for each strain and compute the inter-strain distance matrix.

    embeddings = torch.stack([torch.stack(strain_embeddings[label]).mean(dim=0) for label in labels])
    matrix = torch.zeros(len(labels), len(labels), dtype=torch.float32)
    if distance_mode == "cosine":
        normalized = F.normalize(embeddings, p=2, dim=1)
        matrix = 1.0 - normalized @ normalized.transpose(0, 1)
    else:
        matrix = torch.cdist(embeddings, embeddings, p=2)
    return normalize_distance_matrix_zwx(matrix.detach().cpu())


def collect_strain_evidence_zwx(
    model: CRISPRRepeatAxialModelZwx,
    loader: DataLoader,
    device: torch.device,
    within_group_agg: str,
    group_weight_mode: str,
) -> Tuple[List[str], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], int], Dict[str, List[torch.Tensor]], int]:
    # Run model inference and collect strain evidence for distance fusion and embedding fusion.

    model.eval()
    strain_set = set()
    distance_sums: Dict[Tuple[str, str], float] = defaultdict(float)
    weight_sums: Dict[Tuple[str, str], float] = defaultdict(float)
    evidence_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    strain_embeddings: Dict[str, List[torch.Tensor]] = defaultdict(list)
    group_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device_zwx(batch, device)
            outputs = model(batch)
            pred = outputs["pairwise_distance"].detach().cpu()
            embeddings = outputs["row_embeddings"].detach().cpu()
            row_mask = batch["row_mask"].detach().cpu()
            for batch_index in range(pred.size(0)):
                size = int(row_mask[batch_index].sum().item())
                strains = [str(label) for label in batch["strain_labels"][batch_index][:size]]
                strain_set.update(strains)
                group_count += 1
                pred_matrix = pred[batch_index, :size, :size]
                pred_matrix = 0.5 * (pred_matrix + pred_matrix.transpose(0, 1))
                group_pairs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
                for row_index, strain in enumerate(strains):
                    strain_embeddings[strain].append(embeddings[batch_index, row_index])
                    for col_index in range(row_index + 1, size):
                        other = strains[col_index]
                        if strain == other:
                            continue
                        group_pairs[pair_key_zwx(strain, other)].append(float(pred_matrix[row_index, col_index]))
                weight = group_weight_zwx(group_weight_mode, len(set(strains)))
                for key, values in group_pairs.items():
                    distance_sums[key] += weight * aggregate_values_zwx(values, within_group_agg)
                    weight_sums[key] += weight
                    evidence_counts[key] += 1
    return sorted(strain_set), distance_sums, weight_sums, dict(evidence_counts), dict(strain_embeddings), group_count


def write_method_outputs_zwx(output_dir: Path, labels: Sequence[str], matrix: torch.Tensor, tree_name: str) -> None:
    # Write the distance matrix and Newick tree for one strain-tree reconstruction method.

    output_dir.mkdir(parents=True, exist_ok=True)
    write_matrix_zwx(output_dir / "strain_distance_matrix.csv", labels, matrix)
    (output_dir / "strain_tree.nwk").write_text(neighbor_joining_zwx(labels, matrix) + "\n", encoding="utf-8")
    (output_dir / "method.txt").write_text(tree_name + "\n", encoding="utf-8")


def write_summary_zwx(path: Path, rows: Sequence[dict]) -> None:
    # Write a brief comparison of the two strain-tree reconstruction methods.

    fields = ["method", "num_strains", "num_groups", "matrix_path", "tree_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main_zwx() -> None:
    # Run both strain-tree reconstruction methods and save the results.

    args = parse_args_zwx()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    loader = build_loader_zwx(args, checkpoint)
    model = build_model_zwx(checkpoint, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    labels, distance_sums, weight_sums, evidence_counts, strain_embeddings, group_count = collect_strain_evidence_zwx(
        model.to(device),
        loader,
        device,
        within_group_agg=args.within_group_agg,
        group_weight_mode=args.group_weight,
    )
    distance_matrix = build_distance_fusion_matrix_zwx(labels, distance_sums, weight_sums)
    embedding_matrix = build_embedding_matrix_zwx(labels, strain_embeddings, args.embedding_distance)
    distance_dir = output_dir / "method_1_distance_fusion"
    embedding_dir = output_dir / "method_2_embedding_fusion"
    write_method_outputs_zwx(distance_dir, labels, distance_matrix, "method_1_distance_fusion")
    write_method_outputs_zwx(embedding_dir, labels, embedding_matrix, "method_2_embedding_fusion")
    write_matrix_zwx(distance_dir / "shared_group_count_matrix.csv", labels, build_evidence_matrix_zwx(labels, evidence_counts))
    write_summary_zwx(
        output_dir / "strain_tree_methods.csv",
        [
            {
                "method": "method_1_distance_fusion",
                "num_strains": len(labels),
                "num_groups": group_count,
                "matrix_path": str(distance_dir / "strain_distance_matrix.csv"),
                "tree_path": str(distance_dir / "strain_tree.nwk"),
            },
            {
                "method": "method_2_embedding_fusion",
                "num_strains": len(labels),
                "num_groups": group_count,
                "matrix_path": str(embedding_dir / "strain_distance_matrix.csv"),
                "tree_path": str(embedding_dir / "strain_tree.nwk"),
            },
        ],
    )
    # print(f"split={args.split}")
    print(f"input_mode={'csv' if args.csv_path else args.split}")
    print(f"num_groups={group_count}")
    print(f"num_strains={len(labels)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    start = time.perf_counter()
    main_zwx()
    end = time.perf_counter()
    print(f"Runtime: {end - start:.6f} seconds")
