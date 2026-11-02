from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from tree_utils_zwx import pairwise_tree_distance_matrix_zwx, topology_distribution_matrix_zwx


DNA_TO_INDEX_ZWX = {"A": 0, "C": 1, "G": 2, "T": 3}
REPEAT_TOKEN_TO_INDEX_ZWX = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}


@dataclass
class CRISPRGroupSampleZwx:
    # Store model inputs and supervision labels for a repeat/tree group.

    group_id: str
    repeat: str
    labels: List[str]
    strain_labels: List[str]
    newick: str
    msa_vectors: torch.Tensor
    msa_mask: torch.Tensor
    row_mask: torch.Tensor
    col_mask: torch.Tensor
    conservation_weights: torch.Tensor
    repeat_tokens: torch.Tensor
    repeat_mask: torch.Tensor
    target_distance: torch.Tensor
    target_topology: torch.Tensor


def normalize_repeat_zwx(repeat: str) -> str:
    # Normalize repeat sequences, using a fixed placeholder for empty values.

    repeat = str(repeat or "").strip().upper()
    return repeat if repeat else "UNKNOWN_REPEAT"


def levenshtein_distance_zwx(left: str, right: str) -> int:
    # Compute the Levenshtein edit distance between two repeat sequences.

    left = normalize_repeat_zwx(left)
    right = normalize_repeat_zwx(right)
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def group_arrays_by_repeat_zwx(
    arrays: Sequence[dict],
    max_repeat_distance: int = 0,
) -> Dict[str, List[dict]]:
    # Group repeats by edit distance; by default, a single-base difference places them in separate groups.

    groups: List[dict] = []
    for array in arrays:
        repeat = normalize_repeat_zwx(array.get("repeat", ""))
        matched_index = None
        for index, group in enumerate(groups):
            if levenshtein_distance_zwx(repeat, group["repeat"]) <= int(max_repeat_distance):
                matched_index = index
                break
        if matched_index is None:
            groups.append({"repeat": repeat, "arrays": [array]})
        else:
            groups[matched_index]["arrays"].append(array)
    return {f"repeat_group_{index:04d}": group["arrays"] for index, group in enumerate(groups, start=1)}


def read_spacers_from_csv(
    csv_path: str,
    require_repeat_columns: bool = False,
) -> List[Dict]:
    # Read real-data CSV files and organize arrays by strain using the project's common array structure.

    del require_repeat_columns
    grouped = defaultdict(lambda: defaultdict(lambda: {"spacers": [], "repeat": ""}))
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strain_id = row["Folder"].strip()
            array_id = row["Seq_Name"].strip()
            orientation = row.get("Orientation", "+").strip()
            spacer = row["Spacer_Sequence"].strip() if orientation == "+" else row["Reverse_Complement_Sequence"].strip()
            repeat = row.get("DR_Consensus", "").strip() if orientation == "+" else row.get("Reverse_Complement_DR", "").strip()
            entry = grouped[strain_id][array_id]
            entry["spacers"].append(spacer.upper())
            if repeat and not entry["repeat"]:
                entry["repeat"] = repeat

    strains = []
    for strain_id, arrays in grouped.items():
        array_rows = []
        for array_id, payload in arrays.items():
            if payload["spacers"]:
                array_rows.append(
                    {
                        "array_id": array_id,
                        "array_uid": f"{strain_id}::{array_id}",
                        "repeat": normalize_repeat_zwx(payload["repeat"]),
                        "spacers": payload["spacers"],
                    }
                )
        if array_rows:
            strains.append({"strain_id": strain_id, "arrays": array_rows})
    return strains


def spacer_vector_dim_zwx(kmer_size: int = 4) -> int:
    # Return the dimension of the spacer k-mer vector with statistical features appended.

    return 4 ** int(kmer_size) + 4


def spacer_to_vector_zwx(spacer: str, kmer_size: int = 4) -> torch.Tensor:
    # Encode a DNA spacer sequence as a normalized k-mer frequency vector.

    spacer = str(spacer or "").strip().upper()
    kmer_size = int(kmer_size)
    dim = spacer_vector_dim_zwx(kmer_size)
    vector = torch.zeros(dim, dtype=torch.float32)
    valid_bases = [base for base in spacer if base in DNA_TO_INDEX_ZWX]
    for start in range(0, max(0, len(spacer) - kmer_size + 1)):
        kmer = spacer[start:start + kmer_size]
        if any(base not in DNA_TO_INDEX_ZWX for base in kmer):
            continue
        index = 0
        for base in kmer:
            index = index * 4 + DNA_TO_INDEX_ZWX[base]
        vector[index] += 1.0
    kmer_total = vector[: 4 ** kmer_size].sum().clamp_min(1.0)
    vector[: 4 ** kmer_size] = vector[: 4 ** kmer_size] / kmer_total
    length = max(1, len(spacer))
    gc_count = sum(1 for base in valid_bases if base in {"G", "C"})
    at_count = sum(1 for base in valid_bases if base in {"A", "T"})
    vector[-4:] = torch.tensor(
        [
            min(length / 100.0, 1.0),
            gc_count / length,
            at_count / length,
            len(valid_bases) / length,
        ],
        dtype=torch.float32,
    )
    return vector


def repeat_to_tokens_zwx(repeat: str) -> torch.Tensor:
    # Encode repeat DNA sequences as character tokens for the Transformer.

    repeat = normalize_repeat_zwx(repeat)
    tokens = [REPEAT_TOKEN_TO_INDEX_ZWX.get(base, REPEAT_TOKEN_TO_INDEX_ZWX["N"]) for base in repeat]
    return torch.tensor(tokens, dtype=torch.long)


def build_msa_from_arrays_zwx(
    arrays: Sequence[dict],
    kmer_size: int = 4,
    alignment_threshold: float = 0.18,
    max_msa_cols: int | None = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Greedily align arrays using Euclidean distances in spacer embedding space to generate an MSA matrix.

    rows = len(arrays)
    spacer_rows = [[spacer_to_vector_zwx(spacer, kmer_size) for spacer in array.get("spacers", [])] for array in arrays]
    dim = spacer_vector_dim_zwx(kmer_size)
    centroids: List[torch.Tensor] = []
    counts: List[int] = []
    positions: List[List[int]] = []
    assignments: List[Dict[int, torch.Tensor]] = []

    for row_vectors in spacer_rows:
        row_assignment: Dict[int, torch.Tensor] = {}
        used_columns: set[int] = set()
        for position, vector in enumerate(row_vectors):
            best_index = None
            best_distance = float("inf")
            for column_index, centroid in enumerate(centroids):
                if column_index in used_columns:
                    continue
                distance = torch.dist(vector, centroid).item()
                if distance < best_distance:
                    best_distance = distance
                    best_index = column_index
            if best_index is None or best_distance > float(alignment_threshold):
                best_index = len(centroids)
                centroids.append(vector.clone())
                counts.append(0)
                positions.append([])
            counts[best_index] += 1
            centroids[best_index] += (vector - centroids[best_index]) / counts[best_index]
            positions[best_index].append(position)
            row_assignment[best_index] = vector
            used_columns.add(best_index)
        assignments.append(row_assignment)

    column_order = sorted(range(len(centroids)), key=lambda index: sum(positions[index]) / max(1, len(positions[index])))
    if max_msa_cols is not None and int(max_msa_cols) > 0 and len(column_order) > int(max_msa_cols):
        keep = set(
            sorted(
                column_order,
                key=lambda index: (
                    -counts[index],
                    sum(positions[index]) / max(1, len(positions[index])),
                ),
            )[: int(max_msa_cols)]
        )
        column_order = [index for index in column_order if index in keep]
    old_to_new = {old: new for new, old in enumerate(column_order)}
    cols = max(1, len(column_order))
    msa_vectors = torch.zeros(rows, cols, dim, dtype=torch.float32)
    msa_mask = torch.zeros(rows, cols, dtype=torch.bool)
    for row_index, row_assignment in enumerate(assignments):
        for old_column, vector in row_assignment.items():
            if old_column not in old_to_new:
                continue
            new_column = old_to_new[old_column]
            msa_vectors[row_index, new_column] = vector
            msa_mask[row_index, new_column] = True
    col_mask = msa_mask.any(dim=0)
    conservation = msa_mask.sum(dim=0).to(torch.float32) / max(1, rows)
    return msa_vectors, msa_mask, col_mask, conservation


def _collect_arrays_by_group_zwx(strains_data: Sequence[dict]) -> Dict[str, Dict[str, dict]]:
    # Collect arrays from the simulation cache by candidate_group_id.

    grouped: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for strain in strains_data:
        for array in strain.get("arrays", []):
            group_id = str(array.get("candidate_group_id", ""))
            if group_id:
                grouped[group_id][str(array["array_uid"])] = array
    return dict(grouped)


def find_simulated_cache_path_zwx(data_dir: str | Path, split: str) -> Path:
    # Find cached JSON files for the specified split in the simulated data directory.

    split_key = {"training": "train", "validation": "valid"}.get(split, split)
    candidates = sorted((Path(data_dir) / "sim_cache").glob(f"{split_key}_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No simulation cache files found for {split_key}: {Path(data_dir) / 'sim_cache'}")
    return candidates[0]


class CRISPRTreeDatasetZwx(Dataset):
    # Convert cached simulated CRISPR arrays into Axial Transformer training samples.

    def __init__(
        self,
        cache_path: str | Path,
        kmer_size: int = 4,
        alignment_threshold: float = 0.18,
        max_groups: int | None = None,
        max_msa_cols: int | None = 512,
    ):
        self.cache_path = Path(cache_path)
        self.kmer_size = int(kmer_size)
        self.alignment_threshold = float(alignment_threshold)
        self.max_msa_cols = max_msa_cols if max_msa_cols is None else int(max_msa_cols)
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.arrays_by_group = _collect_arrays_by_group_zwx(payload["strains_data"])
        self.group_records = self._build_group_records(payload, max_groups=max_groups)

    def _build_group_records(self, payload: dict, max_groups: int | None) -> List[dict]:
        # Store a lightweight group index to avoid building all MSAs and supervision matrices during initialization.

        records: List[dict] = []
        for group in payload.get("metadata", {}).get("groups", []):
            group_id = str(group["candidate_group_id"])
            array_lookup = self.arrays_by_group.get(group_id, {})
            ordered_uids = [str(uid) for uid in group.get("array_leaf_labels", []) if str(uid) in array_lookup]
            if len(ordered_uids) < 3:
                continue
            records.append(
                {
                    "group_id": group_id,
                    "ordered_uids": ordered_uids,
                    "newick": str(group["newick"]),
                    "repeat": str(group.get("repeat", "")),
                }
            )
            if max_groups is not None and len(records) >= int(max_groups):
                break
        return records

    def _build_sample_from_record(self, record: dict) -> CRISPRGroupSampleZwx:
        group_id = str(record["group_id"])
        array_lookup = self.arrays_by_group[group_id]
        arrays = [array_lookup[uid] for uid in record["ordered_uids"]]
        labels = [str(array["array_uid"]) for array in arrays]
        strain_labels = [label.split("::", 1)[0] for label in labels]
        newick = str(record["newick"])
        msa_vectors, msa_mask, col_mask, conservation = build_msa_from_arrays_zwx(
            arrays,
            kmer_size=self.kmer_size,
            alignment_threshold=self.alignment_threshold,
            max_msa_cols=self.max_msa_cols,
        )
        repeat = normalize_repeat_zwx(arrays[0].get("repeat", record.get("repeat", "")))
        repeat_tokens = repeat_to_tokens_zwx(repeat)
        target_distance = pairwise_tree_distance_matrix_zwx(newick, strain_labels)
        target_topology = topology_distribution_matrix_zwx(newick, strain_labels)
        return CRISPRGroupSampleZwx(
            group_id=group_id,
            repeat=repeat,
            labels=labels,
            strain_labels=strain_labels,
            newick=newick,
            msa_vectors=msa_vectors,
            msa_mask=msa_mask,
            row_mask=torch.ones(len(arrays), dtype=torch.bool),
            col_mask=col_mask,
            conservation_weights=conservation,
            repeat_tokens=repeat_tokens,
            repeat_mask=torch.ones_like(repeat_tokens, dtype=torch.bool),
            target_distance=target_distance,
            target_topology=target_topology,
        )

    def __len__(self) -> int:
        # Return the number of trainable group samples.

        return len(self.group_records)

    def __getitem__(self, index: int) -> CRISPRGroupSampleZwx:
        # Return a CRISPR group sample by index.

        return self._build_sample_from_record(self.group_records[index])


def collate_crispr_tree_samples_zwx(samples: Sequence[CRISPRGroupSampleZwx]) -> Dict[str, object]:
    # Pad variable-length MSAs, repeats, and supervision matrices into a batch.

    batch_size = len(samples)
    max_rows = max(sample.msa_vectors.size(0) for sample in samples)
    max_cols = max(sample.msa_vectors.size(1) for sample in samples)
    spacer_dim = samples[0].msa_vectors.size(-1)
    max_repeat_len = max(sample.repeat_tokens.numel() for sample in samples)
    msa_vectors = torch.zeros(batch_size, max_rows, max_cols, spacer_dim, dtype=torch.float32)
    msa_mask = torch.zeros(batch_size, max_rows, max_cols, dtype=torch.bool)
    row_mask = torch.zeros(batch_size, max_rows, dtype=torch.bool)
    col_mask = torch.zeros(batch_size, max_cols, dtype=torch.bool)
    conservation = torch.zeros(batch_size, max_cols, dtype=torch.float32)
    repeat_tokens = torch.zeros(batch_size, max_repeat_len, dtype=torch.long)
    repeat_mask = torch.zeros(batch_size, max_repeat_len, dtype=torch.bool)
    target_distance = torch.zeros(batch_size, max_rows, max_rows, dtype=torch.float32)
    target_topology = torch.zeros(batch_size, max_rows, max_rows, dtype=torch.float32)

    for batch_index, sample in enumerate(samples):
        rows, cols, _ = sample.msa_vectors.shape
        repeat_len = sample.repeat_tokens.numel()
        msa_vectors[batch_index, :rows, :cols] = sample.msa_vectors
        msa_mask[batch_index, :rows, :cols] = sample.msa_mask
        row_mask[batch_index, :rows] = sample.row_mask
        col_mask[batch_index, :cols] = sample.col_mask
        conservation[batch_index, :cols] = sample.conservation_weights
        repeat_tokens[batch_index, :repeat_len] = sample.repeat_tokens
        repeat_mask[batch_index, :repeat_len] = sample.repeat_mask
        target_distance[batch_index, :rows, :rows] = sample.target_distance
        target_topology[batch_index, :rows, :rows] = sample.target_topology

    return {
        "msa_vectors": msa_vectors,
        "msa_mask": msa_mask,
        "row_mask": row_mask,
        "col_mask": col_mask,
        "conservation_weights": conservation,
        "repeat_tokens": repeat_tokens,
        "repeat_mask": repeat_mask,
        "target_distance": target_distance,
        "target_topology": target_topology,
        "labels": [sample.labels for sample in samples],
        "strain_labels": [sample.strain_labels for sample in samples],
        "newick": [sample.newick for sample in samples],
        "group_id": [sample.group_id for sample in samples],
    }


def build_dataset_from_simulated_dir_zwx(
    data_dir: str | Path,
    split: str,
    kmer_size: int = 4,
    alignment_threshold: float = 0.18,
    max_groups: int | None = None,
    max_msa_cols: int | None = 512,
) -> CRISPRTreeDatasetZwx:
    # Quickly build the dataset for the specified split from the simulated_data_theta directory.

    cache_path = find_simulated_cache_path_zwx(data_dir, split)
    return CRISPRTreeDatasetZwx(
        cache_path=cache_path,
        kmer_size=kmer_size,
        alignment_threshold=alignment_threshold,
        max_groups=max_groups,
        max_msa_cols=max_msa_cols,
    )


class CRISPRCsvTreeDatasetZwx(Dataset):
    # Convert real-data CSV files into inference samples without supervision labels.

    def __init__(
        self,
        csv_path: str | Path,
        kmer_size: int = 4,
        alignment_threshold: float = 0.18,
        max_groups: int | None = None,
        max_repeat_distance: int = 0,
        min_arrays: int = 3,
        min_strains: int = 2,
    ):
        self.csv_path = Path(csv_path)
        self.kmer_size = int(kmer_size)
        self.alignment_threshold = float(alignment_threshold)
        strains = read_spacers_from_csv(str(self.csv_path))
        arrays = [array for strain in strains for array in strain.get("arrays", [])]
        groups = group_arrays_by_repeat_zwx(arrays, max_repeat_distance=max_repeat_distance)
        self.samples = self._build_samples(groups, max_groups, min_arrays, min_strains)

    def _build_samples(
        self,
        groups: Dict[str, List[dict]],
        max_groups: int | None,
        min_arrays: int,
        min_strains: int,
    ) -> List[CRISPRGroupSampleZwx]:
        samples = []
        for group_id, arrays in groups.items():
            arrays = sorted(arrays, key=lambda item: str(item["array_uid"]))
            labels = [str(array["array_uid"]) for array in arrays]
            strain_labels = [label.split("::", 1)[0] for label in labels]
            if len(arrays) < int(min_arrays) or len(set(strain_labels)) < int(min_strains):
                continue

            msa_vectors, msa_mask, col_mask, conservation = build_msa_from_arrays_zwx(
                arrays,
                kmer_size=self.kmer_size,
                alignment_threshold=self.alignment_threshold,
            )
            repeat = normalize_repeat_zwx(arrays[0].get("repeat", ""))
            repeat_tokens = repeat_to_tokens_zwx(repeat)
            size = len(arrays)
            zero_target = torch.zeros(size, size, dtype=torch.float32)

            samples.append(
                CRISPRGroupSampleZwx(
                    group_id=group_id,
                    repeat=repeat,
                    labels=labels,
                    strain_labels=strain_labels,
                    newick="",
                    msa_vectors=msa_vectors,
                    msa_mask=msa_mask,
                    row_mask=torch.ones(size, dtype=torch.bool),
                    col_mask=col_mask,
                    conservation_weights=conservation,
                    repeat_tokens=repeat_tokens,
                    repeat_mask=torch.ones_like(repeat_tokens, dtype=torch.bool),
                    target_distance=zero_target,
                    target_topology=zero_target.clone(),
                )
            )
            if max_groups is not None and len(samples) >= int(max_groups):
                break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CRISPRGroupSampleZwx:
        return self.samples[index]


# def build_dataset_from_csv_zwx(
#     csv_path: str | Path,
#     kmer_size: int = 4,
#     alignment_threshold: float = 0.18,
#     max_groups: int | None = None,
#     max_repeat_distance: int = 0,
#     min_arrays: int = 3,
#     min_strains: int = 2,
# ) -> CRISPRCsvTreeDatasetZwx:
#     return CRISPRCsvTreeDatasetZwx(
#         csv_path=csv_path,
#         kmer_size=kmer_size,
#         alignment_threshold=alignment_threshold,
#         max_groups=max_groups,
#         max_repeat_distance=max_repeat_distance,
#         min_arrays=min_arrays,
#         min_strains=min_strains,
#     )


class CRISPRListDatasetZwx(Dataset):
    # Wrap a list of prebuilt samples as a Dataset.

    def __init__(self, samples: Sequence[CRISPRGroupSampleZwx]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CRISPRGroupSampleZwx:
        return self.samples[index]


def build_dataset_from_csv_zwx(
    csv_path: str | Path,
    kmer_size: int = 4,
    alignment_threshold: float = 0.18,
    max_groups: int | None = None,
    max_repeat_distance: int = 0,
) -> CRISPRListDatasetZwx:
    # Build an inference dataset directly from real-data CSV files, grouping arrays by repeat.

    strains = read_spacers_from_csv(str(csv_path))
    arrays = [array for strain in strains for array in strain.get("arrays", [])]
    grouped = group_arrays_by_repeat_zwx(arrays, max_repeat_distance=max_repeat_distance)

    samples: List[CRISPRGroupSampleZwx] = []
    for group_id, group_arrays in grouped.items():
        if len(group_arrays) < 3:
            continue

        labels = [str(array["array_uid"]) for array in group_arrays]
        strain_labels = [label.split("::", 1)[0] for label in labels]
        msa_vectors, msa_mask, col_mask, conservation = build_msa_from_arrays_zwx(
            group_arrays,
            kmer_size=kmer_size,
            alignment_threshold=alignment_threshold,
        )
        repeat = normalize_repeat_zwx(group_arrays[0].get("repeat", ""))
        repeat_tokens = repeat_to_tokens_zwx(repeat)
        size = len(group_arrays)
        empty_target = torch.zeros(size, size, dtype=torch.float32)

        samples.append(
            CRISPRGroupSampleZwx(
                group_id=group_id,
                repeat=repeat,
                labels=labels,
                strain_labels=strain_labels,
                newick="",
                msa_vectors=msa_vectors,
                msa_mask=msa_mask,
                row_mask=torch.ones(size, dtype=torch.bool),
                col_mask=col_mask,
                conservation_weights=conservation,
                repeat_tokens=repeat_tokens,
                repeat_mask=torch.ones_like(repeat_tokens, dtype=torch.bool),
                target_distance=empty_target,
                target_topology=empty_target.clone(),
            )
        )

        if max_groups is not None and len(samples) >= int(max_groups):
            break

    return CRISPRListDatasetZwx(samples)


__all__ = [
    "CRISPRGroupSampleZwx",
    "CRISPRTreeDatasetZwx",
    "build_dataset_from_simulated_dir_zwx",
    "build_msa_from_arrays_zwx",
    "collate_crispr_tree_samples_zwx",
    "group_arrays_by_repeat_zwx",
    "levenshtein_distance_zwx",
    "normalize_repeat_zwx",
    "read_spacers_from_csv",
    "repeat_to_tokens_zwx",
    "spacer_to_vector_zwx",
    "spacer_vector_dim_zwx",
    "CRISPRListDatasetZwx",
    "build_dataset_from_csv_zwx",
]
