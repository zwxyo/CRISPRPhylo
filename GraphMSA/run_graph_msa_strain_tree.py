from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from graph_msa_dataset import ArrayPath, convert_strains_to_array_paths, group_array_paths_by_repeat_family
from path_alignment import AlignmentWeights, build_pairwise_distance_matrix
from project_config import TRAIN_CONFIG
from run_graph_msa_stage1 import (
    build_runtime_config,
    distance_matrix_to_newick,
    read_spacers_from_csv,
    robinson_foulds_distance,
    _pairwise_distances_from_tree,
)
from simulated_data import load_split_strains
from spacer_graph import build_repeat_family_graph, build_spacer_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strain-level trees by aggregating Graph-MSA array distances."
    )
    parser.add_argument(
        "--data-source",
        choices=["simulated", "real"],
        default="real",
        help="Use a simulated cache or a real spacer CSV.",
    )
    parser.add_argument(
        "--data-dir",
        default="simulated_1_3_seed42_benchmark_new/arrays_20",
        help="Simulated data directory containing sim_cache/*.json.",
    )
    parser.add_argument("--csv-path", default="", help="Real CSV path when --data-source real is selected.")
    parser.add_argument("--simulation-cache-file", default="", help="Use this simulated JSON cache directly.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["training", "validation", "test"])
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG["simulation_seed"]))
    parser.add_argument("--simulation-num-strains", type=int, default=0)
    parser.add_argument("--simulation-num-groups", type=int, default=0)
    parser.add_argument("--max-levenshtein-distance", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--tree-method", default="nj", help="NJ or any scipy linkage method accepted by stage1.")
    parser.add_argument("--within-family-agg", choices=["mean", "min"], default="mean")
    parser.add_argument(
        "--family-weight",
        choices=["uniform", "array_count"],
        default="uniform",
        help="Weight shared-family evidence uniformly or by the number of cross-array observations.",
    )
    parser.add_argument(
        "--profile-distance",
        choices=["jaccard", "weighted_jaccard"],
        default="jaccard",
        help="Distance used by method_2_spacer_profile.",
    )
    parser.add_argument(
        "--skip-group-evaluation",
        action="store_true",
        help="Do not write per-simulation-group predicted trees and metrics.",
    )
    return parser.parse_args()


def pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _split_key(split: str) -> str:
    return {"training": "train", "validation": "valid"}.get(split, split)


def _find_simulated_cache(data_dir: str | Path, split: str) -> Path | None:
    path = Path(data_dir)
    if path.is_file():
        return path
    cache_dir = path / "sim_cache"
    candidates = sorted(cache_dir.glob(f"{_split_key(split)}_*.json"))
    return candidates[0] if candidates else None


def _load_data(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict], dict]:
    runtime_config = build_runtime_config(args, output_dir)
    runtime_config["data_source"] = args.data_source
    runtime_config["simulation_seed"] = args.seed

    if args.data_source == "real":
        if not args.csv_path:
            raise ValueError("--csv-path is required when --data-source real is selected.")
        runtime_config[f"{_split_key(args.split)}_csv_path"] = str(Path(args.csv_path))
    else:
        cache_path = Path(args.simulation_cache_file) if args.simulation_cache_file else _find_simulated_cache(args.data_dir, args.split)
        if cache_path is not None:
            runtime_config["simulation_cache_file"] = str(cache_path)
        else:
            runtime_config["simulation_cache_dir"] = str(Path(args.data_dir) / "sim_cache")
            runtime_config["simulation_use_cache"] = True

    strains_data, metadata = load_split_strains(args.split, runtime_config, read_spacers_from_csv)
    return strains_data, metadata


def _limit_groups(strains_data: list[dict], metadata: Mapping[str, object], max_groups: int | None) -> tuple[list[dict], dict]:
    if max_groups is None or int(max_groups) <= 0:
        return strains_data, dict(metadata)
    groups = list(metadata.get("groups", []) or [])[: int(max_groups)]
    if not groups:
        return strains_data, dict(metadata)
    group_ids = {str(group.get("candidate_group_id", "")) for group in groups}
    filtered: list[dict] = []
    for strain in strains_data:
        arrays = [
            array
            for array in strain.get("arrays", [])
            if str(array.get("candidate_group_id", "")) in group_ids
        ]
        if arrays:
            filtered.append({**strain, "arrays": arrays})
    limited_metadata = dict(metadata)
    limited_metadata["groups"] = groups
    return filtered, limited_metadata


def _aggregate(values: Sequence[float], mode: str) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty distance list.")
    return float(min(values) if mode == "min" else sum(values) / len(values))


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 0.0)
    maximum = float(np.max(result)) if result.size else 0.0
    if maximum > 1e-12:
        result /= maximum
    return result


def _build_family_results(
    array_paths: Sequence[ArrayPath],
    max_levenshtein_distance: int,
) -> tuple[dict[str, dict], dict[str, ArrayPath]]:
    grouped_paths = group_array_paths_by_repeat_family(list(array_paths))
    family_results: dict[str, dict] = {}
    array_lookup = {path.array_uid: path for path in array_paths}
    for repeat_family, family_paths in sorted(grouped_paths.items()):
        if not family_paths:
            continue
        annotated_paths, spacer_classes = build_spacer_classes(
            family_paths,
            max_levenshtein_distance=max_levenshtein_distance,
        )
        family_graph = build_repeat_family_graph(annotated_paths, spacer_classes)
        labels, matrix, alignment_results = build_pairwise_distance_matrix(
            annotated_paths,
            family_graph,
            AlignmentWeights(),
            max_levenshtein_distance=max_levenshtein_distance,
        )
        family_results[repeat_family] = {
            "paths": annotated_paths,
            "labels": labels,
            "matrix": matrix,
            "alignment_results": alignment_results,
            "num_spacer_classes": len(spacer_classes),
        }
        array_lookup.update({path.array_uid: path for path in annotated_paths})
    return family_results, array_lookup


def build_distance_fusion_matrix(
    strain_labels: Sequence[str],
    family_results: Mapping[str, Mapping[str, object]],
    array_lookup: Mapping[str, ArrayPath],
    within_family_agg: str = "mean",
    family_weight: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:

    pair_family_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for family_name, result in family_results.items():
        labels = [str(label) for label in result["labels"]]
        matrix = np.asarray(result["matrix"], dtype=np.float64)
        for left_index, left_uid in enumerate(labels):
            left_strain = array_lookup[left_uid].strain_id
            for right_index in range(left_index + 1, len(labels)):
                right_uid = labels[right_index]
                right_strain = array_lookup[right_uid].strain_id
                if left_strain == right_strain:
                    continue
                pair_family_values[pair_key(left_strain, right_strain)][family_name].append(
                    float(matrix[left_index, right_index])
                )

    size = len(strain_labels)
    matrix = np.zeros((size, size), dtype=np.float64)
    evidence = np.zeros((size, size), dtype=np.float64)
    observed: list[float] = []
    for pair_values in pair_family_values.values():
        for values in pair_values.values():
            observed.append(_aggregate(values, within_family_agg))
    fill_value = max(observed) if observed else 0.0

    for left_index, left in enumerate(strain_labels):
        for right_index in range(left_index + 1, size):
            right = strain_labels[right_index]
            family_values = pair_family_values.get(pair_key(left, right), {})
            if family_values:
                weighted_values: list[tuple[float, float]] = []
                for values in family_values.values():
                    value = _aggregate(values, within_family_agg)
                    weight = float(len(values)) if family_weight == "array_count" else 1.0
                    weighted_values.append((value, weight))
                total_weight = sum(weight for _, weight in weighted_values)
                value = sum(distance * weight for distance, weight in weighted_values) / max(total_weight, 1e-12)
                evidence[left_index, right_index] = evidence[right_index, left_index] = float(len(family_values))
            else:
                value = fill_value
            matrix[left_index, right_index] = matrix[right_index, left_index] = value
    return _normalize_matrix(matrix), evidence


def _strain_profiles(strains_data: Sequence[dict]) -> dict[str, Counter[str]]:
    profiles: dict[str, Counter[str]] = {}
    for strain in strains_data:
        strain_id = str(strain["strain_id"])
        features: Counter[str] = Counter()
        for array in strain.get("arrays", []):
            family = str(array.get("repeat_family") or array.get("repeat") or "").strip().upper()
            if family:
                features[f"family\x1f{family}"] += 1
            for spacer in array.get("spacers", []):
                normalized_spacer = str(spacer).strip().upper()
                if normalized_spacer:
                    features[f"spacer\x1f{normalized_spacer}"] += 1
        profiles[strain_id] = features
    return profiles


def build_spacer_profile_matrix(
    strain_labels: Sequence[str],
    profiles: Mapping[str, Mapping[str, int]],
    distance_mode: str = "jaccard",
) -> np.ndarray:

    size = len(strain_labels)
    matrix = np.zeros((size, size), dtype=np.float64)
    for left_index, left in enumerate(strain_labels):
        left_profile = profiles.get(left, {})
        for right_index in range(left_index + 1, size):
            right = strain_labels[right_index]
            right_profile = profiles.get(right, {})
            feature_names = set(left_profile) | set(right_profile)
            if distance_mode == "weighted_jaccard":
                intersection_weight = sum(min(left_profile.get(name, 0), right_profile.get(name, 0)) for name in feature_names)
                union_weight = sum(max(left_profile.get(name, 0), right_profile.get(name, 0)) for name in feature_names)
            else:
                intersection_weight = sum(1 for name in feature_names if name in left_profile and name in right_profile)
                union_weight = len(feature_names)
            similarity = intersection_weight / max(1, union_weight)
            matrix[left_index, right_index] = matrix[right_index, left_index] = 1.0 - similarity
    return _normalize_matrix(matrix)


def _safe_label(label: str) -> str:
    return re.sub(r"[\s,:;()]+", "_", str(label))


def _write_matrix(path: Path, labels: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.8f}" for value in row]])


def _write_method_output(
    output_dir: Path,
    labels: Sequence[str],
    matrix: np.ndarray,
    method_name: str,
    tree_method: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_matrix(output_dir / "strain_distance_matrix.csv", labels, matrix)
    tree_labels = [_safe_label(label) for label in labels]
    tree = distance_matrix_to_newick(matrix, tree_labels, tree_method=tree_method)
    (output_dir / "strain_tree.nwk").write_text(tree + "\n", encoding="utf-8")
    (output_dir / "method.txt").write_text(method_name + "\n", encoding="utf-8")


def _write_group_metrics(
    output_dir: Path,
    metadata: Mapping[str, object],
    labels: Sequence[str],
    matrices: Mapping[str, np.ndarray],
    tree_method: str,
) -> list[dict]:
    label_index = {label: index for index, label in enumerate(labels)}
    rows: list[dict] = []
    for group in metadata.get("groups", []) or []:
        group_id = str(group.get("candidate_group_id", ""))
        group_labels = list(dict.fromkeys(str(value) for value in group.get("strain_ids", []) if str(value) in label_index))
        if len(group_labels) < 2:
            continue
        true_newick = str(group.get("newick", "")).strip()
        if not true_newick:
            continue
        group_dir = output_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        true_matrix = _pairwise_distances_from_tree(true_newick, group_labels)
        for method_name, full_matrix in matrices.items():
            indices = [label_index[label] for label in group_labels]
            predicted_matrix = full_matrix[np.ix_(indices, indices)]
            predicted_tree = distance_matrix_to_newick(predicted_matrix, [_safe_label(label) for label in group_labels], tree_method=tree_method)
            predicted_path = group_dir / f"{method_name}.nwk"
            predicted_path.write_text(predicted_tree + "\n", encoding="utf-8")
            _write_matrix(group_dir / f"{method_name}_distance_matrix.csv", group_labels, predicted_matrix)
            metric = {"candidate_group_id": group_id, "method": method_name, "num_strains": len(group_labels)}
            try:
                rf_distance, normalized_rf = robinson_foulds_distance(true_newick, predicted_tree)
                true_values = true_matrix[np.triu_indices(len(group_labels), 1)]
                predicted_values = predicted_matrix[np.triu_indices(len(group_labels), 1)]
                if np.std(predicted_values) <= 1e-12 or np.std(true_values) <= 1e-12:
                    spearman = 0.0
                else:
                    spearman = float(np.corrcoef(np.argsort(np.argsort(predicted_values)), np.argsort(np.argsort(true_values)))[0, 1])
                metric.update({"spearman_correlation": spearman, "rf_distance": rf_distance, "normalized_rf": normalized_rf})
            except (ValueError, KeyError):
                metric.update({"spearman_correlation": "", "rf_distance": "", "normalized_rf": ""})
            rows.append(metric)
    if rows:
        with (output_dir / "group_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["candidate_group_id", "method", "num_strains", "spearman_correlation", "rf_distance", "normalized_rf"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strains_data, metadata = _load_data(args, output_dir)
    strains_data, metadata = _limit_groups(strains_data, metadata, args.max_groups)
    array_paths = convert_strains_to_array_paths(strains_data)
    if not array_paths:
        raise ValueError("No arrays were loaded from the selected input.")

    family_results, array_lookup = _build_family_results(array_paths, args.max_levenshtein_distance)
    strain_labels = sorted({path.strain_id for path in array_paths})
    distance_matrix, evidence_matrix = build_distance_fusion_matrix(
        strain_labels,
        family_results,
        array_lookup,
        within_family_agg=args.within_family_agg,
        family_weight=args.family_weight,
    )
    profile_matrix = build_spacer_profile_matrix(
        strain_labels,
        _strain_profiles(strains_data),
        distance_mode=args.profile_distance,
    )

    method_dirs = {
        "method_1_distance_fusion": output_dir / "method_1_distance_fusion",
        "method_2_spacer_profile": output_dir / "method_2_spacer_profile",
    }
    _write_method_output(
        method_dirs["method_1_distance_fusion"],
        strain_labels,
        distance_matrix,
        "method_1_distance_fusion",
        args.tree_method,
    )
    _write_method_output(
        method_dirs["method_2_spacer_profile"],
        strain_labels,
        profile_matrix,
        "method_2_spacer_profile",
        args.tree_method,
    )
    _write_matrix(method_dirs["method_1_distance_fusion"] / "shared_family_count_matrix.csv", strain_labels, evidence_matrix)

    summary_rows = []
    for method_name, method_dir in method_dirs.items():
        summary_rows.append({
            "method": method_name,
            "num_strains": len(strain_labels),
            "num_arrays": len(array_paths),
            "num_repeat_families": len(family_results),
            "matrix_path": str(method_dir / "strain_distance_matrix.csv"),
            "tree_path": str(method_dir / "strain_tree.nwk"),
        })
    with (output_dir / "strain_tree_methods.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["method", "num_strains", "num_arrays", "num_repeat_families", "matrix_path", "tree_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_matrix(output_dir / "shared_family_count_matrix.csv", strain_labels, evidence_matrix)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    group_rows = []
    if not args.skip_group_evaluation:
        group_rows = _write_group_metrics(
            output_dir / "group_trees",
            metadata,
            strain_labels,
            {
                "method_1_distance_fusion": distance_matrix,
                "method_2_spacer_profile": profile_matrix,
            },
            args.tree_method,
        )

    print(f"input_mode={args.data_source}")
    print(f"num_strains={len(strain_labels)}")
    print(f"num_arrays={len(array_paths)}")
    print(f"num_repeat_families={len(family_results)}")
    evaluated_groups = len({str(row["candidate_group_id"]) for row in group_rows})
    print(f"num_groups_evaluated={evaluated_groups}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Runtime: {time.perf_counter() - start:.6f} seconds")
