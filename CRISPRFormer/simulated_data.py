from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable
import argparse
from copy import deepcopy

from crispr_data_pipeline_zwx import read_spacers_from_csv
from project_config_copy import TRAIN_CONFIG


DNA_ALPHABET = ("A", "C", "G", "T")
SPLIT_TO_CONFIG_KEY = {
    "training": "train",
    "validation": "valid",
    "test": "test",
}


def _safe_mean(values: list[float], default: float) -> float:
    if not values:
        return float(default)
    return float(sum(values) / max(1, len(values)))


def _sample_tree(leaves: list[str], rng: random.Random, tree_type: str = "coalescent") -> dict:
    if len(leaves) == 1:
        return {"name": leaves[0]}
    if str(tree_type).strip().lower() == "coalescent":
        return _sample_coalescent_tree(leaves, rng)
    working = [{"name": leaf} for leaf in leaves]
    while len(working) > 1:
        left_index, right_index = sorted(rng.sample(range(len(working)), 2), reverse=True)
        left = working.pop(left_index)
        right = working.pop(right_index)
        working.append(
            {
                "left": left,
                "right": right,
                "left_length": round(rng.uniform(0.05, 0.35), 4),
                "right_length": round(rng.uniform(0.05, 0.35), 4),
            }
        )
    return working[0]


def _sample_coalescent_tree(leaves: list[str], rng: random.Random) -> dict:
    if len(leaves) == 1:
        return {"name": leaves[0]}
    shuffled = list(leaves)
    rng.shuffle(shuffled)

    def build(local_leaves: list[str], depth: int) -> dict:
        if len(local_leaves) == 1:
            return {"name": local_leaves[0]}
        min_left = 1
        max_left = len(local_leaves) - 1
        left_size = rng.randint(min_left, max_left)
        left_leaves = local_leaves[:left_size]
        right_leaves = local_leaves[left_size:]
        if not left_leaves or not right_leaves:
            midpoint = max(1, len(local_leaves) // 2)
            left_leaves = local_leaves[:midpoint]
            right_leaves = local_leaves[midpoint:]
        size_penalty = max(1.0, math.log2(len(local_leaves) + 1.0))
        base_scale = 0.10 + 0.18 * depth
        left_length = round(min(1.5, rng.expovariate(size_penalty) + base_scale), 4)
        right_length = round(min(1.5, rng.expovariate(size_penalty) + base_scale), 4)
        return {
            "left": build(left_leaves, depth + 1),
            "right": build(right_leaves, depth + 1),
            "left_length": left_length,
            "right_length": right_length,
        }

    return build(shuffled, depth=0)


def _tree_to_newick(node: dict) -> str:
    if "name" in node:
        return str(node["name"])
    left = _tree_to_newick(node["left"])
    right = _tree_to_newick(node["right"])
    left_length = float(node.get("left_length", 0.1))
    right_length = float(node.get("right_length", 0.1))
    return f"({left}:{left_length:.4f},{right}:{right_length:.4f})"


def _count_internal_nodes(node: dict) -> int:
    if "name" in node:
        return 0
    return 1 + _count_internal_nodes(node["left"]) + _count_internal_nodes(node["right"])


def _sample_from_distribution(values: list[int], rng: random.Random, default: int) -> int:
    if not values:
        return int(default)
    return int(rng.choice(values))


def _sample_poisson(lam: float, rng: random.Random) -> int:
    lam = max(0.0, float(lam))
    if lam <= 0.0:
        return 0
    if lam > 40.0:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    threshold = math.exp(-lam)
    count = 0
    product = 1.0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _sample_weighted_label(weights: dict[str, float], rng: random.Random, default: str) -> str:
    filtered = [(str(label), float(weight)) for label, weight in weights.items() if float(weight) > 0.0]
    if not filtered:
        return str(default)
    total = sum(weight for _, weight in filtered)
    threshold = rng.random() * total
    cumulative = 0.0
    for label, weight in filtered:
        cumulative += weight
        if cumulative >= threshold:
            return label
    return filtered[-1][0]


def _sample_float_range(min_value: float, max_value: float, rng: random.Random, default: float) -> float:
    low = float(min_value)
    high = float(max_value)
    if high < low:
        low, high = high, low
    if math.isclose(low, high):
        return low
    return rng.uniform(low, high)


def _sample_weighted_int(
    weights: dict[int, float],
    rng: random.Random,
    default: int,
) -> int:
    filtered = [
        (int(label), float(weight))
        for label, weight in weights.items()
        if int(label) > 0 and float(weight) > 0.0
    ]
    if not filtered:
        return int(default)
    total = sum(weight for _, weight in filtered)
    threshold = rng.random() * total
    cumulative = 0.0
    for label, weight in filtered:
        cumulative += weight
        if cumulative >= threshold:
            return int(label)
    return int(filtered[-1][0])


def _resolve_spacer_length_weights(priors: dict, config: dict) -> dict[int, float]:
    explicit_histogram = dict(config.get("simulation_spacer_length_histogram", {}) or {})
    if explicit_histogram:
        return {
            int(length): float(count)
            for length, count in explicit_histogram.items()
            if int(length) > 0 and float(count) > 0.0
        }

    prior_histogram = dict(priors.get("spacer_length_histogram", {}) or {})
    if prior_histogram:
        return {
            int(length): float(count)
            for length, count in prior_histogram.items()
            if int(length) > 0 and float(count) > 0.0
        }

    return {
        int(length): 1.0
        for length in priors.get("spacer_lengths", []) or [28, 29, 30, 31, 32]
        if int(length) > 0
    }


def _resolve_repeat_length_weights(priors: dict, config: dict) -> dict[int, float]:
    explicit_histogram = dict(config.get("simulation_repeat_length_histogram", {}) or {})
    if explicit_histogram:
        return {
            int(length): float(count)
            for length, count in explicit_histogram.items()
            if int(length) > 0 and float(count) > 0.0
        }

    prior_histogram = dict(priors.get("repeat_length_histogram", {}) or {})
    if prior_histogram:
        return {
            int(length): float(count)
            for length, count in prior_histogram.items()
            if int(length) > 0 and float(count) > 0.0
        }

    fallback_sequences = [
        str(sequence).strip().upper()
        for sequence in priors.get("repeat_sequences", [])
        if str(sequence).strip()
    ]
    if fallback_sequences:
        histogram = Counter(len(sequence) for sequence in fallback_sequences if len(sequence) > 0)
        if histogram:
            return {int(length): float(count) for length, count in histogram.items()}

    return {28: 1.0, 29: 1.0, 30: 1.0, 31: 1.0, 32: 1.0}


def _build_repeat_sequence_index(priors: dict) -> dict[int, list[str]]:
    index = defaultdict(list)
    for raw_sequence in priors.get("repeat_sequences", []):
        sequence = str(raw_sequence).strip().upper()
        if not sequence:
            continue
        index[len(sequence)].append(sequence)
    return dict(index)


def _random_repeat_sequence(length: int, rng: random.Random) -> str:
    length = max(12, int(length))
    return "".join(rng.choice(DNA_ALPHABET) for _ in range(length))


def _sample_repeat_sequence(priors: dict, config: dict, rng: random.Random) -> str:
    repeat_length_weights = _resolve_repeat_length_weights(priors, config)
    repeat_length = _sample_weighted_int(repeat_length_weights, rng, default=30)
    repeat_index = _build_repeat_sequence_index(priors)
    candidates = list(repeat_index.get(repeat_length, []))
    if candidates:
        return rng.choice(candidates)
    return _random_repeat_sequence(repeat_length, rng)


def _random_spacer(
    rng: random.Random,
    length_weights: dict[int, float],
    used_spacers: set[str] | None = None,
    *,
    allow_existing: bool = False,
) -> str:
    length = max(12, _sample_weighted_int(length_weights, rng, default=30))
    used_spacers = used_spacers if used_spacers is not None else set()
    max_attempts = 32 if used_spacers and not allow_existing else 1
    for _ in range(max_attempts):
        candidate = "".join(rng.choice(DNA_ALPHABET) for _ in range(length))
        if allow_existing or candidate not in used_spacers:
            used_spacers.add(candidate)
            return candidate
    while True:
        candidate = "".join(rng.choice(DNA_ALPHABET) for _ in range(length))
        if allow_existing or candidate not in used_spacers:
            used_spacers.add(candidate)
            return candidate


def _weighted_profile(spacers: list[str]) -> dict[str, float]:
    total = max(1, len(spacers))
    return {
        spacer: max((total - index) / total, 1e-6)
        for index, spacer in enumerate(spacers)
    }


def _array_feature_vector(spacers: list[str]) -> list[float]:
    unique_spacers = len(set(spacers))
    top_leader = len(set(spacers[:3]))
    return [
        float(len(spacers)),
        float(unique_spacers / max(1, len(spacers))),
        float(top_leader / max(1, min(len(spacers), 3))),
        float(math.log1p(len(spacers))),
    ]


def _estimate_empirical_priors(empirical_strains: list[dict]) -> dict:
    spacer_lengths = []
    spacer_length_histogram = Counter()
    repeat_length_histogram = Counter()
    repeat_sequences = []
    arrays_per_strain = []
    array_lengths = []
    for strain in empirical_strains:
        arrays = list(strain.get("arrays", []))
        arrays_per_strain.append(len(arrays))
        for array in arrays:
            spacers = [str(spacer) for spacer in array.get("spacers", []) if spacer]
            if spacers:
                array_lengths.append(len(spacers))
            for spacer in spacers:
                spacer_length = len(spacer)
                spacer_lengths.append(spacer_length)
                spacer_length_histogram[spacer_length] += 1
            repeat = str(array.get("repeat", "") or "").strip().upper()
            if repeat:
                repeat_sequences.append(repeat)
                repeat_length_histogram[len(repeat)] += 1
    return {
        "spacer_lengths": spacer_lengths or [28, 29, 30, 31, 32],
        "spacer_length_histogram": dict(spacer_length_histogram) or {
            28: 1.0,
            29: 1.0,
            30: 1.0,
            31: 1.0,
            32: 1.0,
        },
        "repeat_length_histogram": dict(repeat_length_histogram) or {
            28: 1.0,
            29: 1.0,
            30: 1.0,
            31: 1.0,
            32: 1.0,
        },
        "repeat_sequences": repeat_sequences,
        "arrays_per_strain": arrays_per_strain or [1, 2, 3],
        "array_lengths": array_lengths or [5, 6, 7, 8, 9, 10, 12],
    }


def _read_fasta_sequences(fasta_path: Path) -> list[str]:
    sequences: list[str] = []
    current_chunks: list[str] = []
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_chunks:
                    sequences.append("".join(current_chunks).upper())
                    current_chunks = []
                continue
            current_chunks.append(line)
    if current_chunks:
        sequences.append("".join(current_chunks).upper())
    return [sequence for sequence in sequences if sequence]


def _estimate_fasta_priors(
    repeat_sequences: list[str],
    spacer_sequences: list[str],
) -> dict:
    spacer_lengths = [len(sequence) for sequence in spacer_sequences if sequence]
    spacer_length_histogram = Counter(spacer_lengths)
    repeat_sequences = [str(sequence).strip().upper() for sequence in repeat_sequences if str(sequence).strip()]
    repeat_length_histogram = Counter(len(sequence) for sequence in repeat_sequences if sequence)
    return {
        "spacer_lengths": spacer_lengths or [28, 29, 30, 31, 32],
        "spacer_length_histogram": dict(spacer_length_histogram) or {
            28: 1.0,
            29: 1.0,
            30: 1.0,
            31: 1.0,
            32: 1.0,
        },
        "repeat_length_histogram": dict(repeat_length_histogram) or {
            28: 1.0,
            29: 1.0,
            30: 1.0,
            31: 1.0,
            32: 1.0,
        },
        "repeat_sequences": repeat_sequences,
        "arrays_per_strain": [1, 2, 3],
        "array_lengths": [5, 6, 7, 8, 9, 10, 12],
    }


def _sample_group_size(num_strains: int, rng: random.Random, config: dict) -> int:
    raw_values = list(config.get("simulation_group_size_distribution", []))
    values = [int(value) for value in raw_values if int(value) > 0]
    if values:
        return max(1, min(num_strains, int(rng.choice(values))))
    min_group_size = int(config.get("simulation_min_group_size", 4))
    max_group_size = int(config.get("simulation_max_group_size", 14))
    return min(
        num_strains,
        max(min_group_size, rng.randint(min_group_size, max_group_size)),
    )


def _target_membership_budget(num_strains: int, config: dict) -> int | None:
    target_arrays_per_strain = float(config.get("simulation_target_arrays_per_strain", 0.0) or 0.0)
    if target_arrays_per_strain <= 0.0:
        return None
    return max(num_strains, int(round(num_strains * target_arrays_per_strain)))


def _adjust_group_sizes_to_budget(
    group_sizes: list[int],
    target_total: int,
    num_strains: int,
    rng: random.Random,
    config: dict,
) -> list[int]:
    if not group_sizes:
        return group_sizes
    max_group_size = max(
        [int(config.get("simulation_max_group_size", 14))]
        + [int(value) for value in config.get("simulation_group_size_distribution", []) if int(value) > 0]
    )
    max_group_size = max(1, min(num_strains, max_group_size))
    current_total = sum(group_sizes)

    while current_total < target_total:
        adjustable = [index for index, size in enumerate(group_sizes) if size < max_group_size]
        if not adjustable:
            group_sizes.append(_sample_group_size(num_strains, rng, config))
            current_total = sum(group_sizes)
            continue
        group_index = rng.choice(adjustable)
        group_sizes[group_index] += 1
        current_total += 1

    while current_total > target_total:
        adjustable = [index for index, size in enumerate(group_sizes) if size > 1]
        if not adjustable:
            break
        group_index = rng.choice(adjustable)
        group_sizes[group_index] -= 1
        current_total -= 1

    return [size for size in group_sizes if size > 0]


def _sample_root_array_length(priors: dict, rng: random.Random, config: dict) -> int:
    target_values = list(config.get("simulation_target_mean_array_length_distribution", []))
    target_choices = [int(value) for value in target_values if int(value) > 0]
    if target_choices:
        target_mean = float(rng.choice(target_choices))
        sampled = int(round(rng.gauss(target_mean, max(1.5, target_mean * 0.18))))
        return max(4, sampled)
    return max(
        1,
        _sample_from_distribution(
            [int(value) for value in priors.get("array_lengths", [])],
            rng,
            default=8,
        ),
    )


def _resolve_group_memberships(
    strain_ids: list[str],
    num_groups: int,
    rng: random.Random,
    config: dict,
) -> list[list[str]]:
    target_total_memberships = _target_membership_budget(len(strain_ids), config)
    if num_groups <= 0 and target_total_memberships is None:
        return []

    group_sizes: list[int] = []
    if num_groups > 0:
        group_sizes = [
            _sample_group_size(len(strain_ids), rng, config)
            for _ in range(num_groups)
        ]
    else:
        running_total = 0
        while running_total < target_total_memberships:
            sampled = _sample_group_size(len(strain_ids), rng, config)
            group_sizes.append(sampled)
            running_total += sampled

    if target_total_memberships is not None:
        group_sizes = _adjust_group_sizes_to_budget(
            group_sizes,
            target_total_memberships,
            len(strain_ids),
            rng,
            config,
        )
    elif group_sizes:
        total_capacity = sum(group_sizes)
        if total_capacity < len(strain_ids):
            deficit = len(strain_ids) - total_capacity
            for index in range(deficit):
                group_sizes[index % len(group_sizes)] += 1

    shuffled_strains = list(strain_ids)
    rng.shuffle(shuffled_strains)
    memberships = [[] for _ in range(len(group_sizes))]
    available_group_indices = list(range(len(group_sizes)))
    rng.shuffle(available_group_indices)
    strain_cursor = 0
    for group_index in available_group_indices:
        if strain_cursor >= len(shuffled_strains):
            break
        memberships[group_index].append(shuffled_strains[strain_cursor])
        strain_cursor += 1

    while strain_cursor < len(shuffled_strains):
        candidate_groups = [
            group_index
            for group_index in range(len(group_sizes))
            if len(memberships[group_index]) < group_sizes[group_index]
        ]
        if not candidate_groups:
            break
        group_index = rng.choice(candidate_groups)
        memberships[group_index].append(shuffled_strains[strain_cursor])
        strain_cursor += 1

    for group_index in range(len(group_sizes)):
        while len(memberships[group_index]) < group_sizes[group_index]:
            available = [
                strain_id
                for strain_id in strain_ids
                if strain_id not in memberships[group_index]
            ]
            if not available:
                available = list(strain_ids)
            memberships[group_index].append(rng.choice(available))
        memberships[group_index] = sorted(memberships[group_index])
    return memberships


def _sample_block_size(mean_alpha: float, max_block: int, rng: random.Random) -> int:
    max_block = max(1, int(max_block))
    mean_alpha = max(1.0, float(mean_alpha))
    sampled = int(math.ceil(rng.expovariate(1.0 / mean_alpha)))
    return max(1, min(max_block, sampled))


def _sample_boundary_start(num_spacers: int, block_size: int, rng: random.Random) -> int:
    if num_spacers <= block_size:
        return 0
    positions = list(range(0, num_spacers - block_size + 1))
    center = max(0.0, (num_spacers - block_size) / 2.0)
    weights = [1.0 + abs(position - center) for position in positions]
    total = sum(weights)
    threshold = rng.random() * total
    cumulative = 0.0
    for position, weight in zip(positions, weights):
        cumulative += weight
        if cumulative >= threshold:
            return position
    return positions[-1]


def _apply_duplication_event(spacers: list[str], rng: random.Random) -> list[str]:
    if len(spacers) < 2:
        return spacers
    start = rng.randrange(len(spacers))
    block_size = min(len(spacers) - start, rng.randint(1, 2))
    block = list(spacers[start:start + block_size])
    insert_at = rng.randint(0, len(spacers))
    spacers[insert_at:insert_at] = block
    return spacers


def _apply_rearrangement_event(spacers: list[str], rng: random.Random) -> list[str]:
    if len(spacers) < 4:
        return spacers
    first_index = rng.randrange(len(spacers))
    second_index = rng.randrange(len(spacers))
    if first_index == second_index:
        return spacers
    spacers[first_index], spacers[second_index] = spacers[second_index], spacers[first_index]
    return spacers


def _apply_reacquisition_event(
    spacers: list[str],
    rng: random.Random,
    spacer_pool: set[str],
    length_weights: dict[int, float],
) -> list[str]:
    if spacers and rng.random() < 0.70:
        spacers.insert(0, rng.choice(spacers))
    else:
        spacers.insert(
            0,
            _random_spacer(rng, length_weights, spacer_pool, allow_existing=False),
        )
    return spacers


def _simulate_branch_mutation_profile(
    config: dict,
    rng: random.Random,
    branch_length: float,
    array_length: int,
) -> dict[str, float | str]:
    theta = _sample_float_range(
        float(config.get("simulation_theta_min", 4.0)),
        float(config.get("simulation_theta_max", 8.0)),
        rng,
        default=6.0,
    )
    alpha = _sample_float_range(
        float(config.get("simulation_alpha_min", 1.5)),
        float(config.get("simulation_alpha_max", 4.0)),
        rng,
        default=2.7,
    )
    model_weights = dict(config.get("simulation_deletion_model_mixture", {}))
    if not model_weights:
        model_weights = {
            "BDM": float(config.get("simulation_bdm_weight", 0.8)),
            "IDM": float(config.get("simulation_idm_weight", 0.2)),
        }
    deletion_model = _sample_weighted_label(
        model_weights,
        rng,
        default=str(config.get("simulation_deletion_model_preference", "BDM")),
    )
    branch_scale = max(0.05, float(branch_length))
    deletion_boundary_effect = bool(config.get("simulation_enable_boundary_effect", True))
    rho_b = theta / max(1.0, float(max(1, array_length)) * alpha)
    return {
        "theta": theta,
        "alpha": alpha,
        "deletion_model": deletion_model,
        "branch_scale": branch_scale,
        "rho_b": rho_b,
        "boundary_effect": deletion_boundary_effect,
    }


def _mutate_array(
    root_array: list[str],
    rng: random.Random,
    length_weights: dict[int, float],
    spacer_pool: set[str],
    branch_length: float,
    config: dict,
) -> list[str]:
    spacers = list(root_array)
    mutation_profile = _simulate_branch_mutation_profile(
        config,
        rng,
        branch_length=branch_length,
        array_length=len(spacers),
    )
    theta = float(mutation_profile["theta"])
    alpha = float(mutation_profile["alpha"])
    deletion_model = str(mutation_profile["deletion_model"])
    branch_scale = float(mutation_profile["branch_scale"])
    boundary_effect = bool(mutation_profile["boundary_effect"])

    acquisition_events = _sample_poisson(theta * branch_scale, rng)
    max_new_spacers = int(config.get("simulation_max_new_spacers_per_branch", 6))
    acquisition_events = min(acquisition_events, max(1, max_new_spacers))
    for _ in range(acquisition_events):
        spacers.insert(0, _random_spacer(rng, length_weights, spacer_pool, allow_existing=False))

    if spacers:
        if deletion_model.upper() == "BDM":
            rho_b = float(mutation_profile["rho_b"])
            block_events = _sample_poisson(rho_b * len(spacers) * branch_scale, rng)
            block_events = min(block_events, 3)
            for _ in range(block_events):
                if not spacers:
                    break
                max_block = max(1, min(len(spacers), max(2, int(round(alpha * 2.5)))))
                block_size = _sample_block_size(alpha, max_block, rng)
                start = 0
                if len(spacers) > block_size:
                    if boundary_effect:
                        start = _sample_boundary_start(len(spacers), block_size, rng)
                    else:
                        start = rng.randint(0, len(spacers) - block_size)
                del spacers[start:start + block_size]
        else:
            independent_rate = float(config.get("simulation_independent_deletion_rate", 0.10))
            delete_events = _sample_poisson(independent_rate * len(spacers) * branch_scale, rng)
            delete_events = min(delete_events, max(1, len(spacers) // 2))
            for _ in range(delete_events):
                if len(spacers) <= 1:
                    break
                del spacers[rng.randrange(len(spacers))]

    if bool(config.get("simulation_allow_rare_complex_events", True)):
        duplication_rate = float(config.get("simulation_duplication_rate", 0.03))
        rearrangement_rate = float(config.get("simulation_rearrangement_rate", 0.01))
        reacquisition_rate = float(config.get("simulation_reacquisition_rate", 0.01))
        if rng.random() < duplication_rate * max(0.25, branch_scale):
            spacers = _apply_duplication_event(spacers, rng)
        if rng.random() < rearrangement_rate * max(0.25, branch_scale):
            spacers = _apply_rearrangement_event(spacers, rng)
        if rng.random() < reacquisition_rate * max(0.25, branch_scale):
            spacers = _apply_reacquisition_event(spacers, rng, spacer_pool, length_weights)

    if not spacers:
        spacers = [_random_spacer(rng, length_weights, spacer_pool, allow_existing=False)]
    return spacers


def _propagate_arrays_on_tree(
    node: dict,
    current_array: list[str],
    leaf_arrays: dict[str, list[str]],
    leaf_paths: dict[str, str],
    rng: random.Random,
    length_weights: dict[int, float],
    spacer_pool: set[str],
    config: dict,
    current_path: str = "",
) -> None:
    if "name" in node:
        leaf_arrays[str(node["name"])] = list(current_array)
        leaf_paths[str(node["name"])] = current_path or "ROOT"
        return

    left_array = _mutate_array(
        current_array,
        rng,
        length_weights,
        spacer_pool,
        float(node.get("left_length", 0.1)),
        config,
    )
    right_array = _mutate_array(
        current_array,
        rng,
        length_weights,
        spacer_pool,
        float(node.get("right_length", 0.1)),
        config,
    )
    _propagate_arrays_on_tree(
        node["left"],
        left_array,
        leaf_arrays,
        leaf_paths,
        rng,
        length_weights,
        spacer_pool,
        config,
        current_path=f"{current_path}L",
    )
    _propagate_arrays_on_tree(
        node["right"],
        right_array,
        leaf_arrays,
        leaf_paths,
        rng,
        length_weights,
        spacer_pool,
        config,
        current_path=f"{current_path}R",
    )

def _check_group_spacer_overlap(
    leaf_arrays: dict[str, list[str]],
    min_shared_spacers: int = 1,
) -> bool:
    # Ensure every array has at least one shared spacer with another array in the same group.

    strain_ids = list(leaf_arrays.keys())

    spacer_sets = {
        sid: set(leaf_arrays[sid])
        for sid in strain_ids
    }

    for sid in strain_ids:
        has_overlap = False

        for other in strain_ids:
            if sid == other:
                continue

            overlap = (
                spacer_sets[sid]
                &
                spacer_sets[other]
            )

            if len(overlap) >= min_shared_spacers:
                has_overlap = True
                break

        if not has_overlap:
            return False

    return True

def _simulate_group(
    *,
    group_id: str,
    repeat_sequence: str,
    strain_ids: list[str],
    priors: dict,
    rng: random.Random,
    config: dict,
) -> tuple[dict[str, list[str]], dict]:
    tree = _sample_tree(
        strain_ids,
        rng,
        tree_type=str(config.get("simulation_tree_type", "coalescent")),
    )
    root_length = _sample_root_array_length(priors, rng, config)
    length_weights = _resolve_spacer_length_weights(priors, config)
    spacer_pool: set[str] = set()
    root_array = [
        _random_spacer(rng, length_weights, spacer_pool, allow_existing=False)
        for _ in range(root_length)
    ]
    leaf_arrays: dict[str, list[str]] = {}
    leaf_paths: dict[str, str] = {}
    _propagate_arrays_on_tree(
        tree,
        root_array,
        leaf_arrays,
        leaf_paths,
        rng,
        length_weights,
        spacer_pool,
        config,
    )

    max_retry = int(
    config.get(
        "simulation_group_overlap_retry",
        20
    )
)

    retry = 0

    while not _check_group_spacer_overlap(
        leaf_arrays,
        min_shared_spacers=int(
            config.get(
                "simulation_min_shared_spacers",
                1
            )
        ),
    ):

        retry += 1

        if retry >= max_retry:
            break


        # regenerate tree evolution
        leaf_arrays.clear()
        leaf_paths.clear()

        tree = _sample_tree(
            strain_ids,
            rng,
            tree_type=str(
                config.get(
                    "simulation_tree_type",
                    "coalescent"
                )
            ),
        )

        _propagate_arrays_on_tree(
            tree,
            root_array,
            leaf_arrays,
            leaf_paths,
            rng,
            length_weights,
            spacer_pool,
            config,
        )

    group_theta = _sample_float_range(
        float(config.get("simulation_theta_min", 4.0)),
        float(config.get("simulation_theta_max", 8.0)),
        rng,
        default=6.0,
    )
    group_alpha = _sample_float_range(
        float(config.get("simulation_alpha_min", 1.5)),
        float(config.get("simulation_alpha_max", 4.0)),
        rng,
        default=2.7,
    )
    group_model = _sample_weighted_label(
        dict(config.get("simulation_deletion_model_mixture", {}))
        or {
            "BDM": float(config.get("simulation_bdm_weight", 0.8)),
            "IDM": float(config.get("simulation_idm_weight", 0.2)),
        },
        rng,
        default="BDM",
    )
    metadata = {
        "candidate_group_id": group_id,
        "repeat_family": repeat_sequence,
        "repeat": repeat_sequence,
        "num_member_strains": len(strain_ids),
        "newick": _tree_to_newick(tree) + ";",
        "num_internal_nodes": _count_internal_nodes(tree),
        "root_array_length": len(root_array),
        "leaf_paths": dict(leaf_paths),
        "simulation_tree_type": str(config.get("simulation_tree_type", "coalescent")),
        "group_theta": group_theta,
        "group_alpha": group_alpha,
        "group_deletion_model": group_model,
    }
    return leaf_arrays, metadata


def _ensure_every_strain_has_array(
    strains_data: list[dict],
    priors: dict,
    rng: random.Random,
    config: dict,
) -> None:
    length_weights = _resolve_spacer_length_weights(priors, config)
    for strain in strains_data:
        if strain["arrays"]:
            continue
        repeat_sequence = _sample_repeat_sequence(priors, config, rng)
        used_spacers: set[str] = set()
        spacers = [
            _random_spacer(rng, length_weights, used_spacers, allow_existing=False)
            for _ in range(rng.randint(8, 18))
        ]
        array_id = "fallback_array_0001"
        strain["arrays"].append(
            {
                "array_id": array_id,
                "array_uid": f"{strain['strain_id']}::{array_id}",
                "candidate_group_id": f"group_fallback::{strain['strain_id']}",
                "spacers": spacers,
                "repeat": repeat_sequence,
                "repeat_family": repeat_sequence,
                "array_level_features": _array_feature_vector(spacers),
            }
        )


def generate_simulated_dataset(
    split_name: str,
    config: dict,
    empirical_strains: list[dict] | None = None,
    priors: dict | None = None,
) -> tuple[list[dict], dict]:
    empirical_strains = list(empirical_strains or [])
    priors = dict(priors or _estimate_empirical_priors(empirical_strains))
    split_key = SPLIT_TO_CONFIG_KEY.get(split_name, split_name)
    split_seed_offset = {"train": 0, "valid": 1000, "test": 2000}.get(split_key, 3000)
    rng = random.Random(int(config.get("simulation_seed", 42)) + split_seed_offset)

    num_strains = int(config.get(f"simulation_{split_key}_num_strains", 96))
    num_groups = int(config.get(f"simulation_{split_key}_num_groups", 18))
    strain_ids = [f"{split_key}_strain_{index:04d}" for index in range(1, num_strains + 1)]
    strains_data = [
        {
            "strain_id": strain_id,
            "arrays": [],
            "num_arrays": 0,
        }
        for strain_id in strain_ids
    ]
    strain_lookup = {row["strain_id"]: row for row in strains_data}
    group_memberships = _resolve_group_memberships(
        strain_ids,
        num_groups,
        rng,
        config,
    )
    metadata = {
        "split_name": split_name,
        "num_strains": num_strains,
        "num_groups": len(group_memberships),
        "source": "simulated",
        "groups": [],
    }

    for group_index, member_strain_ids in enumerate(group_memberships, start=1):
        group_id = f"group_{group_index:04d}"
        repeat_sequence = _sample_repeat_sequence(priors, config, rng)
        leaf_arrays, group_metadata = _simulate_group(
            group_id=group_id,
            repeat_sequence=repeat_sequence,
            strain_ids=member_strain_ids,
            priors=priors,
            rng=rng,
            config=config,
        )
        for member_index, strain_id in enumerate(member_strain_ids, start=1):
            spacers = list(leaf_arrays[strain_id])
            tree_path = str(group_metadata["leaf_paths"].get(strain_id, ""))
            array_id = f"{group_id}_array_{member_index:04d}"
            strain_lookup[strain_id]["arrays"].append(
                {
                    "array_id": array_id,
                    "array_uid": f"{strain_id}::{array_id}",
                    "candidate_group_id": group_id,
                    "spacers": spacers,
                    "repeat": repeat_sequence,
                    "repeat_family": repeat_sequence,
                    "array_level_features": _array_feature_vector(spacers),
                    "simulated_tree_path": tree_path,
                    "simulated_tree_depth": len(tree_path) if tree_path and tree_path != "ROOT" else 0,
                }
            )
        metadata["groups"].append(
            {
                **group_metadata,
                "strain_ids": member_strain_ids,
                "array_leaf_labels": [
                    f"{strain_id}::{group_id}_array_{member_index:04d}"
                    for member_index, strain_id in enumerate(member_strain_ids, start=1)
                ],
            }
        )

    _ensure_every_strain_has_array(strains_data, priors, rng, config)
    for row in strains_data:
        row["num_arrays"] = len(row["arrays"])

    metadata["mean_arrays_per_strain"] = _safe_mean(
        [len(row["arrays"]) for row in strains_data],
        default=1.0,
    )
    metadata["mean_group_size"] = _safe_mean(
        [len(group["strain_ids"]) for group in metadata["groups"]],
        default=0.0,
    )
    metadata["num_fallback_arrays"] = sum(
        1
        for strain in strains_data
        for array in strain.get("arrays", [])
        if str(array.get("candidate_group_id", "")).startswith("group_fallback::")
    )
    return strains_data, metadata


def _resolve_empirical_prior_strains(
    config: dict,
    csv_loader: Callable[..., list[dict]],
) -> list[dict]:
    empirical_prior_csv_path = str(
        config.get("empirical_prior_csv_path")
        or config.get("processed_csv_path")
        or config.get("train_csv_path")
        or ""
    ).strip()
    if not empirical_prior_csv_path:
        return []
    empirical_path = Path(empirical_prior_csv_path)
    if not empirical_path.exists():
        return []
    require_repeat_columns = bool(config.get("require_repeat_columns_for_priors", False))
    return csv_loader(str(empirical_path), require_repeat_columns=require_repeat_columns)


def _resolve_fasta_prior_paths(config: dict) -> tuple[Path | None, Path | None]:
    repeat_path_raw = str(config.get("empirical_prior_repeat_fsa_path") or "").strip()
    spacer_path_raw = str(config.get("empirical_prior_spacer_fsa_path") or "").strip()
    repeat_path = Path(repeat_path_raw) if repeat_path_raw else None
    spacer_path = Path(spacer_path_raw) if spacer_path_raw else None
    return repeat_path, spacer_path


def _resolve_simulation_priors(
    config: dict,
    csv_loader: Callable[..., list[dict]],
) -> tuple[dict, dict]:
    repeat_fsa_path, spacer_fsa_path = _resolve_fasta_prior_paths(config)
    if repeat_fsa_path or spacer_fsa_path:
        missing_paths = [
            str(path)
            for path in (repeat_fsa_path, spacer_fsa_path)
            if path is not None and not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Configured prior .fsa file(s) not found: " + ", ".join(missing_paths)
            )
        repeat_sequences = _read_fasta_sequences(repeat_fsa_path) if repeat_fsa_path else []
        spacer_sequences = _read_fasta_sequences(spacer_fsa_path) if spacer_fsa_path else []
        if repeat_sequences or spacer_sequences:
            return _estimate_fasta_priors(repeat_sequences, spacer_sequences), {
                "prior_source": "fsa",
                "prior_repeat_fsa_path": str(repeat_fsa_path or ""),
                "prior_spacer_fsa_path": str(spacer_fsa_path or ""),
                "prior_repeat_sequences": len(repeat_sequences),
                "prior_spacer_sequences": len(spacer_sequences),
            }

    empirical_strains = _resolve_empirical_prior_strains(config, csv_loader)
    empirical_prior_csv_path = str(
        config.get("empirical_prior_csv_path")
        or config.get("processed_csv_path")
        or config.get("train_csv_path")
        or ""
    ).strip()
    return _estimate_empirical_priors(empirical_strains), {
        "prior_source": "csv" if empirical_strains else "default",
        "prior_csv_path": empirical_prior_csv_path,
        "prior_num_strains": len(empirical_strains),
    }


def _build_prior_signature(config: dict) -> str:
    signature_payload: dict[str, str | int] = {}
    for key in (
        "empirical_prior_csv_path",
        "empirical_prior_repeat_fsa_path",
        "empirical_prior_spacer_fsa_path",
    ):
        raw_value = str(config.get(key) or "").strip()
        if not raw_value:
            continue
        path = Path(raw_value)
        if path.exists():
            stats = path.stat()
            signature_payload[key] = f"{path}:{stats.st_size}:{stats.st_mtime_ns}"
        else:
            signature_payload[key] = f"{path}:missing"
    if not signature_payload:
        signature_payload["prior_source"] = "embedded_defaults"
    digest = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:10]


def _cache_path_for_split(split_name: str, config: dict) -> Path:
    cache_dir = Path(config.get("simulation_cache_dir", "simulated_cache"))
    seed = int(config.get("simulation_seed", 42))
    split_key = SPLIT_TO_CONFIG_KEY.get(split_name, split_name)
    version = str(config.get("simulation_cache_version", "v2"))
    prior_signature = _build_prior_signature(config)
    return cache_dir / f"{split_key}_{version}_{prior_signature}_seed_{seed}.json"


def load_split_strains(
    split_name: str,
    config: dict,
    csv_loader: Callable[..., list[dict]],
) -> tuple[list[dict], dict]:
    data_source = str(config.get("data_source", "real")).strip().lower()
    split_key = SPLIT_TO_CONFIG_KEY.get(split_name, split_name)
    if data_source != "simulated":
        csv_path = str(config.get(f"{split_key}_csv_path", "")).strip()
        if not csv_path:
            raise ValueError(f"Missing CSV path for split '{split_name}'.")
        return csv_loader(csv_path), {
            "split_name": split_name,
            "source": "real",
            "input_path": csv_path,
        }

    cache_path = _cache_path_for_split(split_name, config)
    if bool(config.get("simulation_use_cache", True)) and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload["strains_data"], payload.get("metadata", {})

    priors, prior_metadata = _resolve_simulation_priors(config, csv_loader)
    strains_data, metadata = generate_simulated_dataset(
        split_name=split_name,
        config=config,
        priors=priors,
    )
    metadata = dict(metadata)
    metadata.update(prior_metadata)
    metadata["cache_path"] = str(cache_path)
    if bool(config.get("simulation_use_cache", True)):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "strains_data": strains_data,
                    "metadata": metadata,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    return strains_data, metadata


def parse_args_zwx() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated CRISPR dataset cache.")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test", "training", "validation"])
    parser.add_argument("--output-dir", default="simulated_data_theta_4_8_seed42_array50")
    parser.add_argument("--prior-csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-strains", type=int, default=None)
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--cache-version", default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main_zwx() -> None:
    args = parse_args_zwx()
    config = deepcopy(TRAIN_CONFIG)

    split_key = SPLIT_TO_CONFIG_KEY.get(args.split, args.split)
    output_dir = Path(args.output_dir)
    sim_cache_dir = output_dir / "sim_cache"

    config["data_source"] = "simulated"
    config["simulation_cache_dir"] = str(sim_cache_dir)
    config["simulation_seed"] = int(args.seed)
    config["simulation_use_cache"] = not bool(args.no_cache)

    if args.cache_version:
        config["simulation_cache_version"] = str(args.cache_version)

    prior_csv = str(args.prior_csv).strip()
    if prior_csv:
        config["empirical_prior_csv_path"] = prior_csv
        config["processed_csv_path"] = prior_csv

    if args.num_strains is not None:
        config[f"simulation_{split_key}_num_strains"] = int(args.num_strains)
    if args.num_groups is not None:
        config[f"simulation_{split_key}_num_groups"] = int(args.num_groups)

    strains_data, metadata = load_split_strains(
        split_name=args.split,
        config=config,
        csv_loader=read_spacers_from_csv,
    )

    print(f"split={args.split}")
    print(f"source={metadata.get('source', 'unknown')}")
    print(f"num_strains={len(strains_data)}")
    print(f"num_groups={metadata.get('num_groups', 'NA')}")
    print(f"cache_path={metadata.get('cache_path', 'NA')}")


if __name__ == "__main__":
    main_zwx()