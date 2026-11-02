from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

from Bio import Phylo

from project_config import TRAIN_CONFIG
from simulated_data import (
    SPLIT_TO_CONFIG_KEY,
    _array_feature_vector,
    _count_internal_nodes,
    _estimate_empirical_priors,
    _propagate_arrays_on_tree,
    _random_spacer,
    _resolve_group_memberships,
    _resolve_spacer_length_weights,
    _sample_repeat_sequence,
    _sample_root_array_length,
    _sample_tree,
    _tree_to_newick,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate strains whose groups share one global true tree."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["single_group", "global_groups"], default="global_groups")
    parser.add_argument("--split", choices=["training", "validation", "test"], default="test")
    parser.add_argument("--seed", type=int, default=int(TRAIN_CONFIG["simulation_seed"]))
    parser.add_argument("--num-strains", type=int, default=None)
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--target-arrays-per-strain", type=float, default=None)
    parser.add_argument("--tree-type", default=str(TRAIN_CONFIG.get("simulation_tree_type", "coalescent")))
    parser.add_argument("--cache-name", default="", help="Optional output JSON filename inside sim_cache.")
    return parser.parse_args()


def split_key(split: str) -> str:
    return SPLIT_TO_CONFIG_KEY.get(split, split)


def build_config(args: argparse.Namespace) -> dict:
    config = dict(TRAIN_CONFIG)
    key = split_key(args.split)
    config["simulation_seed"] = int(args.seed)
    config["simulation_tree_type"] = str(args.tree_type)
    if args.num_strains is not None:
        config[f"simulation_{key}_num_strains"] = int(args.num_strains)
    if args.num_groups is not None:
        config[f"simulation_{key}_num_groups"] = int(args.num_groups)
    if args.target_arrays_per_strain is not None:
        config["simulation_target_arrays_per_strain"] = float(args.target_arrays_per_strain)
    return config


def sample_global_tree(strain_ids: list[str], rng: random.Random, config: dict) -> tuple[dict, str]:
    tree = _sample_tree(strain_ids, rng, tree_type=str(config.get("simulation_tree_type", "coalescent")))
    return tree, _tree_to_newick(tree) + ";"


def induced_subtree_newick(global_newick: str, members: list[str]) -> str:
    tree = Phylo.read(StringIO(global_newick), "newick")
    member_set = set(members)
    for terminal in list(tree.get_terminals()):
        if str(terminal.name) not in member_set:
            tree.prune(terminal)
    handle = StringIO()
    Phylo.write(tree, handle, "newick")
    return handle.getvalue().strip()


def sample_array_replicate(
    global_tree: dict,
    strain_ids: list[str],
    repeat_sequence: str,
    priors: dict,
    rng: random.Random,
    config: dict,
) -> tuple[dict[str, list[str]], dict[str, str], int]:
    length_weights = _resolve_spacer_length_weights(priors, config)
    root_length = _sample_root_array_length(priors, rng, config)
    spacer_pool: set[str] = set()
    root_array = [
        _random_spacer(rng, length_weights, spacer_pool, allow_existing=False)
        for _ in range(root_length)
    ]
    leaf_arrays: dict[str, list[str]] = {}
    leaf_paths: dict[str, str] = {}
    _propagate_arrays_on_tree(
        global_tree,
        root_array,
        leaf_arrays,
        leaf_paths,
        rng,
        length_weights,
        spacer_pool,
        config,
    )
    missing = [strain_id for strain_id in strain_ids if strain_id not in leaf_arrays]
    if missing:
        raise RuntimeError(f"Global tree propagation missed leaves: {missing[:5]}")
    return leaf_arrays, leaf_paths, root_length


def allocate_array_counts(num_strains: int, target_mean: float, rng: random.Random) -> list[int]:
    num_strains = max(1, int(num_strains))
    target_total = max(num_strains, int(round(num_strains * max(1.0, float(target_mean)))))
    counts = [1] * num_strains
    remaining = target_total - num_strains
    while remaining > 0:
        counts[rng.randrange(num_strains)] += 1
        remaining -= 1
    return counts


def group_parameters(rng: random.Random, config: dict) -> dict:
    mixture = dict(config.get("simulation_deletion_model_mixture", {}) or {"BDM": 0.8, "IDM": 0.2})
    total = sum(float(value) for value in mixture.values() if float(value) > 0.0)
    threshold = rng.random() * max(total, 1.0)
    cumulative = 0.0
    model = "BDM"
    for name, weight in mixture.items():
        cumulative += max(0.0, float(weight))
        if cumulative >= threshold:
            model = str(name)
            break
    theta_min = float(config.get("simulation_theta_min", 4.0))
    theta_max = float(config.get("simulation_theta_max", 8.0))
    alpha_min = float(config.get("simulation_alpha_min", 1.5))
    alpha_max = float(config.get("simulation_alpha_max", 4.0))
    return {
        "group_theta": rng.uniform(min(theta_min, theta_max), max(theta_min, theta_max)),
        "group_alpha": rng.uniform(min(alpha_min, alpha_max), max(alpha_min, alpha_max)),
        "group_deletion_model": model,
    }


def append_array(
    strains: dict[str, dict],
    strain_id: str,
    group_id: str,
    array_index: int,
    spacers: list[str],
    leaf_path: str,
) -> str:
    array_id = f"{group_id}_array_{array_index:04d}"
    array_uid = f"{strain_id}::{array_id}"
    strains[strain_id]["arrays"].append(
        {
            "array_id": array_id,
            "array_uid": array_uid,
            "candidate_group_id": group_id,
            "spacers": list(spacers),
            "repeat": strains[strain_id].get("_current_repeat", ""),
            "repeat_family": strains[strain_id].get("_current_repeat", ""),
            "array_level_features": _array_feature_vector(list(spacers)),
            "simulated_tree_path": str(leaf_path),
            "simulated_tree_depth": len(leaf_path) if leaf_path and leaf_path != "ROOT" else 0,
        }
    )
    return array_uid


def build_single_group(
    strain_ids: list[str],
    global_tree: dict,
    global_newick: str,
    priors: dict,
    rng: random.Random,
    config: dict,
) -> tuple[list[dict], dict]:
    group_id = "group_0001"
    repeat_sequence = _sample_repeat_sequence(priors, config, rng)
    target_mean = float(config.get("simulation_target_arrays_per_strain", 2.2))
    counts = allocate_array_counts(len(strain_ids), target_mean, rng)
    strains = {
        strain_id: {"strain_id": strain_id, "arrays": [], "num_arrays": 0, "_current_repeat": repeat_sequence}
        for strain_id in strain_ids
    }
    primary_labels: list[str] = []
    all_labels: list[str] = []
    max_arrays = max(counts)
    root_lengths: list[int] = []
    leaf_paths: dict[str, str] = {}
    for replicate_index in range(max_arrays):
        leaf_arrays, replicate_paths, root_length = sample_array_replicate(
            global_tree, strain_ids, repeat_sequence, priors, rng, config
        )
        root_lengths.append(root_length)
        if replicate_index == 0:
            leaf_paths = dict(replicate_paths)
        for strain_index, strain_id in enumerate(strain_ids):
            if counts[strain_index] <= replicate_index:
                continue
            array_uid = append_array(
                strains,
                strain_id,
                group_id,
                replicate_index + 1,
                leaf_arrays[strain_id],
                replicate_paths[strain_id],
            )
            all_labels.append(array_uid)
            if replicate_index == 0:
                primary_labels.append(array_uid)
    output_strains = []
    for strain_id in strain_ids:
        strains[strain_id].pop("_current_repeat", None)
        strains[strain_id]["num_arrays"] = len(strains[strain_id]["arrays"])
        output_strains.append(strains[strain_id])
    metadata_group = {
        "candidate_group_id": group_id,
        "repeat_family": repeat_sequence,
        "repeat": repeat_sequence,
        "num_member_strains": len(strain_ids),
        "num_arrays": len(all_labels),
        "newick": global_newick,
        "num_internal_nodes": _count_internal_nodes(global_tree),
        "root_array_length": int(round(sum(root_lengths) / max(1, len(root_lengths)))),
        "leaf_paths": leaf_paths,
        "array_leaf_labels": primary_labels,
        "all_array_leaf_labels": all_labels,
        "simulation_tree_type": str(config.get("simulation_tree_type", "coalescent")),
        **group_parameters(rng, config),
        "strain_ids": list(strain_ids),
    }
    return output_strains, metadata_group


def build_global_groups(
    strain_ids: list[str],
    global_tree: dict,
    global_newick: str,
    priors: dict,
    rng: random.Random,
    config: dict,
    num_groups: int,
) -> tuple[list[dict], list[dict]]:
    # Group count is an explicit structural request.  Do not use the normal
    # array-membership budget here: that budget is about arrays per strain,
    # whereas one group membership is only one observation in the old
    # generator.  Extra arrays are generated below within the requested groups.
    membership_config = dict(config)
    membership_config["simulation_target_arrays_per_strain"] = 0.0
    memberships = _resolve_group_memberships(strain_ids, int(num_groups), rng, membership_config)
    covered = {strain_id for members in memberships for strain_id in members}
    missing = [strain_id for strain_id in strain_ids if strain_id not in covered]
    if missing:
        if not memberships:
            memberships = [list(missing)]
        else:
            memberships[0].extend(missing)
            memberships[0] = sorted(set(memberships[0]))

    strains = {
        strain_id: {"strain_id": strain_id, "arrays": [], "num_arrays": 0}
        for strain_id in strain_ids
    }
    target_mean = float(config.get("simulation_target_arrays_per_strain", 2.2))
    target_counts = allocate_array_counts(len(strain_ids), target_mean, rng)
    target_count_by_strain = dict(zip(strain_ids, target_counts))
    member_groups: dict[str, list[int]] = defaultdict(list)
    for group_index, members in enumerate(memberships):
        for strain_id in members:
            member_groups[strain_id].append(group_index)
    # The membership resolver normally covers all strains; retain a defensive
    # fallback so every strain still receives its requested array count.
    for strain_id in strain_ids:
        if not member_groups.get(strain_id):
            memberships[0].append(strain_id)
            member_groups[strain_id] = [0]
    remaining_counts = dict(target_count_by_strain)
    metadata_groups: list[dict] = []
    for group_index, members in enumerate(memberships, start=1):
        group_id = f"group_{group_index:04d}"
        repeat_sequence = _sample_repeat_sequence(priors, config, rng)
        for strain_id in members:
            strains[strain_id]["_current_repeat"] = repeat_sequence
        zero_based_group_index = group_index - 1
        group_counts: dict[str, int] = {}
        for strain_id in members:
            group_members_left = max(1, len(member_groups[strain_id]) - member_groups[strain_id].index(zero_based_group_index))
            group_counts[strain_id] = max(0, (remaining_counts[strain_id] + group_members_left - 1) // group_members_left)
        max_group_arrays = max(group_counts.values(), default=0)
        primary_labels: list[str] = []
        all_group_labels: list[str] = []
        root_lengths: list[int] = []
        leaf_paths: dict[str, str] = {}
        array_index = 1
        for replicate_index in range(max_group_arrays):
            leaf_arrays, replicate_paths, root_length = sample_array_replicate(
                global_tree, strain_ids, repeat_sequence, priors, rng, config
            )
            root_lengths.append(root_length)
            for strain_id in members:
                if group_counts[strain_id] <= replicate_index:
                    continue
                array_uid = append_array(
                    strains,
                    strain_id,
                    group_id,
                    array_index,
                    leaf_arrays[strain_id],
                    replicate_paths[strain_id],
                )
                array_index += 1
                all_group_labels.append(array_uid)
                if replicate_index == 0:
                    primary_labels.append(array_uid)
                remaining_counts[strain_id] -= 1
                leaf_paths[strain_id] = replicate_paths[strain_id]
        for strain_id in members:
            strains[strain_id].pop("_current_repeat", None)
        group_newick = induced_subtree_newick(global_newick, list(members))
        metadata_groups.append(
            {
                "candidate_group_id": group_id,
                "repeat_family": repeat_sequence,
                "repeat": repeat_sequence,
                "num_member_strains": len(members),
                "num_arrays": len(all_group_labels),
                "newick": group_newick,
                "global_newick": global_newick,
                "num_internal_nodes": max(0, len(members) - 1),
                "root_array_length": root_length,
                "leaf_paths": {member: leaf_paths[member] for member in members},
                "array_leaf_labels": primary_labels,
                "all_array_leaf_labels": all_group_labels,
                "simulation_tree_type": str(config.get("simulation_tree_type", "coalescent")),
                "strain_ids": list(members),
                **group_parameters(rng, config),
            }
        )

    output_strains = []
    for strain_id in strain_ids:
        strains[strain_id]["num_arrays"] = len(strains[strain_id]["arrays"])
        output_strains.append(strains[strain_id])
    return output_strains, metadata_groups


def generate(args: argparse.Namespace) -> tuple[list[dict], dict, dict]:
    config = build_config(args)
    key = split_key(args.split)
    num_strains = int(args.num_strains or config.get(f"simulation_{key}_num_strains", 96))
    if num_strains < 2:
        raise ValueError("--num-strains must be at least 2.")
    default_groups = int(config.get(f"simulation_{key}_num_groups", 18))
    num_groups = int(args.num_groups if args.num_groups is not None else default_groups)
    if args.mode == "single_group":
        num_groups = 1
    elif num_groups < 1:
        raise ValueError("--num-groups must be at least 1 in global_groups mode.")

    split_offset = {"train": 0, "valid": 1000, "test": 2000}.get(key, 3000)
    rng = random.Random(int(args.seed) + split_offset)
    strain_ids = [f"{key}_strain_{index:04d}" for index in range(1, num_strains + 1)]
    priors = _estimate_empirical_priors([])
    global_tree, global_newick = sample_global_tree(strain_ids, rng, config)

    if args.mode == "single_group":
        strains_data, group = build_single_group(
            strain_ids, global_tree, global_newick, priors, rng, config
        )
        groups = [group]
    else:
        strains_data, groups = build_global_groups(
            strain_ids, global_tree, global_newick, priors, rng, config, num_groups
        )

    metadata = {
        "split_name": args.split,
        "source": "simulated_global_tree",
        "mode": args.mode,
        "seed": int(args.seed),
        "num_strains": len(strains_data),
        "num_groups": len(groups),
        "num_arrays": sum(len(row["arrays"]) for row in strains_data),
        "mean_arrays_per_strain": sum(len(row["arrays"]) for row in strains_data) / max(1, len(strains_data)),
        "mean_group_size": sum(len(group["strain_ids"]) for group in groups) / max(1, len(groups)),
        "simulation_tree_type": str(config.get("simulation_tree_type", "coalescent")),
        "global_newick": global_newick,
        "global_num_internal_nodes": _count_internal_nodes(global_tree),
        "global_tree_leaf_labels": list(strain_ids),
        "groups": groups,
        "config_snapshot": {
            key: value
            for key, value in config.items()
            if key.startswith("simulation_")
            and isinstance(value, (str, int, float, bool, list, dict))
        },
    }
    return strains_data, metadata, config


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strains_data, metadata, config = generate(args)
    cache_dir = output_dir / "sim_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = args.cache_name or f"{split_key(args.split)}_global_tree_{args.mode}_seed_{args.seed}.json"
    cache_path = cache_dir / filename
    cache_path.write_text(
        json.dumps({"strains_data": strains_data, "metadata": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "global_tree.nwk").write_text(metadata["global_newick"] + "\n", encoding="utf-8")
    summary = {
        "cache_path": str(cache_path),
        "mode": metadata["mode"],
        "split": metadata["split_name"],
        "seed": metadata["seed"],
        "num_strains": metadata["num_strains"],
        "num_groups": metadata["num_groups"],
        "num_arrays": metadata["num_arrays"],
        "mean_arrays_per_strain": metadata["mean_arrays_per_strain"],
        "mean_group_size": metadata["mean_group_size"],
        "target_arrays_per_strain": config.get("simulation_target_arrays_per_strain"),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"mode={metadata['mode']}")
    print(f"split={metadata['split_name']}")
    print(f"num_strains={metadata['num_strains']}")
    print(f"num_groups={metadata['num_groups']}")
    print(f"num_arrays={metadata['num_arrays']}")
    print(f"mean_arrays_per_strain={metadata['mean_arrays_per_strain']:.6f}")
    print(f"cache_path={cache_path}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Runtime: {time.perf_counter() - start:.6f} seconds")
