from __future__ import annotations

import argparse
import json
from pathlib import Path

from crispr_data_pipeline_zwx import read_spacers_from_csv
from project_config_copy import TRAIN_CONFIG
from simulated_data import load_split_strains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated CRISPR datasets whose every candidate group has a fixed "
            "number of arrays (for example 20, 50, 100)."
        )
    )
    parser.add_argument(
        "--array-counts",
        type=int,
        nargs="+",
        # default=[20, 50, 100],
        default=[50],
        help="Target number of arrays per simulated candidate group.",
    )
    parser.add_argument(
        "--output-root",
        default="simulated_array_count_benchmark_450000strains",
        help="Directory that will contain one subdirectory per array count.",
    )
    parser.add_argument("--train-groups", type=int, default=1000)
    parser.add_argument("--valid-groups", type=int, default=10)
    parser.add_argument("--test-groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-version-prefix",
        default="fixed_arrays",
        help="Prefix embedded in generated sim_cache filenames.",
    )
    parser.add_argument(
        "--prior-csv",
        default=None,
        help="Optional empirical CSV used to estimate spacer/repeat priors.",
    )
    parser.add_argument(
        "--prior-repeat-fsa",
        default=None,
        help="Optional repeat .fsa file used to estimate repeat priors.",
    )
    parser.add_argument(
        "--prior-spacer-fsa",
        default=None,
        help="Optional spacer .fsa file used to estimate spacer priors.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cache files even if they already exist.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, array_count: int, output_root: Path) -> dict:
    config = dict(TRAIN_CONFIG)
    config["data_source"] = "simulated"
    config["simulation_seed"] = int(args.seed)
    config["simulation_use_cache"] = True
    config["simulation_cache_version"] = f"{args.cache_version_prefix}_arrays_{array_count}"
    config["simulation_cache_dir"] = str(output_root / f"arrays_{array_count}" / "sim_cache")

    # Keep every generated candidate group at exactly `array_count` members.
    config["simulation_min_group_size"] = int(array_count)
    config["simulation_max_group_size"] = int(array_count)
    config["simulation_group_size_distribution"] = [int(array_count)]
    config["simulation_target_arrays_per_strain"] = 0.0

    split_to_groups = {
        "train": int(args.train_groups),
        "valid": int(args.valid_groups),
        "test": int(args.test_groups),
    }
    for split_name, num_groups in split_to_groups.items():
        config[f"simulation_{split_name}_num_groups"] = int(num_groups)
        # Match total capacity exactly so the generator does not expand group sizes.
        config[f"simulation_{split_name}_num_strains"] = int(array_count) * int(num_groups)

    if args.prior_csv:
        config["empirical_prior_csv_path"] = str(Path(args.prior_csv))
        config["processed_csv_path"] = str(Path(args.prior_csv))
    if args.prior_repeat_fsa:
        config["empirical_prior_repeat_fsa_path"] = str(Path(args.prior_repeat_fsa))
    if args.prior_spacer_fsa:
        config["empirical_prior_spacer_fsa_path"] = str(Path(args.prior_spacer_fsa))
    return config


def validate_group_sizes(metadata: dict, expected_array_count: int) -> None:
    observed_sizes = [len(group.get("strain_ids", [])) for group in metadata.get("groups", [])]
    bad_sizes = sorted({size for size in observed_sizes if size != int(expected_array_count)})
    if bad_sizes:
        raise ValueError(
            f"Expected every group to have {expected_array_count} arrays, "
            f"but observed group sizes: {bad_sizes}"
        )


def write_summary(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, dict] = {}

    for array_count in args.array_counts:
        if int(array_count) <= 0:
            raise ValueError(f"array count must be positive, got {array_count}")

        config = build_config(args, int(array_count), output_root)
        if args.force:
            config["simulation_use_cache"] = False

        per_count_summary: dict[str, dict] = {}
        for split_name in ("train", "valid", "test"):
            strains_data, metadata = load_split_strains(split_name, config, read_spacers_from_csv)
            validate_group_sizes(metadata, int(array_count))

            cache_path = Path(metadata.get("cache_path", ""))
            per_count_summary[split_name] = {
                "cache_path": str(cache_path),
                "num_strains": len(strains_data),
                "num_groups": len(metadata.get("groups", [])),
                "mean_arrays_per_strain": metadata.get("mean_arrays_per_strain"),
                "expected_arrays_per_group": int(array_count),
            }
            print(
                f"[arrays={array_count}] split={split_name} "
                f"groups={per_count_summary[split_name]['num_groups']} "
                f"strains={per_count_summary[split_name]['num_strains']} "
                f"cache={cache_path}"
            )

        count_output_dir = output_root / f"arrays_{array_count}"
        write_summary(count_output_dir, per_count_summary)
        all_summaries[str(array_count)] = per_count_summary

    write_summary(output_root, all_summaries)


if __name__ == "__main__":
    main()
