from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path("/kaggle/working")
LOCAL_SIMULATION_ROOT = PROJECT_ROOT / "simulated_cache"

KAGGLE_DATA_ROOT = Path("/kaggle/input/datasets/ahsinzhao/cronobacter")


def _pick_optional_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


TRAIN_CSV = PROJECT_ROOT / "train_spacers.csv"
VALID_CSV = PROJECT_ROOT / "valid_spacers.csv"
PROCESSED_CSV = KAGGLE_DATA_ROOT / "all_spacers_evidence4_repeat_process.csv"
EMPIRICAL_PRIOR_REPEAT_FSA = _pick_optional_existing(
    PROJECT_ROOT / "direct_repeat_id_no_Salmon_Crono.fsa"
)
EMPIRICAL_PRIOR_SPACER_FSA = _pick_optional_existing(
    PROJECT_ROOT / "spacer_id_no_Salmon_Crono.fsa"
)

PATHS = {
    "processed_csv_path": str(PROCESSED_CSV),
    "empirical_prior_repeat_fsa_path": str(EMPIRICAL_PRIOR_REPEAT_FSA or ""),
    "empirical_prior_spacer_fsa_path": str(EMPIRICAL_PRIOR_SPACER_FSA or ""),
    "train_csv_path": str(TRAIN_CSV),
    "valid_csv_path": str(VALID_CSV),
    "simulation_cache_dir": str(LOCAL_SIMULATION_ROOT),
}


def _expanded_group_size_distribution() -> list[int]:
    """
    Approximate a long-tailed group-size prior.

    The simulator samples uniformly from this list, so repeated values act as
    integer weights. Most groups remain small or medium, while sizes up to 500
    are present often enough to train the long-input path.
    """

    small_sizes = [3, 4, 5, 6, 8, 10, 12, 16, 20]
    medium_sizes = [24, 28, 32, 40, 50, 64, 80, 96, 100, 128]
    large_sizes = [160, 192, 256, 300, 320, 384, 400, 448, 500]
    return (
        [size for size in small_sizes for _ in range(40)]
        + [size for size in medium_sizes for _ in range(20)]
        + [size for size in large_sizes for _ in range(1)]
    )

TRAIN_CONFIG = {
    "data_source": "simulated",
    "train_csv_path": PATHS["train_csv_path"],
    "valid_csv_path": PATHS["valid_csv_path"],
    "processed_csv_path": PATHS["processed_csv_path"],
    "empirical_prior_csv_path": PATHS["processed_csv_path"],
    "empirical_prior_repeat_fsa_path": PATHS["empirical_prior_repeat_fsa_path"],
    "empirical_prior_spacer_fsa_path": PATHS["empirical_prior_spacer_fsa_path"],
    "simulation_cache_dir": PATHS["simulation_cache_dir"],
    "simulation_use_cache": True,
    "simulation_cache_version": "v7_group_size_50_longtail",
    "simulation_seed": 42,
    "simulation_train_num_strains": 10000,
    "simulation_valid_num_strains": 1250,
    "simulation_test_num_strains": 1250,
    "simulation_train_num_groups": 550,
    "simulation_valid_num_groups": 100,
    "simulation_test_num_groups": 100,
    "simulation_target_arrays_per_strain": 2.2,
    "simulation_min_group_size": 3,
    "simulation_max_group_size": 500,
    "simulation_tree_type": "coalescent",
    "simulation_group_size_distribution": _expanded_group_size_distribution(),
    "simulation_target_mean_array_length_distribution": [10, 15, 20, 25, 30],
    "simulation_spacer_length_histogram": {
        30: 31149,
        31: 4079,
        32: 91342,
        33: 35441,
        34: 28411,
        35: 35883,
        36: 44157,
        37: 30347,
        38: 16389,
        39: 9486,
        40: 6628,
        41: 4513,
        42: 2903,
        43: 2024,
    },
    "simulation_repeat_length_histogram": {
        23: 170,
        24: 512,
        25: 573,
        26: 268,
        27: 382,
        28: 4225,
        29: 7018,
        30: 3396,
        31: 538,
        32: 1844,
        33: 582,
        34: 163,
        35: 527,
        36: 4133,
        37: 3612,
        38: 203,
        39: 30,
        40: 15,
    },
    "simulation_theta_min": 1.0,
    "simulation_theta_max": 3.0,
    "simulation_alpha_min": 1.5,
    "simulation_alpha_max": 4.0,
    "simulation_deletion_model_mixture": {
        "BDM": 0.8,
        "IDM": 0.2,
    },
    "simulation_enable_boundary_effect": True,
    "simulation_allow_rare_complex_events": True,
    "simulation_duplication_rate": 0.03,
    "simulation_rearrangement_rate": 0.01,
    "simulation_reacquisition_rate": 0.01,
    "simulation_independent_deletion_rate": 0.10,
    "simulation_max_new_spacers_per_branch": 3,
    "simulation_min_shared_spacers": 1,
    "simulation_group_overlap_retry": 100,
}
