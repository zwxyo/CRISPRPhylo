from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ArrayPath:
    array_uid: str
    strain_id: str
    array_id: str
    repeat_family: str
    spacer_sequences: list[str]
    spacer_class_ids: list[str]
    candidate_group_id: str


def normalize_repeat_family(array_row: dict) -> str:
    repeat_family = str(array_row.get("repeat_family") or array_row.get("repeat") or "").strip().upper()
    return repeat_family

# Convert raw arrays into path objects.
def convert_strains_to_array_paths(strains_data: list[dict]) -> list[ArrayPath]:
    array_paths: list[ArrayPath] = []
    for strain in strains_data:
        strain_id = str(strain["strain_id"])
        for array in strain["arrays"]:
            array_paths.append(
                ArrayPath(
                    array_uid=str(array["array_uid"]),
                    strain_id=strain_id,
                    array_id=str(array["array_id"]),
                    repeat_family=normalize_repeat_family(array),
                    spacer_sequences=[str(spacer) for spacer in array["spacers"]],
                    spacer_class_ids=[],
                    candidate_group_id=str(array["candidate_group_id"]),
                )
            )
    return array_paths

# Group by repeat family.
def group_array_paths_by_repeat_family(array_paths: list[ArrayPath]) -> dict[str, list[ArrayPath]]:
    grouped: dict[str, list[ArrayPath]] = {}
    for array_path in array_paths:
        grouped.setdefault(array_path.repeat_family, []).append(array_path)
    return grouped


def attach_spacer_class_ids(
    array_paths: list[ArrayPath],
    sequence_to_class_id: dict[str, str],
) -> list[ArrayPath]:
    annotated_paths: list[ArrayPath] = []
    for array_path in array_paths:
        spacer_class_ids = [sequence_to_class_id[sequence] for sequence in array_path.spacer_sequences]
        annotated_paths.append(replace(array_path, spacer_class_ids=spacer_class_ids))
    return annotated_paths


def build_array_path_lookup(array_paths: list[ArrayPath]) -> dict[str, ArrayPath]:
    return {array_path.array_uid: array_path for array_path in array_paths}
