from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List


def normalize_repeat_zwx(repeat: str) -> str:
    # Normalize repeat sequences, using a fixed placeholder for empty values.

    repeat = str(repeat or "").strip().upper()
    return repeat if repeat else "UNKNOWN_REPEAT"


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
