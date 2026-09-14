#!/usr/bin/env python3
"""Analyze CNN/ViT/YOLO DrugBank consensus after a heavy-atom cutoff.

The script validates the six molecule-level prediction tables, applies a common
cohort, and prints machine-readable JSON. It does not modify the source CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


MODELS = ("cnn", "vit", "yolo")
TARGETS = {
    "CDK2": "CHEMBL301",
    "AKT1": "CHEMBL4282",
}
THRESHOLDS = (
    ("ge1", ">=1", lambda value: value >= 1),
    ("ge18", ">=18", lambda value: value >= 18),
    ("ge32", ">=32", lambda value: value >= 32),
    ("eq36", "=36", lambda value: value == 36),
)


def read_predictions(path: Path) -> dict[str, dict[str, int | bool]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "molecule_chembl_id",
            "active_rotations",
            "total_rotations",
            "complete_rotations",
            "n_heavy_atoms",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        records: dict[str, dict[str, int | bool]] = {}
        for line_number, row in enumerate(reader, start=2):
            molecule_id = row["molecule_chembl_id"].strip()
            if not molecule_id:
                raise ValueError(f"{path}:{line_number}: empty molecule ID")
            if molecule_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate {molecule_id}")
            active = int(row["active_rotations"])
            total = int(row["total_rotations"])
            complete = row["complete_rotations"].strip().lower() == "true"
            heavy_atoms = int(row["n_heavy_atoms"])
            if not 0 <= active <= total:
                raise ValueError(
                    f"{path}:{line_number}: invalid rotations {active}/{total}"
                )
            records[molecule_id] = {
                "active_rotations": active,
                "total_rotations": total,
                "complete_rotations": complete,
                "n_heavy_atoms": heavy_atoms,
            }
    return records


def analyze(input_dir: Path, heavy_atom_max: int) -> dict[str, object]:
    loaded: dict[tuple[str, str], dict[str, dict[str, int | bool]]] = {}
    for target_id in TARGETS.values():
        for model in MODELS:
            path = input_dir / f"predictions_{target_id}_{model}.csv"
            loaded[(target_id, model)] = read_predictions(path)

    all_id_sets = [set(records) for records in loaded.values()]
    common_ids = set.intersection(*all_id_sets)
    union_ids = set.union(*all_id_sets)
    if common_ids != union_ids:
        sizes = {
            f"{target_id}_{model}": len(loaded[(target_id, model)])
            for target_id in TARGETS.values()
            for model in MODELS
        }
        raise ValueError(
            "Prediction files do not share one identical molecule cohort: "
            f"intersection={len(common_ids)}, union={len(union_ids)}, sizes={sizes}"
        )

    incomplete = []
    total_rotation_counts: Counter[int] = Counter()
    heavy_atom_mismatches = []
    eligible_ids = []
    for molecule_id in sorted(common_ids):
        molecule_records = [records[molecule_id] for records in loaded.values()]
        totals = {int(record["total_rotations"]) for record in molecule_records}
        total_rotation_counts.update(totals)
        if totals != {36} or not all(
            bool(record["complete_rotations"]) for record in molecule_records
        ):
            incomplete.append(molecule_id)
        heavy_atom_values = {
            int(record["n_heavy_atoms"]) for record in molecule_records
        }
        if len(heavy_atom_values) != 1:
            heavy_atom_mismatches.append(
                {"molecule_id": molecule_id, "values": sorted(heavy_atom_values)}
            )
            continue
        if next(iter(heavy_atom_values)) <= heavy_atom_max:
            eligible_ids.append(molecule_id)

    if incomplete:
        raise ValueError(f"Found {len(incomplete)} incomplete 36-rotation molecules")
    if heavy_atom_mismatches:
        raise ValueError(
            f"Found {len(heavy_atom_mismatches)} heavy-atom mismatches; "
            f"first={heavy_atom_mismatches[0]}"
        )

    result: dict[str, object] = {
        "validation": {
            "source_rows_per_file": len(common_ids),
            "identical_molecule_cohort": True,
            "complete_36_rotations": True,
            "identical_heavy_atom_values": True,
            "heavy_atom_max": heavy_atom_max,
            "eligible_molecules": len(eligible_ids),
            "excluded_molecules": len(common_ids) - len(eligible_ids),
        },
        "targets": {},
    }

    targets_result: dict[str, object] = {}
    for target_name, target_id in TARGETS.items():
        target_rows = []
        threshold_results: dict[str, object] = {}
        for threshold_key, threshold_label, predicate in THRESHOLDS:
            individual_ids: dict[str, list[str]] = {}
            vote_groups = {"exactly_2_of_3": [], "three_of_3": []}
            for model in MODELS:
                individual_ids[model] = [
                    molecule_id
                    for molecule_id in eligible_ids
                    if predicate(
                        int(
                            loaded[(target_id, model)][molecule_id][
                                "active_rotations"
                            ]
                        )
                    )
                ]
            for molecule_id in eligible_ids:
                passing_models = [
                    model
                    for model in MODELS
                    if predicate(
                        int(
                            loaded[(target_id, model)][molecule_id][
                                "active_rotations"
                            ]
                        )
                    )
                ]
                if len(passing_models) == 2:
                    vote_groups["exactly_2_of_3"].append(molecule_id)
                elif len(passing_models) == 3:
                    vote_groups["three_of_3"].append(molecule_id)

            threshold_results[threshold_key] = {
                "label": threshold_label,
                "individual_model_counts": {
                    model: len(ids) for model, ids in individual_ids.items()
                },
                "exactly_2_of_3_count": len(vote_groups["exactly_2_of_3"]),
                "three_of_3_count": len(vote_groups["three_of_3"]),
                "at_least_2_of_3_count": sum(len(ids) for ids in vote_groups.values()),
                "exactly_2_of_3_ids": vote_groups["exactly_2_of_3"],
                "three_of_3_ids": vote_groups["three_of_3"],
            }

        for molecule_id in eligible_ids:
            rotations = {
                model: int(
                    loaded[(target_id, model)][molecule_id]["active_rotations"]
                )
                for model in MODELS
            }
            threshold_votes = {
                threshold_key: sum(predicate(value) for value in rotations.values())
                for threshold_key, _, predicate in THRESHOLDS
            }
            if threshold_votes["ge1"] >= 2:
                target_rows.append(
                    {
                        "target": target_name,
                        "target_chembl_id": target_id,
                        "molecule_id": molecule_id,
                        "n_heavy_atoms": int(
                            loaded[(target_id, "cnn")][molecule_id]["n_heavy_atoms"]
                        ),
                        "cnn_active_rotations": rotations["cnn"],
                        "vit_active_rotations": rotations["vit"],
                        "yolo_active_rotations": rotations["yolo"],
                        **{
                            f"models_{threshold_key}": votes
                            for threshold_key, votes in threshold_votes.items()
                        },
                    }
                )

        targets_result[target_name] = {
            "target_chembl_id": target_id,
            "thresholds": threshold_results,
            "candidate_rows": target_rows,
        }
    result["targets"] = targets_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path.cwd())
    parser.add_argument("--heavy-atom-max", type=int, default=45)
    parser.add_argument("--indent", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input_dir, args.heavy_atom_max), indent=args.indent))


if __name__ == "__main__":
    main()
