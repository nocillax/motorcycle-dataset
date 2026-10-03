#!/usr/bin/env python3
"""
Finalize the Roboflow six-class YOLO export into the canonical five-class
public dataset schema.

Current/internal schema can contain:
    ambiguous_rider
    driver_helmet
    driver_no_helmet
    motorcycle
    passenger_no_helmet
    passenger_helmet

Final/public schema MUST be:
    0 motorcycle
    1 driver_helmet
    2 driver_no_helmet
    3 passenger_helmet
    4 passenger_no_helmet

This script maps by CLASS NAME, not by hard-coded current Roboflow IDs.
Therefore it is safe if Roboflow assigns a different six-class ordering.

IMPORTANT:
- Pass B1 and Pass B2 must already be complete.
- There must be NO `ambiguous_rider` annotations left in YOLO label files.
- The script refuses to modify anything if an ambiguous_rider label remains.
- `images.jsonl` is NOT modified.
- The original-source `sha256` in images.jsonl is NOT modified.
- Label files and data.yaml are backed up under the reports directory before
  they are changed.

Usage:
    python scripts/finalize_yolo_dataset.py

Dry run:
    python scripts/finalize_yolo_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


FINAL_CLASSES = [
    "motorcycle",
    "driver_helmet",
    "driver_no_helmet",
    "passenger_helmet",
    "passenger_no_helmet",
]

FINAL_IDS = {
    name: index
    for index, name in enumerate(FINAL_CLASSES)
}

AMBIGUOUS_CLASS = "ambiguous_rider"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_no}"
                )
            rows.append(value)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    tmp.replace(path)


def load_names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(
        yaml_path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML is not a mapping: {yaml_path}"
        )

    names = data.get("names")

    if isinstance(names, list):
        return {
            index: str(name)
            for index, name in enumerate(names)
        }

    if isinstance(names, dict):
        return {
            int(index): str(name)
            for index, name in names.items()
        }

    raise ValueError(
        f"Could not parse 'names' from {yaml_path}"
    )


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(
        yaml_path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML is not a mapping: {yaml_path}"
        )

    return data


def find_labels(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.glob("**/labels/*.txt")
        if path.is_file()
    )


def backup_file(
    source: Path,
    backup_root: Path,
    dataset_root: Path,
) -> Path:
    relative = source.relative_to(dataset_root)
    destination = backup_root / relative
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(source, destination)
    return destination


def remap_label_text(
    text: str,
    names: dict[int, str],
    source_path: Path,
) -> tuple[str, dict[str, int]]:
    output_lines: list[str] = []
    counts: dict[str, int] = {}

    for line_no, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) != 5:
            raise ValueError(
                f"{source_path}:{line_no}: expected 5 fields, "
                f"found {len(parts)}"
            )

        try:
            old_id = int(parts[0])
            for value in parts[1:]:
                float(value)
        except ValueError as exc:
            raise ValueError(
                f"{source_path}:{line_no}: invalid YOLO value"
            ) from exc

        if old_id not in names:
            raise ValueError(
                f"{source_path}:{line_no}: class ID {old_id} "
                "does not exist in the current YAML"
            )

        class_name = names[old_id]

        if class_name == AMBIGUOUS_CLASS:
            raise ValueError(
                f"{source_path}:{line_no}: unresolved 'ambiguous_rider' "
                "label remains. Finish Pass B2 before finalization."
            )

        if class_name not in FINAL_IDS:
            raise ValueError(
                f"{source_path}:{line_no}: unsupported class "
                f"{class_name!r}"
            )

        new_id = FINAL_IDS[class_name]
        parts[0] = str(new_id)
        output_lines.append(" ".join(parts))

        counts[class_name] = (
            counts.get(class_name, 0) + 1
        )

    output = (
        "\n".join(output_lines)
        + ("\n" if output_lines else "")
    )

    return output, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Roboflow six-class export into the "
            "canonical five-class YOLO dataset."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
    )

    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("dataset/data.yaml"),
    )

    parser.add_argument(
        "--motorcycles-jsonl",
        type=Path,
        default=Path("metadata/motorcycles.jsonl"),
        help=(
            "Motorcycle metadata. Resolved rider class IDs are remapped to "
            "the canonical five-class IDs during finalization."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/finalization_report.json"),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show the proposed remapping without changing files.",
    )

    args = parser.parse_args()

    if not args.dataset_root.exists():
        print(
            f"ERROR: dataset root does not exist: {args.dataset_root}"
        )
        return 1

    if not args.yaml.exists():
        print(
            f"ERROR: YAML does not exist: {args.yaml}"
        )
        return 1

    if not args.motorcycles_jsonl.exists():
        print(
            f"ERROR: motorcycles.jsonl does not exist: {args.motorcycles_jsonl}"
        )
        return 1

    try:
        names = load_names(args.yaml)
        yaml_data = load_yaml(args.yaml)
        motorcycles = load_jsonl(args.motorcycles_jsonl)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    reverse = {
        class_name: class_id
        for class_id, class_name in names.items()
    }

    missing = [
        name
        for name in FINAL_CLASSES
        if name not in reverse
    ]

    if missing:
        print(
            "ERROR: the current YAML is missing final class(es):"
        )
        for name in missing:
            print(f"  - {name}")
        return 1

    if AMBIGUOUS_CLASS not in reverse:
        print(
            "ERROR: current YAML does not contain 'ambiguous_rider'."
        )
        print(
            "Expected the six-class Roboflow export before finalization."
        )
        return 1

    print("Current class schema:")
    for class_id, class_name in sorted(names.items()):
        print(f"  {class_id}: {class_name}")

    print("\nFinal class schema:")
    for class_id, class_name in enumerate(FINAL_CLASSES):
        print(f"  {class_id}: {class_name}")

    print("\nClass-ID mapping:")
    for class_name in names.values():
        if class_name == AMBIGUOUS_CLASS:
            print(
                f"  {class_name}: MUST BE ABSENT FROM LABEL FILES"
            )
        elif class_name in FINAL_IDS:
            print(
                f"  {class_name}: "
                f"{reverse[class_name]} -> {FINAL_IDS[class_name]}"
            )

    label_files = find_labels(args.dataset_root)

    if not label_files:
        print(
            f"ERROR: no label files under {args.dataset_root}/**/labels/"
        )
        return 1

    remapped_files = 0
    total_boxes = 0
    class_counts: dict[str, int] = {}
    proposed: dict[Path, str] = {}

    try:
        for label_path in label_files:
            original = label_path.read_text(
                encoding="utf-8"
            )

            remapped, counts = remap_label_text(
                original,
                names,
                label_path,
            )

            proposed[label_path] = remapped

            if remapped != original:
                remapped_files += 1

            for class_name, count in counts.items():
                class_counts[class_name] = (
                    class_counts.get(class_name, 0)
                    + count
                )
                total_boxes += count

    except Exception as exc:
        print(f"ERROR: {exc}")
        print("No files have been modified.")
        return 1

    final_yaml_names = {
        index: class_name
        for index, class_name in enumerate(FINAL_CLASSES)
    }

    print(
        f"\nLabel files scanned: {len(label_files)}"
    )
    print(
        f"Label files requiring remap: {remapped_files}"
    )
    print(
        f"Total YOLO boxes: {total_boxes}"
    )

    print("Final class counts:")
    for class_name in FINAL_CLASSES:
        print(
            f"  {class_name}: "
            f"{class_counts.get(class_name, 0)}"
        )

    # Remap resolved_class_id in motorcycle metadata by NAME.
    # Ordinary riders may intentionally have resolved_class_id=null because
    # their class_name is already a final class; only explicit resolved_name
    # values need updating here.
    metadata_changes = 0

    for motorcycle in motorcycles:
        for rider in motorcycle.get("riders", []):
            resolved_name = rider.get("resolved_class_name")

            if resolved_name in FINAL_IDS:
                new_id = FINAL_IDS[resolved_name]
                if rider.get("resolved_class_id") != new_id:
                    rider["resolved_class_id"] = new_id
                    metadata_changes += 1

            elif resolved_name is not None:
                raise ValueError(
                    f"Invalid resolved_class_name {resolved_name!r} in "
                    f"{motorcycle.get('motorcycle_id')}/{rider.get('rider_id')}"
                )

    print(
        f"Resolved rider class IDs requiring metadata remap: {metadata_changes}"
    )

    if args.dry_run:
        print("\nDRY RUN: no files modified.")
        return 0

    backup_root = (
        args.report.parent
        / "finalization_backup"
    )

    yaml_backup = backup_root / args.yaml.name
    yaml_backup.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        args.yaml,
        yaml_backup,
    )

    metadata_backup = backup_root / args.motorcycles_jsonl.name
    shutil.copy2(
        args.motorcycles_jsonl,
        metadata_backup,
    )

    changed_files: list[str] = []

    for label_path, new_text in proposed.items():
        original = label_path.read_text(
            encoding="utf-8"
        )

        if new_text == original:
            continue

        backup_file(
            label_path,
            backup_root,
            args.dataset_root,
        )

        tmp = label_path.with_suffix(
            label_path.suffix + ".tmp"
        )

        tmp.write_text(
            new_text,
            encoding="utf-8",
        )

        tmp.replace(label_path)
        changed_files.append(
            str(label_path)
        )

    # Replace only the class definition in the existing YAML, while keeping
    # paths and any other Roboflow fields intact.
    yaml_data["names"] = final_yaml_names

    if "nc" in yaml_data:
        yaml_data["nc"] = len(FINAL_CLASSES)

    yaml_tmp = args.yaml.with_suffix(
        args.yaml.suffix + ".tmp"
    )

    yaml_tmp.write_text(
        yaml.safe_dump(
            yaml_data,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    yaml_tmp.replace(args.yaml)

    # Keep motorcycles.jsonl internally consistent with the final class IDs.
    write_jsonl(
        args.motorcycles_jsonl,
        motorcycles,
    )

    report = {
        "status": "PASS",
        "dry_run": False,
        "source_yaml": str(args.yaml),
        "final_names": final_yaml_names,
        "original_names": names,
        "label_files_scanned": len(label_files),
        "label_files_remapped": len(changed_files),
        "total_yolo_boxes": total_boxes,
        "class_counts": class_counts,
        "backup_root": str(backup_root),
        "changed_files": changed_files,
        "motorcycles_metadata_resolved_id_updates": metadata_changes,
        "note": (
            "images.jsonl and its original-source sha256 values were "
            "not modified. motorcycles.jsonl resolved_class_id values were "
            "normalized to the final five-class IDs."
        ),
    }

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nFINALIZATION COMPLETE")
    print(
        f"Canonical YAML written: {args.yaml}"
    )
    print(
        f"Backup written to: {backup_root}"
    )
    print(
        f"Report: {args.report}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
