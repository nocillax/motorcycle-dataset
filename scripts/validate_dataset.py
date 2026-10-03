#!/usr/bin/env python3

"""
Final dataset validator.

Run this after Pass D.

Default project layout:
    dataset/
        train/images/
        train/labels/
        valid/images/
        valid/labels/
        test/images/
        test/labels/
        data.yaml
    metadata/images.jsonl
    metadata/motorcycles.jsonl
    reports/

Usage:
    python scripts/validate_dataset.py

Optional:
    python scripts/validate_dataset.py \
        --dataset-root dataset \
        --images-jsonl metadata/images.jsonl \
        --motorcycles-jsonl metadata/motorcycles.jsonl \
        --yaml dataset/data.yaml \
        --report reports/validation_report.json

The final public dataset must contain exactly these five classes, in order:
    0 motorcycle
    1 driver_helmet
    2 driver_no_helmet
    3 passenger_helmet
    4 passenger_no_helmet

The internal `ambiguous_rider` class is NOT allowed in final YOLO labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    raise


FINAL_CLASSES = [
    "motorcycle",
    "driver_helmet",
    "driver_no_helmet",
    "passenger_helmet",
    "passenger_no_helmet",
]

EXPECTED_CLASS_IDS = {
    name: index
    for index, name in enumerate(FINAL_CLASSES)
}

VALID_ILLUMINATION = {
    "daylight",
    "dusk",
    "night_artificial",
    "dark_unlit",
}

VALID_WEATHER = {
    "clear",
    "rain",
    "fog",
}

VALID_ROAD_CONDITION = {
    "dry",
    "wet",
}

VALID_GLARE = {
    "none",
    "mild",
    "moderate",
    "severe",
}

VALID_VISIBILITY = {
    "clear",
    "moderate",
    "poor",
}

VALID_MOTION_BLUR = {
    "none",
    "mild",
    "moderate",
    "severe",
}

VALID_TRAFFIC_DENSITY = {
    "low",
    "medium",
    "high",
}

VALID_TRAFFIC_CONTEXT = {
    "free_flow",
    "slow_traffic",
    "traffic_jam",
    "stopped_intersection",
    "uncertain",
}

VALID_ROAD_TYPE = {
    "urban_arterial",
    "intersection",
    "alley",
    "highway_main_road",
    "residential",
    "other",
}

VALID_VIEWPOINT = {
    "front",
    "rear",
    "side_left",
    "side_right",
    "oblique",
    "uncertain",
}

VALID_OCCLUSION = {
    "none",
    "partial",
    "heavy",
}

VALID_APPARENT_SCALE = {
    "large",
    "medium",
    "small",
    "very_small",
}

VALID_IMAGE_STATUS = {
    "verified",
}

VALID_ASSOCIATION_STATUS = {
    "auto_candidate",
    "verified",
}

VALID_CLASSIFICATION_STATUS = {
    "verified",
}

VALID_RIDER_ROLE = {
    "driver",
    "passenger",
    "unknown",
}

VALID_RIDER_HELMET = {
    True,
    False,
    "unknown",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


class Validator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict[str, Any] = {}

        self.image_files: dict[str, Path] = {}
        self.label_files: dict[str, Path] = {}
        self.images: list[dict[str, Any]] = []
        self.motorcycles: list[dict[str, Any]] = []
        self.image_by_id: dict[str, dict[str, Any]] = {}
        self.motorcycles_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.yaml_names: dict[int, str] = {}
        self.class_ids: dict[str, int] = {}
        self.label_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # --------------------------------------------------------
    # Generic helpers
    # --------------------------------------------------------

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    @staticmethod
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
                        f"Expected JSON object in {path} at line {line_no}."
                    )
                rows.append(value)
        return rows

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def as_box4(value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def center_inside(box: list[float], parent: list[float], tolerance: float = 1e-6) -> bool:
        x, y, _, _ = box
        px, py, pw, ph = parent
        px1 = px - pw / 2.0
        py1 = py - ph / 2.0
        px2 = px + pw / 2.0
        py2 = py + ph / 2.0
        return (
            x >= px1 - tolerance
            and x <= px2 + tolerance
            and y >= py1 - tolerance
            and y <= py2 + tolerance
        )

    @staticmethod
    def close_enough(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - b) <= tol

    @staticmethod
    def canonical_image_id(stem: str) -> str:
        # Keep the same canonical IDs used by the pipeline. The validator
        # only needs to handle Roboflow's common *_jpg.rf.* form here.
        marker = "_jpg.rf."
        marker2 = "_jpeg.rf."
        marker3 = "_png.rf."
        lower = stem.lower()
        for m in (marker, marker2, marker3):
            idx = lower.find(m)
            if idx >= 0:
                return stem[:idx]
        return stem

    # --------------------------------------------------------
    # Discovery / loading
    # --------------------------------------------------------

    def discover_files(self) -> None:
        dataset_root = self.args.dataset_root

        if not dataset_root.exists():
            self.error(f"Dataset root does not exist: {dataset_root}")
            return

        for path in dataset_root.glob("**/images/*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_id = self.canonical_image_id(path.stem)
                if image_id in self.image_files:
                    self.error(
                        f"Duplicate image_id discovered: {image_id} "
                        f"({self.image_files[image_id]} and {path})"
                    )
                else:
                    self.image_files[image_id] = path

        for path in dataset_root.glob("**/labels/*.txt"):
            if path.is_file():
                image_id = self.canonical_image_id(path.stem)
                if image_id in self.label_files:
                    self.error(
                        f"Duplicate label/image_id discovered: {image_id} "
                        f"({self.label_files[image_id]} and {path})"
                    )
                else:
                    self.label_files[image_id] = path

        if not self.image_files:
            self.error(f"No images found under {dataset_root}/**/images/")

        if not self.label_files:
            self.error(f"No YOLO labels found under {dataset_root}/**/labels/")

    def load_metadata(self) -> None:
        try:
            self.images = self.load_jsonl(self.args.images_jsonl)
        except Exception as exc:
            self.error(str(exc))
            return

        try:
            self.motorcycles = self.load_jsonl(self.args.motorcycles_jsonl)
        except Exception as exc:
            self.error(str(exc))
            return

        for row in self.images:
            image_id = row.get("image_id")
            if not image_id:
                self.error("images.jsonl contains a record with no image_id.")
                continue
            if image_id in self.image_by_id:
                self.error(f"Duplicate image_id in images.jsonl: {image_id}")
            else:
                self.image_by_id[image_id] = row

        for row in self.motorcycles:
            image_id = row.get("image_id")
            motorcycle_id = row.get("motorcycle_id")
            if not image_id:
                self.error(
                    f"motorcycles.jsonl record {motorcycle_id!r} has no image_id."
                )
                continue
            if not motorcycle_id:
                self.error(
                    f"motorcycles.jsonl record for {image_id} has no motorcycle_id."
                )
                continue
            self.motorcycles_by_image[image_id].append(row)

    def load_yaml(self) -> None:
        try:
            data = yaml.safe_load(
                self.args.yaml.read_text(encoding="utf-8")
            )
        except Exception as exc:
            self.error(f"Could not read YAML {self.args.yaml}: {exc}")
            return

        if not isinstance(data, dict):
            self.error(f"YAML is not a mapping: {self.args.yaml}")
            return

        names = data.get("names")
        try:
            if isinstance(names, list):
                self.yaml_names = {
                    i: str(name)
                    for i, name in enumerate(names)
                }
            elif isinstance(names, dict):
                self.yaml_names = {
                    int(k): str(v)
                    for k, v in names.items()
                }
            else:
                raise ValueError("'names' must be a list or dictionary.")
        except Exception as exc:
            self.error(f"Invalid class names in {self.args.yaml}: {exc}")
            return

        self.class_ids = {
            name: class_id
            for class_id, name in self.yaml_names.items()
        }

        expected_names = {
            i: name
            for i, name in enumerate(FINAL_CLASSES)
        }

        if self.yaml_names != expected_names:
            self.error(
                "Final YAML class schema is not exactly:\n"
                + json.dumps(expected_names, ensure_ascii=False)
                + "\n"
                + f"Found: {self.yaml_names}"
            )

    # --------------------------------------------------------
    # File parity
    # --------------------------------------------------------

    def validate_file_parity(self) -> None:
        image_ids = set(self.image_files)
        label_ids = set(self.label_files)
        metadata_ids = set(self.image_by_id)
        motorcycle_image_ids = set(self.motorcycles_by_image)

        for image_id in sorted(image_ids - label_ids):
            self.error(f"Image has no YOLO label: {image_id}")

        for image_id in sorted(label_ids - image_ids):
            self.error(f"YOLO label has no image: {image_id}")

        for image_id in sorted(image_ids - metadata_ids):
            self.error(f"Image has no images.jsonl record: {image_id}")

        for image_id in sorted(metadata_ids - image_ids):
            self.error(f"images.jsonl record has no image file: {image_id}")

        for image_id in sorted(motorcycle_image_ids - image_ids):
            self.error(
                f"motorcycles.jsonl contains image_id with no image file: {image_id}"
            )

        self.stats["image_files"] = len(image_ids)
        self.stats["label_files"] = len(label_ids)
        self.stats["image_metadata_records"] = len(metadata_ids)
        self.stats["motorcycle_metadata_records"] = len(self.motorcycles)

    # --------------------------------------------------------
    # YOLO labels
    # --------------------------------------------------------

    def validate_labels(self) -> None:
        if not self.yaml_names:
            return

        class_counts = Counter()
        total_boxes = 0

        for image_id, label_path in self.label_files.items():
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                self.error(f"Could not read label {label_path}: {exc}")
                continue

            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) != 5:
                    self.error(
                        f"{label_path}:{line_no}: expected 5 fields, found {len(parts)}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = [float(v) for v in parts[1:]]
                except ValueError:
                    self.error(
                        f"{label_path}:{line_no}: non-numeric YOLO value"
                    )
                    continue

                total_boxes += 1
                class_counts[class_id] += 1

                if class_id not in range(5):
                    self.error(
                        f"{label_path}:{line_no}: invalid final class ID {class_id}; "
                        "only 0-4 are allowed."
                    )

                for name, value in (
                    ("x_center", x),
                    ("y_center", y),
                    ("width", w),
                    ("height", h),
                ):
                    if not (0.0 <= value <= 1.0):
                        self.error(
                            f"{label_path}:{line_no}: {name}={value} outside [0,1]"
                        )

                if w <= 0 or h <= 0:
                    self.error(
                        f"{label_path}:{line_no}: width/height must be > 0"
                    )

                self.label_records[image_id].append(
                    {
                        "class_id": class_id,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "line_no": line_no,
                    }
                )

        self.stats["total_yolo_boxes"] = total_boxes
        self.stats["class_counts"] = {
            str(class_id): count
            for class_id, count in sorted(class_counts.items())
        }

    # --------------------------------------------------------
    # Scene metadata
    # --------------------------------------------------------

    def validate_image_metadata(self) -> None:
        exact_fields = {
            "illumination": VALID_ILLUMINATION,
            "weather": VALID_WEATHER,
            "road_condition": VALID_ROAD_CONDITION,
            "glare": VALID_GLARE,
            "visibility": VALID_VISIBILITY,
            "motion_blur": VALID_MOTION_BLUR,
            "traffic_density": VALID_TRAFFIC_DENSITY,
            "traffic_context": VALID_TRAFFIC_CONTEXT,
            "road_type": VALID_ROAD_TYPE,
        }

        missing_fields_by_image: Counter[str] = Counter()

        for image_id, row in self.image_by_id.items():
            if row.get("annotation_status") != "verified":
                self.error(
                    f"{image_id}: annotation_status must be 'verified', "
                    f"found {row.get('annotation_status')!r}"
                )

            if row.get("association_status") not in VALID_ASSOCIATION_STATUS:
                self.error(
                    f"{image_id}: association_status must be one of "
                    f"{sorted(VALID_ASSOCIATION_STATUS)}, "
                    f"found {row.get('association_status')!r}"
                )

            if row.get("rider_classification_status") != "verified":
                self.error(
                    f"{image_id}: rider_classification_status must be 'verified', "
                    f"found {row.get('rider_classification_status')!r}"
                )

            for field, allowed in exact_fields.items():
                value = row.get(field)
                if value is None:
                    missing_fields_by_image[field] += 1
                elif value not in allowed:
                    self.error(
                        f"{image_id}: invalid {field}={value!r}; "
                        f"allowed={sorted(allowed)}"
                    )

            hard_negative = row.get("hard_negative")
            if not isinstance(hard_negative, bool):
                self.error(
                    f"{image_id}: hard_negative must be boolean, "
                    f"found {hard_negative!r}"
                )

            width = row.get("width")
            height = row.get("height")
            image_path = self.image_files.get(image_id)

            if image_path is not None:
                try:
                    from PIL import Image
                    with Image.open(image_path) as image:
                        actual_width, actual_height = image.size
                    if width != actual_width or height != actual_height:
                        self.error(
                            f"{image_id}: metadata dimensions {width}x{height} "
                            f"do not match image {actual_width}x{actual_height}"
                        )
                except ImportError:
                    self.warning(
                        "Pillow not installed; image dimension cross-check skipped."
                    )
                except Exception as exc:
                    self.error(
                        f"{image_id}: could not inspect image dimensions: {exc}"
                    )

            if width is None or height is None:
                self.error(f"{image_id}: width/height missing")
            elif not isinstance(width, int) or not isinstance(height, int):
                self.error(
                    f"{image_id}: width/height must be integers; "
                    f"found {width!r} x {height!r}"
                )

            # Check source provenance fields exist. Exact source values are not
            # imposed here because the source ledger is external to this schema.
            for field in (
                "source_id",
                "filename",
                "original_filename",
                "frame_idx",
                "timestamp_sec",
            ):
                if field not in row:
                    self.error(
                        f"{image_id}: required field missing: {field}"
                    )

        for field, count in sorted(missing_fields_by_image.items()):
            self.warning(
                f"{count} image(s) have {field}=null."
            )

        # `sha256` in images.jsonl records the ORIGINAL extracted frame bytes
        # used by the deduplication stage. Roboflow can re-encode the exported
        # image, so comparing that provenance hash to the current export bytes
        # would create false failures. Validate format + uniqueness here.
        sha_records = {}

        for image_id, row in self.image_by_id.items():
            sha = row.get("sha256")

            if sha:
                if not isinstance(sha, str) or len(sha) != 64:
                    self.error(
                        f"{image_id}: sha256 must be a 64-character hexadecimal string"
                    )
                else:
                    try:
                        int(sha, 16)
                    except ValueError:
                        self.error(
                            f"{image_id}: sha256 contains non-hexadecimal characters"
                        )

                if sha in sha_records:
                    self.error(
                        f"Duplicate sha256 between {sha_records[sha]} and {image_id}"
                    )
                else:
                    sha_records[sha] = image_id
            else:
                self.warning(
                    f"{image_id}: sha256 field is missing; source-frame provenance hash unavailable."
                )

            # Optional final-file hash. If present, THIS field is compared to
            # the bytes currently present in the dataset.
            final_sha = row.get("final_sha256")
            if final_sha:
                path = self.image_files.get(image_id)
                if path is None:
                    continue

                if not isinstance(final_sha, str) or len(final_sha) != 64:
                    self.error(
                        f"{image_id}: final_sha256 must be a 64-character hexadecimal string"
                    )
                    continue

                try:
                    int(final_sha, 16)
                    actual_final_sha = self.sha256(path)
                except ValueError:
                    self.error(
                        f"{image_id}: final_sha256 contains non-hexadecimal characters"
                    )
                    continue
                except Exception as exc:
                    self.error(
                        f"{image_id}: could not calculate final SHA-256: {exc}"
                    )
                    continue

                if final_sha != actual_final_sha:
                    self.error(
                        f"{image_id}: final_sha256 mismatch; metadata={final_sha}, actual={actual_final_sha}"
                    )

        self.stats["sha256_records_checked"] = len(sha_records)

    # --------------------------------------------------------
    # Motorcycle metadata
    # --------------------------------------------------------

    def validate_motorcycle_metadata(self) -> None:
        motorcycle_ids: set[str] = set()
        riders_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        final_class_counts = Counter()
        overloaded_count = 0

        for motorcycle in self.motorcycles:
            mc_id = motorcycle.get("motorcycle_id")
            image_id = motorcycle.get("image_id")

            if not mc_id:
                self.error(
                    f"Motorcycle record in {image_id!r} has no motorcycle_id."
                )
                continue

            if mc_id in motorcycle_ids:
                self.error(
                    f"Duplicate motorcycle_id: {mc_id}"
                )
            motorcycle_ids.add(mc_id)

            if image_id not in self.image_by_id:
                self.error(
                    f"{mc_id}: image_id {image_id!r} missing from images.jsonl"
                )

            expected_prefix = f"MC_{image_id}_"
            if image_id and not str(mc_id).startswith(expected_prefix):
                self.error(
                    f"{mc_id}: motorcycle_id does not match deterministic pattern "
                    f"MC_{image_id}_XX"
                )

            box = self.as_box4(
                motorcycle.get("motorcycle_box")
            )
            if box is None:
                self.error(
                    f"{mc_id}: motorcycle_box must be [x,y,w,h]"
                )
            else:
                if not all(0.0 <= value <= 1.0 for value in box):
                    self.error(
                        f"{mc_id}: motorcycle_box values must be normalized to [0,1]"
                    )
                if box[2] <= 0 or box[3] <= 0:
                    self.error(
                        f"{mc_id}: motorcycle_box width/height must be > 0"
                    )

            viewpoint = motorcycle.get("viewpoint")
            if viewpoint not in VALID_VIEWPOINT:
                self.error(
                    f"{mc_id}: invalid viewpoint={viewpoint!r}"
                )

            apparent_scale = motorcycle.get("apparent_scale")
            if apparent_scale not in VALID_APPARENT_SCALE:
                self.error(
                    f"{mc_id}: invalid apparent_scale={apparent_scale!r}"
                )

            area_ratio = motorcycle.get("bbox_area_ratio")
            if not isinstance(area_ratio, (int, float)):
                self.error(
                    f"{mc_id}: bbox_area_ratio must be numeric"
                )
            elif not (0.0 < float(area_ratio) <= 1.0):
                self.error(
                    f"{mc_id}: bbox_area_ratio={area_ratio} outside (0,1]"
                )

            occlusion = motorcycle.get("occlusion")
            if occlusion not in VALID_OCCLUSION:
                self.error(
                    f"{mc_id}: invalid occlusion={occlusion!r}"
                )

            if motorcycle.get("association_status") not in VALID_ASSOCIATION_STATUS:
                self.error(
                    f"{mc_id}: association_status must be one of "
                    f"{sorted(VALID_ASSOCIATION_STATUS)}, "
                    f"found {motorcycle.get('association_status')!r}"
                )

            if motorcycle.get("rider_classification_status") != "verified":
                self.error(
                    f"{mc_id}: rider_classification_status must be 'verified', "
                    f"found {motorcycle.get('rider_classification_status')!r}"
                )

            riders = motorcycle.get("riders")
            unresolved = motorcycle.get("unresolved_riders")

            if not isinstance(riders, list):
                self.error(
                    f"{mc_id}: riders must be a list"
                )
                riders = []

            if not isinstance(unresolved, list):
                self.error(
                    f"{mc_id}: unresolved_riders must be a list"
                )
                unresolved = []

            if unresolved:
                self.error(
                    f"{mc_id}: unresolved_riders is not empty ({len(unresolved)} item(s))"
                )

            rider_ids_here: set[str] = set()
            for rider in riders:
                rider_id = rider.get("rider_id")

                if not rider_id:
                    self.error(
                        f"{mc_id}: rider has no rider_id"
                    )
                    continue

                if rider_id in rider_ids_here:
                    self.error(
                        f"{mc_id}: duplicate rider_id inside motorcycle: {rider_id}"
                    )
                rider_ids_here.add(rider_id)

                riders_by_image[image_id].append(
                    {
                        "motorcycle_id": mc_id,
                        **rider,
                    }
                )

                self.validate_rider(
                    motorcycle,
                    rider,
                    final_class_counts,
                )

            rider_count = motorcycle.get("rider_count")
            rider_count_status = motorcycle.get("rider_count_status")
            overloaded = motorcycle.get("overloaded")

            if rider_count_status != "complete":
                self.error(
                    f"{mc_id}: rider_count_status must be 'complete', "
                    f"found {rider_count_status!r}"
                )

            if not isinstance(rider_count, int) or rider_count < 0:
                self.error(
                    f"{mc_id}: rider_count must be a non-negative integer"
                )
            else:
                if rider_count != len(riders):
                    self.error(
                        f"{mc_id}: rider_count={rider_count} but riders has {len(riders)}"
                    )

                expected_overloaded = rider_count > 2
                if overloaded != expected_overloaded:
                    self.error(
                        f"{mc_id}: overloaded={overloaded!r} but rider_count={rider_count}; "
                        f"expected {expected_overloaded!r}"
                    )

                if expected_overloaded:
                    overloaded_count += 1

        # A rider must belong to exactly one motorcycle.
        for image_id, riders in riders_by_image.items():
            occurrence_count = Counter(
                rider.get("rider_id")
                for rider in riders
            )
            for rider_id, count in occurrence_count.items():
                if rider_id is None:
                    continue
                if count != 1:
                    self.error(
                        f"{image_id}: rider {rider_id} is associated with {count} motorcycles; "
                        "expected exactly one"
                    )

        self.stats["motorcycle_records"] = len(motorcycle_ids)
        self.stats["riders"] = sum(
            len(mc.get("riders", []))
            for mc in self.motorcycles
            if isinstance(mc.get("riders"), list)
        )
        self.stats["overloaded_motorcycles"] = overloaded_count
        self.stats["final_rider_class_metadata_counts"] = dict(final_class_counts)

    def validate_rider(
        self,
        motorcycle: dict[str, Any],
        rider: dict[str, Any],
        final_class_counts: Counter,
    ) -> None:
        mc_id = motorcycle.get("motorcycle_id")
        rider_id = rider.get("rider_id")

        role = rider.get("role")
        if role not in VALID_RIDER_ROLE:
            self.error(
                f"{mc_id}/{rider_id}: invalid role={role!r}"
            )

        helmet = rider.get("helmet")
        if helmet not in VALID_RIDER_HELMET:
            self.error(
                f"{mc_id}/{rider_id}: invalid helmet={helmet!r}"
            )

        helmet_valid = rider.get("helmet_annotation_valid")
        if not isinstance(helmet_valid, bool):
            self.error(
                f"{mc_id}/{rider_id}: helmet_annotation_valid must be boolean"
            )
        elif helmet_valid != (helmet in {True, False}):
            self.error(
                f"{mc_id}/{rider_id}: helmet_annotation_valid={helmet_valid!r} "
                f"inconsistent with helmet={helmet!r}"
            )

        class_name = rider.get("class_name")
        if class_name not in FINAL_CLASSES and class_name != "ambiguous_rider":
            self.error(
                f"{mc_id}/{rider_id}: invalid class_name={class_name!r}"
            )

        resolved_name = rider.get("resolved_class_name")
        resolved_id = rider.get("resolved_class_id")

        # For ordinary final-class riders, class_name itself is already the
        # resolved public class. The resolved_* fields are primarily needed
        # to record the outcome of an internal ambiguous_rider review.
        effective_class_name = None

        if class_name in FINAL_CLASSES:
            effective_class_name = class_name

            if resolved_name is not None:
                if resolved_name != class_name:
                    self.error(
                        f"{mc_id}/{rider_id}: class_name={class_name!r} but "
                        f"resolved_class_name={resolved_name!r}"
                    )

                expected_id = EXPECTED_CLASS_IDS[resolved_name]
                if resolved_id != expected_id:
                    self.error(
                        f"{mc_id}/{rider_id}: resolved_class_id={resolved_id!r} "
                        f"does not match {resolved_name}={expected_id}"
                    )

        elif class_name == "ambiguous_rider":
            # An ambiguous rider must either have been resolved to one of the
            # four final rider classes or remain genuinely metadata-only.
            if resolved_name is None:
                if resolved_id is not None:
                    self.error(
                        f"{mc_id}/{rider_id}: resolved_class_id must be null when "
                        "resolved_class_name is null"
                    )
            elif resolved_name not in FINAL_CLASSES:
                self.error(
                    f"{mc_id}/{rider_id}: resolved_class_name must be one of the "
                    f"final classes or null; found {resolved_name!r}"
                )
            else:
                expected_id = EXPECTED_CLASS_IDS[resolved_name]
                if resolved_id != expected_id:
                    self.error(
                        f"{mc_id}/{rider_id}: resolved_class_id={resolved_id!r} "
                        f"does not match {resolved_name}={expected_id}"
                    )

                effective_class_name = resolved_name

        if effective_class_name is not None:
            if role in {"driver", "passenger"} and helmet in {True, False}:
                if role == "driver":
                    expected_name = (
                        "driver_helmet"
                        if helmet is True
                        else "driver_no_helmet"
                    )
                else:
                    expected_name = (
                        "passenger_helmet"
                        if helmet is True
                        else "passenger_no_helmet"
                    )

                if effective_class_name != expected_name:
                    self.error(
                        f"{mc_id}/{rider_id}: role/helmet implies "
                        f"{expected_name}, but effective class={effective_class_name!r}"
                    )

            final_class_counts[effective_class_name] += 1

        if rider.get("association_status") not in VALID_ASSOCIATION_STATUS:
            self.error(
                f"{mc_id}/{rider_id}: association_status must be one of "
                f"{sorted(VALID_ASSOCIATION_STATUS)}; "
                f"found {rider.get('association_status')!r}"
            )

        if rider.get("classification_status") != "verified":
            self.error(
                f"{mc_id}/{rider_id}: classification_status must be 'verified'"
            )

        box = self.as_box4(rider.get("box"))
        if box is None:
            self.error(
                f"{mc_id}/{rider_id}: box must be [x,y,w,h]"
            )
        else:
            if not all(0.0 <= value <= 1.0 for value in box):
                self.error(
                    f"{mc_id}/{rider_id}: box values must be normalized [0,1]"
                )
            if box[2] <= 0 or box[3] <= 0:
                self.error(
                    f"{mc_id}/{rider_id}: box width/height must be > 0"
                )

            motorcycle_box = self.as_box4(
                motorcycle.get("motorcycle_box")
            )
            if motorcycle_box is not None:
                if not self.center_inside(
                    box,
                    motorcycle_box,
                ):
                    self.error(
                        f"{mc_id}/{rider_id}: rider center is outside associated "
                        "motorcycle box"
                    )

    # --------------------------------------------------------
    # Cross-check JSONL against YOLO
    # --------------------------------------------------------

    def validate_label_metadata_consistency(self) -> None:
        # ----------------------------------------------------
        # Motorcycle boxes: exact count + close geometric match.
        # ----------------------------------------------------
        for image_id in self.image_files:
            records = self.label_records.get(image_id, [])
            motorcycles = self.motorcycles_by_image.get(
                image_id,
                [],
            )

            mc_labels = [
                record
                for record in records
                if record.get("class_id") == EXPECTED_CLASS_IDS["motorcycle"]
            ]

            if len(mc_labels) != len(motorcycles):
                self.error(
                    f"{image_id}: YOLO has {len(mc_labels)} motorcycle boxes but "
                    f"motorcycles.jsonl has {len(motorcycles)} motorcycle records"
                )

            for motorcycle in motorcycles:
                metadata_box = self.as_box4(
                    motorcycle.get("motorcycle_box")
                )

                if metadata_box is None:
                    continue

                best_iou = 0.0

                for label in mc_labels:
                    label_box = [
                        label["x"],
                        label["y"],
                        label["w"],
                        label["h"],
                    ]

                    best_iou = max(
                        best_iou,
                        self.yolo_iou(
                            metadata_box,
                            label_box,
                        ),
                    )

                if best_iou < 0.95:
                    self.error(
                        f"{motorcycle.get('motorcycle_id')}: motorcycle_box does not "
                        f"match a class-0 YOLO box closely enough (best IoU={best_iou:.3f})"
                    )

        # ----------------------------------------------------
        # Rider classes: every final YOLO rider label must have a
        # reviewed metadata counterpart. Metadata-only riders are allowed
        # when role and/or helmet genuinely remained unknown.
        # ----------------------------------------------------
        metadata_final_counts = Counter()

        for motorcycle in self.motorcycles:
            for rider in motorcycle.get("riders", []):
                class_name = rider.get("class_name")
                resolved = rider.get("resolved_class_name")

                if class_name in {
                    "driver_helmet",
                    "driver_no_helmet",
                    "passenger_helmet",
                    "passenger_no_helmet",
                }:
                    metadata_final_counts[class_name] += 1
                elif resolved in {
                    "driver_helmet",
                    "driver_no_helmet",
                    "passenger_helmet",
                    "passenger_no_helmet",
                }:
                    metadata_final_counts[resolved] += 1

        label_final_counts = Counter()

        for records in self.label_records.values():
            for record in records:
                class_id = record.get("class_id")

                if class_id in {
                    EXPECTED_CLASS_IDS["driver_helmet"],
                    EXPECTED_CLASS_IDS["driver_no_helmet"],
                    EXPECTED_CLASS_IDS["passenger_helmet"],
                    EXPECTED_CLASS_IDS["passenger_no_helmet"],
                }:
                    name = FINAL_CLASSES[class_id]
                    label_final_counts[name] += 1

        # Every final rider label needs metadata. The reverse difference is
        # allowed because a genuinely unknown rider is metadata-only.
        excess_labels = label_final_counts - metadata_final_counts

        if excess_labels:
            self.error(
                "YOLO rider labels exceed resolved rider metadata counts: "
                f"{dict(excess_labels)}"
            )

        metadata_without_labels = (
            metadata_final_counts
            - label_final_counts
        )

        if metadata_without_labels:
            self.warning(
                "Some reviewed riders are metadata-only rather than YOLO "
                "final classes: "
                f"{dict(metadata_without_labels)}"
            )

        self.stats["label_rider_class_counts"] = dict(
            label_final_counts
        )
    @staticmethod
    def yolo_iou(
        a: list[float],
        b: list[float],
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax1 = ax - aw / 2.0
        ay1 = ay - ah / 2.0
        ax2 = ax + aw / 2.0
        ay2 = ay + ah / 2.0

        bx1 = bx - bw / 2.0
        by1 = by - bh / 2.0
        bx2 = bx + bw / 2.0
        by2 = by + bh / 2.0

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    # --------------------------------------------------------
    # Dataset-level checks
    # --------------------------------------------------------

    def validate_splits(self) -> None:
        split_counts = Counter()

        for image_id, path in self.image_files.items():
            parts = path.parts
            split = None
            for candidate in ("train", "valid", "val", "test"):
                if candidate in parts:
                    split = candidate
                    break
            if split is None:
                self.warning(
                    f"{image_id}: could not determine train/valid/test split from path {path}"
                )
            else:
                split_counts[split] += 1

        # Source-level split is not reconstructed here; Roboflow is responsible
        # for assigning complete source videos to splits. For the 13-image
        # end-to-end smoke test, all images may legitimately still be under
        # train/. The stricter all-splits requirement is available through the
        # --require-all-splits flag.
        if self.args.require_all_splits and split_counts:
            if "train" not in split_counts:
                self.error("No train split detected.")
            if not ("valid" in split_counts or "val" in split_counts):
                self.error("No validation split detected.")
            if "test" not in split_counts:
                self.error("No test split detected.")

        self.stats["split_counts"] = dict(split_counts)

    def validate_source_metadata(self) -> None:
        sources: dict[str, set[str]] = defaultdict(set)
        for image_id, row in self.image_by_id.items():
            source_id = row.get("source_id")
            if source_id is not None:
                sources[str(source_id)].add(image_id)

        self.stats["unique_sources"] = len(sources)

        if len(sources) == 1 and self.stats.get("image_files", 0) > 0:
            self.warning(
                "All images currently come from one source_id. "
                "This is acceptable for the 13-frame pipeline smoke test, "
                "but not desirable for the eventual research dataset."
            )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    def run(self) -> bool:
        self.discover_files()
        self.load_metadata()
        self.load_yaml()
        self.validate_file_parity()
        self.validate_labels()
        self.validate_image_metadata()
        self.validate_motorcycle_metadata()
        self.validate_label_metadata_consistency()
        self.validate_splits()
        self.validate_source_metadata()

        report = {
            "status": "PASS" if not self.errors else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.stats,
        }

        report_path = self.args.report
        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        self.print_summary(report_path)

        return not self.errors

    def print_summary(self, report_path: Path) -> None:
        print()
        print("=" * 72)
        print("FINAL DATASET VALIDATION")
        print("=" * 72)

        print(f"Status: {'PASS' if not self.errors else 'FAIL'}")
        print(f"Images: {self.stats.get('image_files', 0)}")
        print(f"YOLO boxes: {self.stats.get('total_yolo_boxes', 0)}")
        print(
            "Motorcycle records: "
            f"{self.stats.get('motorcycle_records', 0)}"
        )
        print(
            "Rider records: "
            f"{self.stats.get('riders', 0)}"
        )

        if self.stats.get("split_counts"):
            print(
                "Splits: "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(
                        self.stats["split_counts"].items()
                    )
                )
            )

        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Report: {report_path}")

        if self.errors:
            print()
            print("ERRORS")
            print("------")
            for index, error in enumerate(self.errors, start=1):
                print(f"{index}. {error}")

        if self.warnings:
            print()
            print("WARNINGS")
            print("--------")
            for index, warning in enumerate(self.warnings, start=1):
                print(f"{index}. {warning}")

        print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the completed motorcycle benchmark dataset."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
    )

    parser.add_argument(
        "--images-jsonl",
        type=Path,
        default=Path("metadata/images.jsonl"),
    )

    parser.add_argument(
        "--motorcycles-jsonl",
        type=Path,
        default=Path("metadata/motorcycles.jsonl"),
    )

    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("dataset/data.yaml"),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/validation_report.json"),
    )

    parser.add_argument(
        "--require-all-splits",
        action="store_true",
        help="Require train + valid/val + test directories. Use for the full research dataset; omit for the 13-image smoke test.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    required_files = [
        args.images_jsonl,
        args.motorcycles_jsonl,
        args.yaml,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        print("ERROR: Missing required file(s):")
        for path in missing:
            print(f"  {path}")
        return 1

    validator = Validator(args)
    passed = validator.run()

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
