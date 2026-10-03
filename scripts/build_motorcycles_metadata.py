import os
import re
import json
import csv
import sys
import yaml
from pathlib import Path


# ============================================================
# Configuration / Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
METADATA_DIR = BASE_DIR / "metadata"
REPORTS_DIR = BASE_DIR / "reports"

YAML_PATH = DATASET_DIR / "data.yaml"

MOTORCYCLES_JSONL = METADATA_DIR / "motorcycles.jsonl"
IMAGES_JSONL = METADATA_DIR / "images.jsonl"
ASSOCIATION_REVIEW_CSV = (
    REPORTS_DIR / "association_review_needed.csv"
)

RIDER_REVIEW_CSV = (
    REPORTS_DIR / "rider_review_needed.csv"
)

METADATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Basic validation
# ============================================================

if not YAML_PATH.exists():
    print(f"[-] Missing {YAML_PATH}")
    print("    Did you unzip the Roboflow export into dataset/?")
    sys.exit(1)


# ============================================================
# Load class mapping from data.yaml
# ============================================================

with open(YAML_PATH, "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)

names = yaml_data.get("names")

if isinstance(names, list):
    class_mapping = {
        idx: str(name).strip().lower()
        for idx, name in enumerate(names)
    }

elif isinstance(names, dict):
    class_mapping = {
        int(idx): str(name).strip().lower()
        for idx, name in names.items()
    }

else:
    raise ValueError("Invalid 'names' structure in data.yaml")

print(f"[+] Loaded class mapping: {class_mapping}")


EXPECTED_CLASSES = {
    "motorcycle",
    "driver_helmet",
    "driver_no_helmet",
    "passenger_helmet",
    "passenger_no_helmet",
    "ambiguous_rider",
}

unknown_classes = set(class_mapping.values()) - EXPECTED_CLASSES

if unknown_classes:
    raise ValueError(
        f"Unexpected classes found in data.yaml: "
        f"{sorted(unknown_classes)}"
    )


# ============================================================
# Helper functions
# ============================================================

def compute_scale(area_ratio: float) -> str:
    """
    Rough apparent-size category based on motorcycle
    bounding-box area relative to the image.

    These thresholds are provisional and should remain
    configurable until calibrated on the final dataset.
    """
    if area_ratio >= 0.15:
        return "large"

    if area_ratio >= 0.04:
        return "medium"

    if area_ratio >= 0.008:
        return "small"

    return "very_small"

def canonicalize_image_id(stem: str) -> str:
    """
    Convert a Roboflow-exported image stem back to the
    canonical dataset image ID.

    Example:
        IMG_000004_jpg.rf.ABC123
        -> IMG_000004

    If the filename has no Roboflow suffix, it is returned
    unchanged.
    """
    return re.split(
        r"_(?:jpg|jpeg|png)\.rf\.",
        stem,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]    


def is_point_in_box(
    px: float,
    py: float,
    box: list[float],
) -> bool:
    """
    Returns True when the point lies inside the YOLO
    normalized xywh bounding box.
    """
    bx_c, by_c, bw, bh = box

    xmin = bx_c - (bw / 2.0)
    xmax = bx_c + (bw / 2.0)

    ymin = by_c - (bh / 2.0)
    ymax = by_c + (bh / 2.0)

    return (
        xmin <= px <= xmax
        and ymin <= py <= ymax
    )


def parse_yolo_line(
    line: str,
    label_file: Path,
) -> tuple[int, str, list[float]]:
    """
    Parse and validate one YOLO annotation line.
    """
    parts = line.split()

    if len(parts) != 5:
        raise ValueError(
            f"Malformed YOLO line in {label_file}: {line}"
        )

    cls_idx = int(parts[0])
    cls_name = class_mapping.get(cls_idx)

    if cls_name is None:
        raise ValueError(
            f"Unknown class ID {cls_idx} in {label_file}"
        )

    x_c, y_c, w, h = map(float, parts[1:5])

    values = [x_c, y_c, w, h]

    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Invalid normalized coordinate {value} "
                f"in {label_file}"
            )

    if w <= 0.0 or h <= 0.0:
        raise ValueError(
            f"Invalid bounding-box size in {label_file}: "
            f"{w} x {h}"
        )

    box = [
        round(x_c, 6),
        round(y_c, 6),
        round(w, 6),
        round(h, 6),
    ]

    return cls_idx, cls_name, box


# ============================================================
# Discover all labels from all dataset splits
# ============================================================

label_files = sorted(
    label_file
    for split_dir in DATASET_DIR.iterdir()
    if split_dir.is_dir()
    for labels_dir in [split_dir / "labels"]
    if labels_dir.is_dir()
    for label_file in labels_dir.glob("*.txt")
)

if not label_files:
    print(f"[-] No label files found under {DATASET_DIR}")
    sys.exit(1)

print(
    f"[+] Found {len(label_files)} label files "
    f"across all dataset splits."
)


# ============================================================
# Containers
# ============================================================

motorcycle_records = []
processed_image_ids = set()

# Association problems:
#   zero motorcycle candidates
#   multiple motorcycle candidates
association_review_records = []

# Rider semantic problems:
#   ambiguous_rider class
rider_review_records = []

image_association_status = {}
image_rider_status = {}


# ============================================================
# Process each image
# ============================================================

for label_file in label_files:

    image_id = canonicalize_image_id(label_file.stem)

    # The split directory is two levels above the label file:
    # dataset/train/labels/image.txt
    #             ^^^^^
    split_name = label_file.parent.parent.name

    processed_image_ids.add(image_id)

    image_has_association_issues = False
    image_has_rider_review_issues = False

    # --------------------------------------------------------
    # Read YOLO labels
    # --------------------------------------------------------

    with open(label_file, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    motorcycles = []
    riders = []

    for line in lines:

        cls_idx, cls_name, box_coords = parse_yolo_line(
            line,
            label_file,
        )

        x_c = box_coords[0]
        y_c = box_coords[1]
        w = box_coords[2]
        h = box_coords[3]

        if cls_name == "motorcycle":

            motorcycles.append({
                "box": box_coords,
                "w": w,
                "h": h,
            })

        elif cls_name in {
            "driver_helmet",
            "driver_no_helmet",
            "passenger_helmet",
            "passenger_no_helmet",
            "ambiguous_rider",
        }:

            if cls_name == "ambiguous_rider":

                role = "unknown"
                helmet = "unknown"
                helmet_annotation_valid = False

            else:

                role = (
                    "driver"
                    if cls_name.startswith("driver_")
                    else "passenger"
                )

                helmet = cls_name in {
                    "driver_helmet",
                    "passenger_helmet",
                }

                helmet_annotation_valid = True

            riders.append({
                "rider_id": f"R{len(riders) + 1:02d}",
                "role": role,
                "helmet": helmet,
                "helmet_annotation_valid": helmet_annotation_valid,
                "class_name": cls_name,
                "box": box_coords,
                "center": (x_c, y_c),
            })

    # --------------------------------------------------------
    # Build candidate motorcycle associations
    #
    # Exactly one candidate:
    #     safe automatic proposal
    #
    # Zero candidates:
    #     annotation/geometry warning
    #
    # Multiple candidates:
    #     ambiguous association
    # --------------------------------------------------------

    rider_candidates = {}

    for rider in riders:

        rx, ry = rider["center"]

        candidates = []

        for mc_idx, motorcycle in enumerate(
            motorcycles,
            start=1,
        ):
            if is_point_in_box(
                rx,
                ry,
                motorcycle["box"],
            ):
                candidates.append(mc_idx)

        rider_candidates[rider["rider_id"]] = candidates

        # ----------------------------------------------------
        # AMBIGUOUS RIDER
        #
        # The rider definitely exists because you manually
        # annotated it as ambiguous_rider.
        #
        # Its role and/or helmet status must be resolved later.
        # This is separate from motorcycle association.
        # ----------------------------------------------------

        if rider["class_name"] == "ambiguous_rider":

            image_has_rider_review_issues = True

            rider_review_records.append({
                "image_id": image_id,
                "split": split_name,
                "issue_type": "ambiguous_rider_classification",
                "rider_id": rider["rider_id"],
                "details": (
                    "Rider was annotated as ambiguous_rider. "
                    "Role and/or helmet status must be resolved."
                ),
            })

        # ----------------------------------------------------
        # ZERO candidates
        #
        # The rider is visible and manually annotated, but the
        # rider center is not inside any motorcycle box.
        #
        # This is an annotation/geometry warning, NOT evidence
        # that the rider does not exist.
        # ----------------------------------------------------

        if len(candidates) == 0:

            image_has_association_issues = True

            association_review_records.append({
                "image_id": image_id,
                "split": split_name,
                "issue_type": "rider_outside_all_motorcycle_boxes",
                "rider_id": rider["rider_id"],
                "details": (
                    f"{rider['class_name']} "
                    f"{rider['rider_id']} at {rider['box']} "
                    "is not inside any motorcycle box."
                ),
            })

        # ----------------------------------------------------
        # MULTIPLE candidates
        #
        # The rider may belong to more than one motorcycle.
        # Human association review is required.
        # ----------------------------------------------------

        elif len(candidates) > 1:

            image_has_association_issues = True

            candidate_motorcycle_ids = [
                f"MC_{image_id}_{mc_idx:02d}"
                for mc_idx in candidates
            ]

            association_review_records.append({
                "image_id": image_id,
                "split": split_name,
                "issue_type": "rider_matches_multiple_motorcycles",
                "rider_id": rider["rider_id"],
                "details": (
                    f"{rider['class_name']} "
                    f"{rider['rider_id']} matches "
                    f"{candidate_motorcycle_ids}."
                ),
            })
    # --------------------------------------------------------
    # Build one metadata record per motorcycle
    # --------------------------------------------------------

    for mc_idx, motorcycle in enumerate(
        motorcycles,
        start=1,
    ):

        mc_id = f"MC_{image_id}_{mc_idx:02d}"

        mc_box = motorcycle["box"]

        area_ratio = round(
            motorcycle["w"] * motorcycle["h"],
            6,
        )

        scale = compute_scale(area_ratio)

        associated_riders = []
        unresolved_riders = []

        for rider in riders:

            candidates = rider_candidates[
                rider["rider_id"]
            ]

            # ------------------------------------------------
            # Exactly one motorcycle candidate
            # ------------------------------------------------

            if candidates == [mc_idx]:

                associated_riders.append({
                    "rider_id": rider["rider_id"],
                    "role": rider["role"],
                    "helmet": rider["helmet"],
                    "helmet_annotation_valid": (
                        rider["helmet_annotation_valid"]
                    ),
                    "class_name": rider["class_name"],
                    "box": rider["box"],

                    # Relationship status
                    "association_status": "auto_candidate",

                    # Semantic rider status
                    "classification_status": (
                        "review_required"
                        if rider["class_name"] == "ambiguous_rider"
                        else "verified"
                    ),

                    # Filled later if ambiguous rider is resolved
                    "resolved_class_name": None,
                    "resolved_class_id": None,
                })

            # ------------------------------------------------
            # Rider may belong to this motorcycle OR another
            # ------------------------------------------------

            elif mc_idx in candidates and len(candidates) > 1:

                unresolved_riders.append({
                    "rider_id": rider["rider_id"],
                    "role": rider["role"],
                    "helmet": rider["helmet"],
                    "helmet_annotation_valid": (
                        rider["helmet_annotation_valid"]
                    ),
                    "class_name": rider["class_name"],
                    "box": rider["box"],
                    "candidate_motorcycles": [
                        f"MC_{image_id}_{candidate_idx:02d}"
                        for candidate_idx in candidates
                    ],
                    "classification_status": (
                        "review_required"
                        if rider["class_name"] == "ambiguous_rider"
                        else "verified"
                    ),
                })

        # ----------------------------------------------------
        # If there is an unresolved rider that could belong
        # to THIS motorcycle, its rider count cannot be trusted.
        #
        # For a rider that matched zero motorcycles, we also
        # invalidate motorcycle-level counts because the missing
        # rider could belong to any motorcycle in this image.
        # ----------------------------------------------------

        orphan_exists = any(
            len(rider_candidates[rider["rider_id"]]) == 0
            for rider in riders
        )

        multiple_candidate_exists = len(
            unresolved_riders
        ) > 0

        if multiple_candidate_exists or orphan_exists:

            # Motorcycle↔rider relationship is uncertain.
            # Rider count cannot safely be derived yet.
            association_status = "review_required"
            rider_count = None
            rider_count_status = "uncertain"
            overloaded = None

        else:

            # Every rider has exactly one motorcycle candidate.
            # An ambiguous rider is still a real rider and
            # therefore contributes to rider_count.
            association_status = "auto_candidate"
            rider_count = len(associated_riders)
            rider_count_status = "complete"
            overloaded = rider_count > 2

        record = {
            "motorcycle_id": mc_id,
            "image_id": image_id,
            "split": split_name,

            # Motorcycle instance attributes
            "motorcycle_box": mc_box,
            "viewpoint": "unassigned",
            "apparent_scale": scale,
            "bbox_area_ratio": area_ratio,
            "occlusion": "unassigned",

            # Derived motorcycle-level attributes
            "rider_count": rider_count,
            "rider_count_status": rider_count_status,
            "overloaded": overloaded,

            # Association state
            "association_status": association_status,

            # Rider semantic classification state
            "rider_classification_status": (
                "review_required"
                if any(
                    rider.get("class_name") == "ambiguous_rider"
                    for rider in associated_riders
                )
                else "verified"
            ),

            # Verified/ambiguous riders with unique associations
            "riders": associated_riders,

            # Riders requiring motorcycle-association review
            "unresolved_riders": unresolved_riders,
        }

        motorcycle_records.append(record)

    # --------------------------------------------------------
    # IMPORTANT:
    # Pass A is already manually verified in Roboflow.
    #
    # Therefore:
    #     annotation_status = verified
    #
    # Association is tracked independently.
    # --------------------------------------------------------

    image_association_status[image_id] = (
        "review_required"
        if image_has_association_issues
        else "auto_candidate"
    )

    image_rider_status[image_id] = (
        "review_required"
        if image_has_rider_review_issues
        else "verified"
    )


# ============================================================
# Write motorcycles.jsonl
# ============================================================

with open(
    MOTORCYCLES_JSONL,
    "w",
    encoding="utf-8",
) as f:

    for record in motorcycle_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            ) + "\n"
        )


# ============================================================
# Write association review report
# ============================================================

if ASSOCIATION_REVIEW_CSV.exists():
    ASSOCIATION_REVIEW_CSV.unlink()

if association_review_records:

    with open(
        ASSOCIATION_REVIEW_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "split",
                "issue_type",
                "rider_id",
                "details",
            ],
        )

        writer.writeheader()
        writer.writerows(
            association_review_records
        )


# ============================================================
# Write rider semantic review report
# ============================================================

if RIDER_REVIEW_CSV.exists():
    RIDER_REVIEW_CSV.unlink()

if rider_review_records:

    with open(
        RIDER_REVIEW_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "split",
                "issue_type",
                "rider_id",
                "details",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rider_review_records
        )


# ============================================================
# Update images.jsonl
#
# Pass A is already manually verified in Roboflow.
#
# Therefore:
#
# annotation_status = verified
#
# Association and ambiguous-rider classification are tracked
# independently.
# ============================================================

if IMAGES_JSONL.exists():

    updated_images = []

    with open(
        IMAGES_JSONL,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            data = json.loads(line)

            img_id = data.get("image_id")

            if img_id in processed_image_ids:

                data["annotation_status"] = "verified"

                data["association_status"] = (
                    image_association_status.get(
                        img_id,
                        "not_processed",
                    )
                )

                data["rider_classification_status"] = (
                    image_rider_status.get(
                        img_id,
                        "not_processed",
                    )
                )

            updated_images.append(data)

    with open(
        IMAGES_JSONL,
        "w",
        encoding="utf-8",
    ) as f:

        for data in updated_images:

            f.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                ) + "\n"
            )

# ============================================================
# Final summary
# ============================================================

review_image_count = sum(
    1
    for status in image_association_status.values()
    if status == "review_required"
)

auto_candidate_image_count = sum(
    1
    for status in image_association_status.values()
    if status == "auto_candidate"
)

print("[✓] Association metadata generation complete.")
print(f"    - Processed images: {len(processed_image_ids)}")
print(f"    - Motorcycles cataloged: {len(motorcycle_records)}")
print(
    f"    - Images with clean associations: "
    f"{auto_candidate_image_count}"
)
print(
    f"    - Images requiring association review: "
    f"{review_image_count}"
)
print(
    f"    - Association review issues: "
    f"{len(association_review_records)}"
)
print(
    f"    - Rider classification issues: "
    f"{len(rider_review_records)}"
)
print(f"    - Output saved to: {MOTORCYCLES_JSONL}")

if association_review_records:
    print(
        f"    - Association review CSV: "
        f"{ASSOCIATION_REVIEW_CSV}"
    )
else:
    print("    - No motorcycle-association issues detected.")

if rider_review_records:
    print(
        f"    - Rider review CSV: "
        f"{RIDER_REVIEW_CSV}"
    )
else:
    print("    - No ambiguous-rider cases detected.")

if IMAGES_JSONL.exists():
    print(
        f"    - Image metadata updated: "
        f"{IMAGES_JSONL}"
    )
else:
    print(
        "    - WARNING: images.jsonl was not found, "
        "so image metadata status was not updated."
    )