#!/usr/bin/env python3
"""
Pass B2 — Ambiguous Rider Review UI
Resolves "driver/passenger" (Role) and "helmet/no helmet" for ambiguous riders.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw, ImageTk

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    raise

# ============================================================
# FINAL PUBLIC CLASSES & CONFIG
# ============================================================

FINAL_CLASSES = {
    "motorcycle": 0,
    "driver_helmet": 1,
    "driver_no_helmet": 2,
    "passenger_helmet": 3,
    "passenger_no_helmet": 4,
}

FINAL_CLASS_FROM_STATE = {
    ("driver", "helmet"): "driver_helmet",
    ("driver", "no_helmet"): "driver_no_helmet",
    ("passenger", "helmet"): "passenger_helmet",
    ("passenger", "no_helmet"): "passenger_no_helmet",
}

ROLE_VALUES = ("driver", "passenger", "unknown")
HELMET_VALUES = ("helmet", "no_helmet", "unknown")


# ============================================================
# FILE HELPERS
# ============================================================

def canonicalize_image_id(stem: str) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", stem, maxsplit=1, flags=re.IGNORECASE)[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys()) if rows else [
        "image_id", "rider_id", "motorcycle_id", "class_name", "role", "helmet"
    ]
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    tmp.replace(path)


def load_class_map(path: Path) -> dict[str, int]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, list):
        return {str(name): idx for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {str(name): int(idx) for idx, name in names.items()}
    raise ValueError(f"Could not parse 'names' from {path}")


def find_dataset_files(dataset_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    image_files: dict[str, Path] = {}
    label_files: dict[str, Path] = {}

    for path in dataset_root.glob("**/images/*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_files.setdefault(canonicalize_image_id(path.stem), path)

    for path in dataset_root.glob("**/labels/*.txt"):
        label_files.setdefault(canonicalize_image_id(path.stem), path)

    return image_files, label_files


# ============================================================
# YOLO GEOMETRY & ANNOTATIONS
# ============================================================

def yolo_to_xyxy(box: list[float], image_width: float, image_height: float) -> list[float]:
    x_center, y_center, width, height = box
    return [
        (x_center - width / 2.0) * image_width,
        (y_center - height / 2.0) * image_height,
        (x_center + width / 2.0) * image_width,
        (y_center + height / 2.0) * image_height,
    ]


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return (intersection / union) if union > 0 else 0.0


def update_yolo_label(
    label_path: Path,
    rider_box_norm: list[float],
    ambiguous_class_id: int,
    replacement_class_id: int | None,
    image_size: tuple[int, int],
) -> None:
    image_width, image_height = image_size
    rider_box_xyxy = yolo_to_xyxy(rider_box_norm, image_width, image_height)

    lines = label_path.read_text(encoding="utf-8").splitlines()
    candidates: list[tuple[int, list[float]]] = []

    for line_index, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            coords = [float(val) for val in parts[1:]]
        except ValueError:
            continue

        if class_id == ambiguous_class_id:
            candidates.append((line_index, yolo_to_xyxy(coords, image_width, image_height)))

    if not candidates:
        raise ValueError(f"No ambiguous_rider (id {ambiguous_class_id}) annotations found in {label_path.name}.")

    best_line_index, best_box = max(candidates, key=lambda item: iou(rider_box_xyxy, item[1]))
    best_iou = iou(rider_box_xyxy, best_box)

    if best_iou < 0.25:
        raise ValueError(f"IoU match failed ({best_iou:.3f} < 0.25) between metadata box and YOLO label.")

    if replacement_class_id is None:
        del lines[best_line_index]
    else:
        parts = lines[best_line_index].split()
        parts[0] = str(replacement_class_id)
        lines[best_line_index] = " ".join(parts)

    tmp = label_path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(label_path)


# ============================================================
# MAIN UI APPLICATION
# ============================================================

class RiderReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        images_jsonl: Path,
        motorcycles_jsonl: Path,
        review_csv: Path,
        dataset_root: Path,
        yaml_path: Path,
    ) -> None:
        self.root = root
        self.root.title("Pass B2 — Ambiguous Rider Review")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 700)

        self.images_jsonl = images_jsonl
        self.motorcycles_jsonl = motorcycles_jsonl
        self.review_csv = review_csv
        self.dataset_root = dataset_root
        self.yaml_path = yaml_path

        self.images = load_jsonl(images_jsonl)
        self.motorcycles = load_jsonl(motorcycles_jsonl)
        self.review_rows = read_review_csv(review_csv)
        self.image_files, self.label_files = find_dataset_files(dataset_root)
        self.class_map = load_class_map(yaml_path)

        if "ambiguous_rider" not in self.class_map:
            raise ValueError("The dataset YAML does not contain 'ambiguous_rider'.")

        missing_final = [name for name in FINAL_CLASSES if name not in self.class_map]
        if missing_final:
            raise ValueError("YAML missing required final classes:\n" + "\n".join(f" - {n}" for n in missing_final))

        self.ambiguous_class_id = self.class_map["ambiguous_rider"]

        self.index = 0
        self.photo: ImageTk.PhotoImage | None = None

        self.role_var = tk.StringVar(value="unknown")
        self.helmet_var = tk.StringVar(value="unknown")
        self.counter_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_ui()
        self.root.update_idletasks()
        self._refresh()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Image view
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.image_label = ttk.Label(left, anchor="center")
        self.image_label.pack(fill="both", expand=True)

        # Control Panel
        right = ttk.Frame(main, width=380)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        ttk.Label(right, text="Pass B2 — Rider Review", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(right, textvariable=self.counter_var, font=("TkDefaultFont", 11)).pack(anchor="w", pady=(0, 10))

        # Details Box (Image Name, Target Rider, Motorcycle ID)
        self.info = tk.Text(right, height=7, width=44, wrap="word", state="disabled", font=("TkDefaultFont", 10))
        self.info.pack(fill="x", pady=(0, 12))

        # Classification dropdowns
        classification = ttk.LabelFrame(right, text="Classification", padding=10)
        classification.pack(fill="x", pady=(0, 10))

        ttk.Label(classification, text="Role:").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Combobox(
            classification, textvariable=self.role_var, values=ROLE_VALUES, state="readonly", width=18
        ).grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=6)

        ttk.Label(classification, text="Helmet:").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(
            classification, textvariable=self.helmet_var, values=HELMET_VALUES, state="readonly", width=18
        ).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=6)

        classification.columnconfigure(1, weight=1)

        # Quick Presets
        shortcuts = ttk.LabelFrame(right, text="Quick Presets", padding=8)
        shortcuts.pack(fill="x", pady=(0, 15))

        buttons = [
            ("Driver + Helmet (Unknown)", "driver", "unknown"),
            ("Passenger + Helmet (Unknown)", "passenger", "unknown"),
            ("Unknown + Helmet", "unknown", "helmet"),
            ("Unknown + Helmet (Unknown)", "unknown", "unknown"),
        ]
        for row, (lbl, r, h) in enumerate(buttons):
            ttk.Button(shortcuts, text=lbl, command=lambda r=r, h=h: self._set_state(r, h)).grid(
                row=row, column=0, sticky="ew", pady=2
            )
        shortcuts.columnconfigure(0, weight=1)

        # Actions
        ttk.Button(right, text="SAVE & NEXT (Enter)", command=self._save_current).pack(fill="x", pady=5)
        ttk.Button(right, text="EXIT", command=self.root.destroy).pack(fill="x", pady=(15, 5))

        ttk.Label(right, textvariable=self.status_var, wraplength=360).pack(anchor="w", pady=10)

        # Keyboard Shortcut: Enter only
        self.root.bind("<Return>", lambda _e: self._save_current())

    def _set_state(self, role: str, helmet: str) -> None:
        self.role_var.set(role)
        self.helmet_var.set(helmet)

    def _set_info(self, text: str) -> None:
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.configure(state="disabled")

    def _current_row(self) -> dict[str, str] | None:
        if 0 <= self.index < len(self.review_rows):
            return self.review_rows[self.index]
        return None

    def _find_rider(self, image_id: str, rider_id: str) -> tuple[dict[str, Any], dict[str, Any], bool] | None:
        matches = []
        for moto in self.motorcycles:
            if moto.get("image_id") != image_id:
                continue
            for rider in moto.get("riders", []):
                if rider.get("rider_id") == rider_id:
                    matches.append((moto, rider, False))
            for rider in moto.get("unresolved_riders", []):
                if rider.get("rider_id") == rider_id:
                    matches.append((moto, rider, True))

        if not matches:
            return None
        return matches[0]

    def _draw_current_image(self) -> None:
        row = self._current_row()
        if row is None:
            self.image_label.configure(image="")
            self.photo = None
            return

        image_id = row.get("image_id", "")
        rider_id = row.get("rider_id", "")

        image_path = self.image_files.get(image_id)
        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Image not found for {image_id}.")

        match = self._find_rider(image_id, rider_id)
        if match is None:
            raise ValueError(f"Rider {rider_id} not found in motorcycles.jsonl.")

        motorcycle, target_rider, is_unresolved = match

        # Sync existing state
        cur_role = target_rider.get("role", "unknown")
        raw_helmet = target_rider.get("helmet", "unknown")
        cur_helmet = "helmet" if raw_helmet is True else ("no_helmet" if raw_helmet is False else str(raw_helmet))
        self._set_state(cur_role, cur_helmet)

        with Image.open(image_path) as src:
            orig_image = src.convert("RGB")

        orig_w, orig_h = orig_image.size

        # Fit image to viewport
        max_w = max(600, self.root.winfo_width() - 400)
        max_h = max(500, self.root.winfo_height() - 40)
        scale = min(max_w / orig_w, max_h / orig_h, 1.0)
        disp_w = max(1, int(orig_w * scale))
        disp_h = max(1, int(orig_h * scale))

        scene_img = orig_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(scene_img)

        # ONLY draw the target rider in CYAN
        r_box = target_rider.get("box")
        if isinstance(r_box, list) and len(r_box) == 4:
            x1, y1, x2, y2 = yolo_to_xyxy([float(v) for v in r_box], disp_w, disp_h)

            # Cyan box
            draw.rectangle([x1, y1, x2, y2], outline="cyan", width=4)

            # Solid badge: CYAN: TARGET RIDER (rider_id)
            label_text = f" CYAN: TARGET RIDER ({rider_id}) "
            badge_y = max(0.0, y1 - 18)
            bbox = draw.textbbox((x1, badge_y), label_text)
            draw.rectangle(bbox, fill="cyan")
            draw.text((x1, badge_y), label_text, fill="black")

        self.photo = ImageTk.PhotoImage(scene_img)
        self.image_label.configure(image=self.photo)

        # Display details on right panel
        self._set_info(
            f"Image: {image_id}\n"
            f"Target Rider: {rider_id}\n"
            f"Motorcycle ID: {motorcycle.get('motorcycle_id')}\n"
            f"Original Class: {target_rider.get('class_name', 'N/A')}"
        )

        if is_unresolved:
            self.status_var.set("B1 association unresolved. Run review_associations.py first.")
        else:
            self.status_var.set("Select Role and Helmet, then press Enter.")

    def _refresh(self) -> None:
        if not self.review_rows:
            self.counter_var.set("Review queue is empty.")
            self._set_info("All ambiguous riders have been reviewed.")
            self.status_var.set("Pass B2 complete.")
            self.image_label.configure(image="")
            self.photo = None
            return

        self.index = max(0, min(self.index, len(self.review_rows) - 1))
        self.counter_var.set(f"Review {self.index + 1} of {len(self.review_rows)}")

        try:
            self._draw_current_image()
        except Exception as exc:
            messagebox.showerror("Display Error", str(exc))
            self.status_var.set(str(exc))

    def _update_statuses(self) -> None:
        riders_by_image: dict[str, list[dict[str, Any]]] = {}

        for moto in self.motorcycles:
            img_id = moto.get("image_id", "")
            riders = moto.get("riders", [])
            riders_by_image.setdefault(img_id, []).extend(riders)

            moto["rider_classification_status"] = (
                "review_required"
                if any(r.get("classification_status") == "review_required" for r in riders)
                else "verified"
            )

        for img in self.images:
            img_id = img.get("image_id", "")
            riders = riders_by_image.get(img_id, [])
            img["rider_classification_status"] = (
                "review_required"
                if any(r.get("classification_status") == "review_required" for r in riders)
                else "verified"
            )

    def _save_current(self) -> None:
        row = self._current_row()
        if row is None:
            return

        image_id = row.get("image_id", "")
        rider_id = row.get("rider_id", "")
        role = self.role_var.get()
        helmet = self.helmet_var.get()

        if role not in ROLE_VALUES or helmet not in HELMET_VALUES:
            messagebox.showerror("Invalid Input", "Please select valid role and helmet states.")
            return

        match = self._find_rider(image_id, rider_id)
        if match is None:
            messagebox.showerror("Error", f"Rider {rider_id} was not found in metadata.")
            return

        motorcycle, rider, is_unresolved = match
        if is_unresolved:
            messagebox.showwarning("Incomplete B1", "Rider association is still unresolved. Complete B1 first.")
            return

        rider_box = rider.get("box")
        if not (isinstance(rider_box, list) and len(rider_box) == 4):
            messagebox.showerror("Error", f"Rider {rider_id} has invalid bounding box geometry.")
            return

        label_path = self.label_files.get(image_id)
        image_path = self.image_files.get(image_id)
        if not label_path or not label_path.exists():
            messagebox.showerror("Error", f"No YOLO label file found for {image_id}.")
            return
        if not image_path or not image_path.exists():
            messagebox.showerror("Error", f"No image file found for {image_id}.")
            return

        resolved_class = FINAL_CLASS_FROM_STATE.get((role, helmet))
        replacement_class_id = self.class_map[resolved_class] if resolved_class else None

        # Update in-memory metadata
        rider["role"] = role
        rider["helmet_annotation_valid"] = (helmet != "unknown")
        rider["helmet"] = True if helmet == "helmet" else (False if helmet == "no_helmet" else "unknown")
        rider["resolved_class_name"] = resolved_class
        rider["resolved_class_id"] = replacement_class_id
        rider["classification_status"] = "verified"
        rider["association_status"] = "verified"

        try:
            with Image.open(image_path) as img:
                image_size = img.size

            backup_once(self.motorcycles_jsonl)
            backup_once(self.images_jsonl)
            backup_once(self.review_csv)
            backup_once(label_path)

            update_yolo_label(
                label_path=label_path,
                rider_box_norm=[float(v) for v in rider_box],
                ambiguous_class_id=self.ambiguous_class_id,
                replacement_class_id=replacement_class_id,
                image_size=image_size,
            )
        except Exception as exc:
            messagebox.showerror("Save Failed", f"Failed updating YOLO annotations:\n{exc}")
            return

        # Advance queue and commit metadata files
        del self.review_rows[self.index]
        self._update_statuses()

        atomic_write_jsonl(self.motorcycles_jsonl, self.motorcycles)
        atomic_write_jsonl(self.images_jsonl, self.images)
        write_review_csv(self.review_csv, self.review_rows)

        self._refresh()

        if not self.review_rows:
            messagebox.showinfo("Pass B2 Complete", "All ambiguous riders have been reviewed!")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass B2 ambiguous rider review UI")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--images-jsonl", type=Path, default=Path("metadata/images.jsonl"))
    parser.add_argument("--motorcycles-jsonl", type=Path, default=Path("metadata/motorcycles.jsonl"))
    parser.add_argument("--review-csv", type=Path, default=Path("reports/rider_review_needed.csv"))
    parser.add_argument("--yaml", type=Path, default=Path("dataset/data.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = [args.dataset_root, args.images_jsonl, args.motorcycles_jsonl, args.review_csv, args.yaml]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required path(s):", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 1

    root = tk.Tk()
    try:
        RiderReviewApp(
            root=root,
            images_jsonl=args.images_jsonl,
            motorcycles_jsonl=args.motorcycles_jsonl,
            review_csv=args.review_csv,
            dataset_root=args.dataset_root,
            yaml_path=args.yaml,
        )
    except Exception as exc:
        root.destroy()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())