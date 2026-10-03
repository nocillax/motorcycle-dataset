#!/usr/bin/env python3
import json
import re
import shutil
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw, ImageTk

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
METADATA_DIR = BASE_DIR / "metadata"
IMAGES_JSONL = METADATA_DIR / "images.jsonl"
MOTORCYCLES_JSONL = METADATA_DIR / "motorcycles.jsonl"

# Metadata vocabularies
IMAGE_FIELDS = {
    "illumination": ["day", "dusk", "night_artificial", "dark_unlit"],
    "weather": ["clear", "rain", "fog"],
    "road_condition": ["dry", "wet"],
    "glare": ["none", "mild", "moderate", "severe"],
    "visibility": ["clear", "moderate", "poor"],
    "motion_blur": ["none", "mild", "moderate", "severe"],
    "traffic_density": ["low", "medium", "high"],
    "traffic_context": ["free_flow", "slow_traffic", "traffic_jam", "stopped_intersection", "uncertain"],
    "road_type": ["urban_arterial", "intersection", "alley", "highway_main_road", "residential", "other"],
}

MOTORCYCLE_FIELDS = {
    "viewpoint": ["front", "rear", "side_left", "side_right", "oblique", "uncertain"],
    "occlusion": ["none", "partial", "heavy"],
}


def canonicalize_image_id(stem: str) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", stem, maxsplit=1, flags=re.IGNORECASE)[0]


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def atomic_write_jsonl(path, records):
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def backup_once(path):
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def build_image_index():
    index = {}
    if not DATASET_DIR.exists():
        return index
    for split_dir in DATASET_DIR.iterdir():
        if not split_dir.is_dir():
            continue
        images_dir = split_dir / "images"
        if not images_dir.is_dir():
            continue
        for image_path in images_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image_id = canonicalize_image_id(image_path.stem)
            index[image_id] = {"path": image_path, "split": split_dir.name}
    return index


# Load metadata
image_records = load_jsonl(IMAGES_JSONL)
motorcycle_records = load_jsonl(MOTORCYCLES_JSONL)

if not image_records:
    print("[-] images.jsonl is empty or missing.")
    sys.exit(1)

image_index = build_image_index()
motorcycles_by_image = {}

for motorcycle in motorcycle_records:
    motorcycles_by_image.setdefault(motorcycle["image_id"], []).append(motorcycle)

image_records.sort(
    key=lambda record: (
        record.get("source_id", ""),
        float(record.get("timestamp_sec", 0.0)),
    )
)


class MetadataEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Motorcycle Dataset — Pass C/D Metadata")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 750)
        self.current_index = 0
        self.current_motorcycle_index = 0
        self.image_photo = None
        self.image_vars = {}
        self.motorcycle_vars = {}
        self.clipboard_image_vars = {}
        self.clipboard_hard_negative = False
        self.build_ui()
        self.load_current_image()

    def build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.image_label = ttk.Label(left, text="Loading...", anchor="center")
        self.image_label.pack(fill="both", expand=True)

        right = ttk.Frame(main, width=420)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        self.image_info_label = ttk.Label(right, text="", wraplength=420)
        self.image_info_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(
            right,
            text="PASS C — IMAGE / SCENE METADATA",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        for field, choices in IMAGE_FIELDS.items():
            row = ttk.Frame(right)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=field, width=18).pack(side="left")
            var = tk.StringVar()
            ttk.Combobox(
                row,
                textvariable=var,
                values=choices,
                state="readonly",
                width=22,
            ).pack(side="left")
            self.image_vars[field] = var

        self.hard_negative_var = tk.BooleanVar()
        ttk.Checkbutton(
            right,
            text="Hard negative (No motorcycles present)",
            variable=self.hard_negative_var,
        ).pack(anchor="w", pady=5)

        cp_frame = ttk.Frame(right)
        cp_frame.pack(fill="x", pady=5)
        ttk.Button(cp_frame, text="Copy Metadata", command=self.copy_metadata).pack(
            side="left", expand=True, fill="x", padx=(0, 2)
        )
        ttk.Button(cp_frame, text="Paste Metadata", command=self.paste_metadata).pack(
            side="right", expand=True, fill="x", padx=(2, 0)
        )

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(
            right,
            text="PASS D — MOTORCYCLE METADATA",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        self.motorcycle_list = tk.Listbox(
            right, width=55, height=7, exportselection=False
        )
        self.motorcycle_list.pack(fill="x", pady=(5, 5))
        self.motorcycle_list.bind("<<ListboxSelect>>", self.select_motorcycle)

        for field in MOTORCYCLE_FIELDS:
            row = ttk.Frame(right)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=field, width=18).pack(side="left")
            var = tk.StringVar()
            ttk.Combobox(
                row,
                textvariable=var,
                values=MOTORCYCLE_FIELDS[field],
                state="readonly",
                width=22,
            ).pack(side="left")
            self.motorcycle_vars[field] = var

        self.scale_label = ttk.Label(right, text="")
        self.scale_label.pack(anchor="w", pady=5)

        ttk.Button(
            right,
            text="Save Current Motorcycle",
            command=self.save_current_motorcycle,
        ).pack(fill="x", pady=5)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(
            right,
            text="SAVE ALL & NEXT (Enter)",
            command=self.save_and_next,
        ).pack(fill="x", pady=5)

        ttk.Button(
            right,
            text="Previous Image",
            command=self.previous_image,
        ).pack(fill="x", pady=5)

        self.status_label = ttk.Label(right, text="", wraplength=420)
        self.status_label.pack(anchor="w", pady=10)

        self.root.bind("<Return>", lambda _e: self.save_and_next())

    def current_image(self):
        if self.current_index >= len(image_records):
            return None
        return image_records[self.current_index]

    def load_image_preview(self):
        record = self.current_image()
        if record is None:
            return

        image_id = record["image_id"]
        image_info = image_index.get(image_id)

        if image_info is None:
            self.image_label.configure(
                text=f"Could not locate image {image_id} under dataset/*/images/"
            )
            return

        image = Image.open(image_info["path"]).convert("RGB")
        max_width, max_height = 1000, 820
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        motorcycles = motorcycles_by_image.get(image_id, [])
        if motorcycles and 0 <= self.current_motorcycle_index < len(motorcycles):
            selected_mc = motorcycles[self.current_motorcycle_index]
            box = selected_mc.get("motorcycle_box") or selected_mc.get("box")

            if isinstance(box, list) and len(box) == 4:
                draw = ImageDraw.Draw(image)
                disp_w, disp_h = new_size
                x_c, y_c, w, h = [float(v) for v in box]

                xmin = int((x_c - w / 2.0) * disp_w)
                xmax = int((x_c + w / 2.0) * disp_w)
                ymin = int((y_c - h / 2.0) * disp_h)
                ymax = int((y_c + h / 2.0) * disp_h)

                draw.rectangle(
                    [xmin, ymin, xmax, ymax],
                    outline="cyan",
                    width=4,
                )

                mc_id = selected_mc.get(
                    "motorcycle_id",
                    f"MC_{self.current_motorcycle_index + 1}",
                )
                tag = f" {mc_id} "
                tag_y = max(0, ymin - 18)
                bbox = draw.textbbox((xmin, tag_y), tag)
                draw.rectangle(bbox, fill="cyan")
                draw.text((xmin, tag_y), tag, fill="black")

        self.image_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.image_photo, text="")

    def load_current_image(self):
        record = self.current_image()
        if record is None:
            return

        image_id = record["image_id"]
        motorcycles = motorcycles_by_image.get(image_id, [])

        self.image_info_label.configure(
            text=(
                f"Image {self.current_index + 1} of {len(image_records)}\n\n"
                f"ID: {image_id}\n"
                f"Source: {record.get('source_id')}\n"
                f"Timestamp: {record.get('timestamp_sec')} sec\n"
                f"Split: {record.get('split', 'unknown')}\n"
                f"Motorcycles: {len(motorcycles)}"
            )
        )

        for field in IMAGE_FIELDS:
            value = record.get(field)
            self.image_vars[field].set("" if value is None else str(value))

        self.hard_negative_var.set(bool(record.get("hard_negative", False)))

        self.motorcycle_list.delete(0, tk.END)
        for index, motorcycle in enumerate(motorcycles):
            rider_count = motorcycle.get("rider_count")
            rider_text = "?" if rider_count is None else str(rider_count)
            self.motorcycle_list.insert(
                tk.END,
                f"{index + 1:02d}. {motorcycle['motorcycle_id']} | "
                f"riders={rider_text} | view={motorcycle.get('viewpoint', '')}",
            )

        self.current_motorcycle_index = 0

        if motorcycles:
            self.motorcycle_list.selection_set(0)
            self.select_motorcycle(None)
        else:
            for var in self.motorcycle_vars.values():
                var.set("")
            self.scale_label.configure(text="No motorcycles in this image.")

        self.load_image_preview()
        self.status_label.configure(
            text="Set Pass C metadata, then set Pass D for each motorcycle."
        )

    def select_motorcycle(self, event):
        selection = self.motorcycle_list.curselection()
        if selection:
            self.current_motorcycle_index = selection[0]

        record = self.current_image()
        if record is None:
            return

        motorcycles = motorcycles_by_image.get(record["image_id"], [])
        if not motorcycles or self.current_motorcycle_index >= len(motorcycles):
            return

        motorcycle = motorcycles[self.current_motorcycle_index]

        for field in MOTORCYCLE_FIELDS:
            self.motorcycle_vars[field].set(str(motorcycle.get(field, "") or ""))

        self.scale_label.configure(
            text=(
                f"Apparent scale: {motorcycle.get('apparent_scale')}\n"
                f"BBox area ratio: {motorcycle.get('bbox_area_ratio')}"
            )
        )
        self.load_image_preview()

    def copy_metadata(self):
        for field, var in self.image_vars.items():
            self.clipboard_image_vars[field] = var.get()
        self.clipboard_hard_negative = self.hard_negative_var.get()
        self.status_label.configure(text="Copied Pass C metadata to clipboard.")

    def paste_metadata(self):
        if not self.clipboard_image_vars:
            messagebox.showinfo(
                "Empty Clipboard",
                "No metadata has been copied yet.",
            )
            return

        for field, value in self.clipboard_image_vars.items():
            self.image_vars[field].set(value)
        self.hard_negative_var.set(self.clipboard_hard_negative)
        self.status_label.configure(
            text="Pasted Pass C metadata. Review and hit Save & Next."
        )

    def save_current_motorcycle(self):
        record = self.current_image()
        if record is None:
            return

        motorcycles = motorcycles_by_image.get(record["image_id"], [])
        if not motorcycles:
            return

        motorcycle = motorcycles[self.current_motorcycle_index]

        for field in MOTORCYCLE_FIELDS:
            value = self.motorcycle_vars[field].get().strip()
            if not value:
                messagebox.showwarning("Missing value", f"Please set {field}.")
                return
            motorcycle[field] = value

        backup_once(MOTORCYCLES_JSONL)
        atomic_write_jsonl(MOTORCYCLES_JSONL, motorcycle_records)
        self.status_label.configure(text=f"Saved {motorcycle['motorcycle_id']}.")

    def save_and_next(self):
        record = self.current_image()
        if record is None:
            return

        for field in IMAGE_FIELDS:
            value = self.image_vars[field].get().strip()
            if not value:
                messagebox.showwarning(
                    "Missing image metadata",
                    f"Please set {field}.",
                )
                return
            record[field] = value

        record["hard_negative"] = self.hard_negative_var.get()
        motorcycles = motorcycles_by_image.get(record["image_id"], [])

        if motorcycles:
            motorcycle = motorcycles[self.current_motorcycle_index]

            for field in MOTORCYCLE_FIELDS:
                value = self.motorcycle_vars[field].get().strip()
                if not value:
                    messagebox.showwarning(
                        "Missing motorcycle metadata",
                        f"Please set {field} for {motorcycle['motorcycle_id']}.",
                    )
                    return
                motorcycle[field] = value

            for motorcycle_check in motorcycles:
                for field in MOTORCYCLE_FIELDS:
                    if not motorcycle_check.get(field):
                        messagebox.showwarning(
                            "Incomplete Pass D",
                            f"{motorcycle_check['motorcycle_id']} is missing {field}.",
                        )
                        return

        backup_once(IMAGES_JSONL)
        backup_once(MOTORCYCLES_JSONL)
        atomic_write_jsonl(IMAGES_JSONL, image_records)
        atomic_write_jsonl(MOTORCYCLES_JSONL, motorcycle_records)

        self.current_index += 1

        if self.current_index >= len(image_records):
            messagebox.showinfo(
                "Complete",
                "All image records have completed Pass C/D in this metadata editor.",
            )
            self.current_index = len(image_records) - 1
            return

        self.load_current_image()

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = MetadataEditor(root)
    root.mainloop()