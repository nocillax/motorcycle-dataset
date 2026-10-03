import csv
import json
import re
import shutil
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw, ImageTk

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
METADATA_DIR = BASE_DIR / "metadata"
REPORTS_DIR = BASE_DIR / "reports"

MOTORCYCLES_JSONL = METADATA_DIR / "motorcycles.jsonl"
IMAGES_JSONL = METADATA_DIR / "images.jsonl"
REVIEW_CSV = REPORTS_DIR / "association_review_needed.csv"


def canonicalize_image_id(stem: str) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", stem, maxsplit=1, flags=re.IGNORECASE)[0]


def load_jsonl(path):
    if not path.exists():
        return []
    records = []
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


def load_review_rows():
    if not REVIEW_CSV.exists():
        return []
    with open(REVIEW_CSV, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_review_rows(rows):
    if not rows:
        if REVIEW_CSV.exists():
            REVIEW_CSV.unlink()
        return

    temp_path = REVIEW_CSV.with_suffix(".tmp")
    fieldnames = ["image_id", "split", "issue_type", "rider_id", "details"]

    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(REVIEW_CSV)


def backup_once(path):
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def build_image_index():
    """Maps canonical image IDs to actual exported dataset images."""
    index = {}

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


def draw_box(draw, box, image_size, outline, width=3):
    image_width, image_height = image_size
    x_c, y_c, w, h = box
    xmin = int((x_c - w / 2) * image_width)
    xmax = int((x_c + w / 2) * image_width)
    ymin = int((y_c - h / 2) * image_height)
    ymax = int((y_c + h / 2) * image_height)
    draw.rectangle([xmin, ymin, xmax, ymax], outline=outline, width=width)


def reconcile_association_metadata(motorcycle_records, image_records):
    changed = False
    motorcycles_by_image = {}

    for motorcycle in motorcycle_records:
        motorcycles_by_image.setdefault(
            motorcycle["image_id"], []
        ).append(motorcycle)

        riders = motorcycle.get("riders", [])
        unresolved = motorcycle.get("unresolved_riders", [])

        # B1 owns association state; B2 owns classification state.
        # Never overwrite an existing classification_status.
        for rider in riders:
            if "classification_status" not in rider:
                rider["classification_status"] = (
                    "review_required"
                    if rider.get("class_name") == "ambiguous_rider"
                    else "verified"
                )
                changed = True

        if unresolved:
            if motorcycle.get("association_status") != "review_required":
                motorcycle["association_status"] = "review_required"
                changed = True

            if motorcycle.get("rider_count") is not None:
                motorcycle["rider_count"] = None
                changed = True

            if motorcycle.get("rider_count_status") != "uncertain":
                motorcycle["rider_count_status"] = "uncertain"
                changed = True

            if motorcycle.get("overloaded") is not None:
                motorcycle["overloaded"] = None
                changed = True
        else:
            rider_count = len(riders)

            if motorcycle.get("rider_count") != rider_count:
                motorcycle["rider_count"] = rider_count
                changed = True

            if motorcycle.get("rider_count_status") != "complete":
                motorcycle["rider_count_status"] = "complete"
                changed = True

            expected_overloaded = rider_count > 2
            if motorcycle.get("overloaded") != expected_overloaded:
                motorcycle["overloaded"] = expected_overloaded
                changed = True

        rider_classification_status = (
            "review_required"
            if any(
                rider.get("classification_status") == "review_required"
                for rider in riders
            )
            else "verified"
        )

        if motorcycle.get("rider_classification_status") != rider_classification_status:
            motorcycle["rider_classification_status"] = rider_classification_status
            changed = True

    for image in image_records:
        image_id = image["image_id"]
        motorcycles = motorcycles_by_image.get(image_id, [])

        image_association_status = (
            "review_required"
            if any(m.get("unresolved_riders") for m in motorcycles)
            else "verified"
        )

        if image.get("association_status") != image_association_status:
            image["association_status"] = image_association_status
            changed = True

    return changed


# Load data
motorcycle_records = load_jsonl(MOTORCYCLES_JSONL)
image_records = load_jsonl(IMAGES_JSONL)
review_rows = load_review_rows()

metadata_changed = reconcile_association_metadata(
    motorcycle_records, image_records
)

if metadata_changed:
    backup_once(MOTORCYCLES_JSONL)
    backup_once(IMAGES_JSONL)
    atomic_write_jsonl(MOTORCYCLES_JSONL, motorcycle_records)
    atomic_write_jsonl(IMAGES_JSONL, image_records)
    print("[✓] Association metadata reconciled.")

if not review_rows:
    print("[✓] No association issues require review.")
    sys.exit(0)

image_index = build_image_index()

motorcycles_by_image = {}
for record in motorcycle_records:
    motorcycles_by_image.setdefault(record["image_id"], []).append(record)

images_by_id = {record["image_id"]: record for record in image_records}

# Build review queue
queue = []
seen_queue_keys = set()

for row in review_rows:
    key = (row["image_id"], row.get("rider_id", ""))
    if key in seen_queue_keys:
        continue
    seen_queue_keys.add(key)
    queue.append(row)


class AssociationReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Motorcycle Dataset — Pass B Association Review")
        self.root.geometry("1500x900")
        self.current_index = 0
        self.photo = None
        self.build_ui()
        self.load_current_issue()

    def build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_panel = ttk.Frame(main)
        self.image_panel.pack(side="left", fill="both", expand=True)

        self.control_panel = ttk.Frame(main)
        self.control_panel.pack(side="right", fill="y", padx=(10, 0))

        self.image_label = ttk.Label(self.image_panel, text="Loading...")
        self.image_label.pack(fill="both", expand=True)

        self.issue_label = ttk.Label(
            self.control_panel, text="", wraplength=400, justify="left"
        )
        self.issue_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(
            self.control_panel, text="Motorcycles in this image:"
        ).pack(anchor="w")

        self.motorcycle_list = tk.Listbox(
            self.control_panel, width=60, height=25
        )
        self.motorcycle_list.pack(fill="y", expand=True)

        ttk.Button(
            self.control_panel,
            text="ASSIGN RIDER TO SELECTED MOTORCYCLE",
            command=self.assign_rider,
        ).pack(fill="x", pady=(10, 5))

        ttk.Button(
            self.control_panel,
            text="Next Issue",
            command=self.next_issue,
        ).pack(fill="x", pady=5)

        self.status_label = ttk.Label(
            self.control_panel, text="", wraplength=400
        )
        self.status_label.pack(anchor="w", pady=(10, 0))

    def current_issue(self):
        if self.current_index >= len(queue):
            return None
        return queue[self.current_index]

    def get_rider_info(self, image_id, rider_id):
        records = motorcycles_by_image.get(image_id, [])

        for record in records:
            for rider in record.get("unresolved_riders", []):
                if rider.get("rider_id") == rider_id:
                    return {
                        "rider_id": rider_id,
                        "role": rider["role"],
                        "helmet": rider["helmet"],
                        "helmet_annotation_valid": rider["helmet_annotation_valid"],
                        "class_name": rider["class_name"],
                        "box": rider["box"],
                    }

        for record in records:
            for rider in record.get("riders", []):
                if rider.get("rider_id") == rider_id:
                    return {
                        "rider_id": rider_id,
                        "role": rider["role"],
                        "helmet": rider["helmet"],
                        "helmet_annotation_valid": rider["helmet_annotation_valid"],
                        "class_name": rider["class_name"],
                        "box": rider["box"],
                    }

        return None

    def display_image(self):
        issue = self.current_issue()
        if issue is None:
            return

        image_id = issue["image_id"]
        image_info = image_index.get(image_id)

        if image_info is None:
            self.image_label.configure(
                text=f"Image {image_id} could not be located in dataset/*/images/"
            )
            return

        image = Image.open(image_info["path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        records = motorcycles_by_image.get(image_id, [])
        rider_id = issue.get("rider_id", "")
        rider_info = self.get_rider_info(image_id, rider_id)

        colors = [
            "red", "blue", "green", "yellow",
            "orange", "purple", "cyan", "magenta",
        ]

        for index, record in enumerate(records):
            color = colors[index % len(colors)]
            draw_box(draw, record["motorcycle_box"], image.size, color, width=4)

            x_c, y_c, _, _ = record["motorcycle_box"]
            x = int(x_c * image.width)
            y = int(y_c * image.height)
            draw.text((x, y), f"MC_{index + 1:02d}", fill=color)

        if rider_info:
            draw_box(draw, rider_info["box"], image.size, "white", width=6)

        max_width, max_height = 1000, 820
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        display_size = (
            int(image.width * scale),
            int(image.height * scale),
        )

        image = image.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo, text="")

    def load_current_issue(self):
        issue = self.current_issue()
        self.motorcycle_list.delete(0, tk.END)

        if issue is None:
            self.issue_label.configure(text="No association issues remain.")
            self.status_label.configure(text="PASS B COMPLETE")
            messagebox.showinfo(
                "Complete",
                "All association-review issues are resolved.",
            )
            return

        image_id = issue["image_id"]
        rider_id = issue.get("rider_id", "")

        self.issue_label.configure(
            text=(
                f"Issue {self.current_index + 1} of {len(queue)}\n\n"
                f"Image: {image_id}\n"
                f"Rider: {rider_id}\n"
                f"Type: {issue['issue_type']}\n\n"
                f"{issue['details']}\n\n"
                "Select the motorcycle this rider actually belongs to, "
                "then click ASSIGN."
            )
        )

        records = motorcycles_by_image.get(image_id, [])
        candidate_ids = set()

        for record in records:
            for rider in record.get("unresolved_riders", []):
                if rider.get("rider_id") == rider_id:
                    candidate_ids.update(rider.get("candidate_motorcycles", []))

        for index, record in enumerate(records):
            rider_count = record.get("rider_count")
            count_text = "?" if rider_count is None else str(rider_count)
            mc_id = record["motorcycle_id"]
            candidate_marker = "  <-- candidate" if mc_id in candidate_ids else ""

            self.motorcycle_list.insert(
                tk.END,
                f"{index + 1:02d}. {mc_id} | riders={count_text} | "
                f"status={record['association_status']}{candidate_marker}",
            )

            if mc_id in candidate_ids:
                self.motorcycle_list.selection_set(index)

        self.status_label.configure(
            text="Select the correct motorcycle. Candidate(s) are highlighted in the list."
        )
        self.display_image()

    def assign_rider(self):
        issue = self.current_issue()
        if issue is None:
            return

        selection = self.motorcycle_list.curselection()
        if not selection:
            messagebox.showwarning(
                "No selection",
                "Select the motorcycle that owns this rider.",
            )
            return

        selected_index = selection[0]
        image_id = issue["image_id"]
        rider_id = issue.get("rider_id", "")
        records = motorcycles_by_image.get(image_id, [])

        if selected_index >= len(records):
            return

        target = records[selected_index]
        rider = self.get_rider_info(image_id, rider_id)

        if rider is None:
            messagebox.showerror(
                "Rider not found",
                "The rider could not be recovered from motorcycles.jsonl.",
            )
            return

        # Remove rider from every motorcycle in this image.
        for record in records:
            record["unresolved_riders"] = [
                r for r in record.get("unresolved_riders", [])
                if r.get("rider_id") != rider_id
            ]
            record["riders"] = [
                r for r in record.get("riders", [])
                if r.get("rider_id") != rider_id
            ]

        # Add rider to selected motorcycle.
        target["riders"].append({
            "rider_id": rider["rider_id"],
            "role": rider["role"],
            "helmet": rider["helmet"],
            "helmet_annotation_valid": rider["helmet_annotation_valid"],
            "class_name": rider["class_name"],
            "box": rider["box"],
            "association_status": "verified",
            "classification_status": (
                rider.get("classification_status")
                if rider.get("classification_status")
                else (
                    "review_required"
                    if rider["class_name"] == "ambiguous_rider"
                    else "verified"
                )
            ),
            "resolved_class_name": rider.get("resolved_class_name"),
            "resolved_class_id": rider.get("resolved_class_id"),
        })

        global review_rows

        review_rows = [
            row for row in review_rows
            if not (
                row["image_id"] == image_id
                and row.get("rider_id", "") == rider_id
            )
        ]

        write_review_rows(review_rows)

        image_still_has_review = any(
            row["image_id"] == image_id
            for row in review_rows
        )

        json_still_unresolved = any(
            record.get("unresolved_riders")
            for record in records
        )

        association_complete = not (
            image_still_has_review or json_still_unresolved
        )

        for record in records:
            unresolved_riders = record.get("unresolved_riders", [])

            if unresolved_riders:
                record["association_status"] = "review_required"
                record["rider_count"] = None
                record["rider_count_status"] = "uncertain"
                record["overloaded"] = None
                continue

            # Preserve existing auto_candidate/verified state.
            if not record.get("association_status"):
                record["association_status"] = "auto_candidate"

            record["rider_count"] = len(record.get("riders", []))
            record["rider_count_status"] = "complete"
            record["overloaded"] = record["rider_count"] > 2

        for record in image_records:
            if record["image_id"] == image_id:
                record["association_status"] = (
                    "verified" if association_complete else "review_required"
                )

        backup_once(MOTORCYCLES_JSONL)
        backup_once(IMAGES_JSONL)
        atomic_write_jsonl(MOTORCYCLES_JSONL, motorcycle_records)
        atomic_write_jsonl(IMAGES_JSONL, image_records)

        queue.pop(self.current_index)

        if self.current_index >= len(queue):
            self.current_index = max(0, len(queue) - 1)

        self.load_current_issue()

    def next_issue(self):
        if not queue:
            return

        self.current_index += 1
        if self.current_index >= len(queue):
            self.current_index = 0

        self.load_current_issue()


root = tk.Tk()
app = AssociationReviewer(root)
root.mainloop()