import json
import cv2
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
METADATA_DIR = BASE_DIR / "metadata"
VIS_DIR = BASE_DIR / "visualizations"

MOTORCYCLES_JSONL = METADATA_DIR / "motorcycles.jsonl"
VIS_DIR.mkdir(parents=True, exist_ok=True)

def canonicalize_image_id(stem: str) -> str:
    """Strip Roboflow suffixes so we can match physical files to clean metadata IDs."""
    return re.split(
        r"_(?:jpg|jpeg|png)\.rf\.",
        stem,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

# Find where the images are and map them using their CLEAN IDs
image_paths = {}
for split in ["train", "valid", "test"]:
    split_dir = DATASET_DIR / split / "images"
    if split_dir.exists():
        for img_file in split_dir.glob("*.jpg"):
            clean_id = canonicalize_image_id(img_file.stem)
            image_paths[clean_id] = str(img_file)

if not MOTORCYCLES_JSONL.exists():
    print(f"[-] Cannot find {MOTORCYCLES_JSONL}")
    exit(1)

records_by_image = {}
with open(MOTORCYCLES_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        record = json.loads(line)
        img_id = record["image_id"]
        if img_id not in records_by_image:
            records_by_image[img_id] = []
        records_by_image[img_id].append(record)

print(f"[+] Generating pointer visualizations for {len(records_by_image)} images...")

def draw_label_with_pointer(img, text, target_pt, offset, color):
    """Draws text connected to a target point by a thin leader line."""
    tx = max(target_pt[0] + offset[0], 10)
    ty = max(target_pt[1] + offset[1], 20)
    
    if tx > img.shape[1] - 150:
        tx = img.shape[1] - 150
        
    cv2.line(img, (tx, ty - 5), target_pt, color, 1, cv2.LINE_AA)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2  
    cv2.putText(img, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

skipped_count = 0

for img_id, records in records_by_image.items():
    if img_id not in image_paths:
        print(f"[!] Warning: Image file for {img_id} not found in dataset/. Skipping.")
        skipped_count += 1
        continue
        
    img = cv2.imread(image_paths[img_id])
    if img is None:
        continue
        
    h_img, w_img = img.shape[:2]

    for rec in records:
        mc_id_short = rec["motorcycle_id"].split("_")[-1]
        
        # 1. Motorcycle Box
        mx_c, my_c, mw, mh = rec["motorcycle_box"]
        mx1, my1 = int((mx_c - mw/2) * w_img), int((my_c - mh/2) * h_img)
        mx2, my2 = int((mx_c + mw/2) * w_img), int((my_c + mh/2) * h_img)
        
        cv2.rectangle(img, (mx1, my1), (mx2, my2), (255, 0, 0), 2)
        
        mc_text = f"MC_{mc_id_short} (Riders: {rec['rider_count']})"
        draw_label_with_pointer(img, mc_text, target_pt=(mx1, my1), offset=(-20, -20), color=(255, 0, 0))

        # 2. Associated Riders
        for rider in rec.get("riders", []):
            rx_c, ry_c, rw, rh = rider["box"]
            rx1, ry1 = int((rx_c - rw/2) * w_img), int((ry_c - rh/2) * h_img)
            rx2, ry2 = int((rx_c + rw/2) * w_img), int((ry_c + rh/2) * h_img)
            
            color = (0, 255, 0) if rider["helmet"] else (0, 0, 255)
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)
            
            r_text = f"{rider['rider_id']}:{rider['role'][:3]}"
            draw_label_with_pointer(img, r_text, target_pt=(rx2, ry1), offset=(15, -15), color=color)

        # 3. Unresolved Riders
        for rider in rec.get("unresolved_riders", []):
            rx_c, ry_c, rw, rh = rider["box"]
            rx1, ry1 = int((rx_c - rw/2) * w_img), int((ry_c - rh/2) * h_img)
            rx2, ry2 = int((rx_c + rw/2) * w_img), int((ry_c + rh/2) * h_img)
            
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            
            r_text = "? CONFLICT"
            draw_label_with_pointer(img, r_text, target_pt=(rx2, ry1), offset=(15, -15), color=(0, 255, 255))

    out_path = VIS_DIR / f"{img_id}_vis.jpg"
    cv2.imwrite(str(out_path), img)

if skipped_count > 0:
    print(f"[-] Skipped {skipped_count} images (files missing).")
print(f"[✓] Done! Check {VIS_DIR.name}/ for clean, single-text pointer annotations.")