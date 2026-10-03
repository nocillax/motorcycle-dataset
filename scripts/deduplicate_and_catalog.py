import os
import json
import csv
import hashlib
import re
import cv2
from PIL import Image
import imagehash

# Configuration
HAMMING_DISTANCE_THRESHOLD = 6  # pHash distance <= 6 is considered duplicate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "raw_images")
DEDUP_IMAGES_DIR = os.path.join(BASE_DIR, "dedup_images")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(DEDUP_IMAGES_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def get_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Fix 6: Strict Regex for Source ID and Timestamp
FILENAME_PATTERN = re.compile(r"^([A-Za-z0-9_]+?)_(\d+)_t([\d\.]+)s\.(?:jpg|jpeg|png)$", re.IGNORECASE)

def extract_metadata_from_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    if match:
        source_id = match.group(1)
        frame_idx = int(match.group(2))
        timestamp_sec = float(match.group(3))
        return source_id, frame_idx, timestamp_sec
    return "SRC_UNKNOWN", 0, 0.0

raw_files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not raw_files:
    print("[-] No raw images found in raw_images/")
    exit(1)

# Fix 2: Chronological Sort by (source_id, timestamp_sec)
raw_files.sort(key=lambda fn: (extract_metadata_from_filename(fn)[0], extract_metadata_from_filename(fn)[2]))

print(f"[+] Found {len(raw_files)} raw frames to process.")

# Fix 3: Robust ID Calculation (Max ID + 1)
images_jsonl_path = os.path.join(METADATA_DIR, "images.jsonl")

existing_img_ids = set()
existing_original_filenames = set()
seen_sha256 = set()
source_last_phash = {}
source_last_timestamp = {}

if os.path.exists(images_jsonl_path):
    with open(images_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            data = json.loads(line_str)

            # Existing image IDs
            if "image_id" in data:
                existing_img_ids.add(data["image_id"])

            # Existing original filenames
            if "original_filename" in data:
                existing_original_filenames.add(data["original_filename"])

            # Persisted SHA-256 of the ORIGINAL raw frame
            if data.get("sha256"):
                seen_sha256.add(data["sha256"])

            # Restore the latest kept pHash for each source
            if data.get("phash") and data.get("source_id") is not None:
                source_id = data["source_id"]
                timestamp_sec = float(data.get("timestamp_sec", 0.0))

                try:
                    restored_phash = imagehash.hex_to_hash(data["phash"])
                except ValueError:
                    restored_phash = None

                if restored_phash is not None:
                    previous_timestamp = source_last_timestamp.get(source_id, -1.0)

                    if timestamp_sec >= previous_timestamp:
                        source_last_timestamp[source_id] = timestamp_sec
                        source_last_phash[source_id] = restored_phash

max_id = 0
for img_id in existing_img_ids:
    match = re.search(r"IMG_(\d+)", img_id)
    if match:
        max_id = max(max_id, int(match.group(1)))

next_img_num = max_id + 1

dedup_records = []
surviving_images = []

for filename in raw_files:
    raw_path = os.path.join(RAW_IMAGES_DIR, filename)
    source_id, frame_idx, timestamp_sec = extract_metadata_from_filename(filename)

    # Skip files that have already been cataloged in a previous run
    if filename in existing_original_filenames:
        continue

    # Fix 4: Single disk read + in-memory RGB conversion
    img_cv = cv2.imread(raw_path)
    if img_cv is None:
        dedup_records.append({
            "filename": filename,
            "source_id": source_id,
            "decision": "REMOVED",
            "reason": "Corrupted or unreadable image",
            "distance": "N/A"
        })
        continue

    height, width = img_cv.shape[:2]
    if width < 320 or height < 240:
        dedup_records.append({
            "filename": filename,
            "source_id": source_id,
            "decision": "REMOVED",
            "reason": "Resolution below minimum threshold",
            "distance": "N/A"
        })
        continue

    # Check Exact Hash (SHA-256)
    sha_hash = get_sha256(raw_path)
    if sha_hash in seen_sha256:
        dedup_records.append({
            "filename": filename,
            "source_id": source_id,
            "decision": "REMOVED",
            "reason": "Exact byte duplicate (SHA-256 match)",
            "distance": 0
        })
        continue

    # Check Perceptual Hash (pHash) scoped per source
    rgb_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_cv)
    current_phash = imagehash.phash(pil_img)

    last_kept_phash = source_last_phash.get(source_id)
    if last_kept_phash is not None:
        distance = current_phash - last_kept_phash
        if distance <= HAMMING_DISTANCE_THRESHOLD:
            dedup_records.append({
                "filename": filename,
                "source_id": source_id,
                "decision": "REMOVED",
                "reason": f"Perceptual duplicate (distance {distance} <= {HAMMING_DISTANCE_THRESHOLD})",
                "distance": distance
            })
            continue
    else:
        distance = "N/A"

    # Frame passes validation
    seen_sha256.add(sha_hash)
    source_last_phash[source_id] = current_phash

    assigned_img_id = f"IMG_{next_img_num:06d}"
    next_img_num += 1
    new_filename = f"{assigned_img_id}.jpg"
    dest_path = os.path.join(DEDUP_IMAGES_DIR, new_filename)

    cv2.imwrite(dest_path, img_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])

    surviving_images.append({
        "image_id": assigned_img_id,
        "source_id": source_id,
        "filename": new_filename,
        "original_filename": filename,
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "sha256": sha_hash,
        "phash": str(current_phash),
        "width": width,
        "height": height,

        # Image-level / scene-level attributes
        "illumination": None,      # night_artificial, dusk, daylight, dark_unlit
        "weather": None,           # clear, rain, fog
        "road_condition": None,    # dry, wet
        "glare": None,             # none, moderate, severe
        "visibility": None,        # good, moderate, poor
        "motion_blur": None,       # none, mild, severe
        "traffic_density": None,   # low, medium, high
        "traffic_context": None,     # free_flow, stop_and_go, congested
        "road_type": None,         # urban_arterial, intersection, alley, highway

        "hard_negative": False,
        "annotation_status": "not_annotated",
        "association_status": "not_processed",
        "rider_classification_status": "not_processed"
    })

    dedup_records.append({
        "filename": filename,
        "source_id": source_id,
        "decision": "KEPT",
        "reason": f"Assigned to {assigned_img_id}",
        "distance": distance
    })

# Append to reports/dedup_report.csv
report_csv_path = os.path.join(REPORTS_DIR, "dedup_report.csv")
write_header = not os.path.exists(report_csv_path)

with open(report_csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "source_id", "decision", "reason", "distance"])
    if write_header:
        writer.writeheader()
    writer.writerows(dedup_records)

# Append to metadata/images.jsonl
with open(images_jsonl_path, "a", encoding="utf-8") as f:
    for rec in surviving_images:
        f.write(json.dumps(rec) + "\n")

print(f"[✓] Processing complete.")
print(f"    - Clean frames preserved: {len(surviving_images)} -> {DEDUP_IMAGES_DIR}")
print(f"    - Duplicates/Corrupt removed: {len(dedup_records) - len(surviving_images)}")
print(f"    - Report updated: {report_csv_path}")
print(f"    - Metadata updated: {images_jsonl_path}")