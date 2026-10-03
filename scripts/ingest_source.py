import os
import json
import subprocess
from datetime import datetime
import cv2

# Configuration
SOURCE_ID = "SRC_001"
VIDEO_URL = "https://youtu.be/T3nLRe_Q6Fk"  # Clean URL without tracking params
START_TIME = "00:00:00"  # Start time in HH:MM:SS
END_TIME = "00:00:14"    # End time in HH:MM:SS (video is 14s)

SOURCE_METADATA = {
    "source_id": SOURCE_ID,
    "platform": "YouTube",
    "url": VIDEO_URL,
    "title": "Night Traffic Test Clip",
    "uploader": "Test Channel",
    "license": "CC BY 4.0",
    "license_verified": True,
    "access_date": datetime.now().strftime("%Y-%m-%d"),
    "country": "Bangladesh",
    "city": "Dhaka",
    "public_release_status": "approved_for_smoke_test"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEOS_DIR = os.path.join(BASE_DIR, "raw_videos")
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "raw_images")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Step 3: Download targeted video segment using yt-dlp
video_out_path = os.path.join(RAW_VIDEOS_DIR, f"{SOURCE_ID}.mp4")

# Correct format: *START-END
section_arg = f"*{START_TIME}-{END_TIME}"

ytdlp_cmd = [
    "yt-dlp",
    "--download-sections", section_arg,
    "--force-keyframes-at-cuts",
    "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
    "-o", video_out_path,
    VIDEO_URL
]

print(f"[+] Downloading segment ({section_arg}) for {SOURCE_ID}...")
subprocess.run(ytdlp_cmd, check=True)

# Step 2: Record Source Metadata ONLY if download succeeded
sources_file = os.path.join(METADATA_DIR, "sources.jsonl")

# Ensure no duplicate source_id entries
existing_ids = set()
if os.path.exists(sources_file):
    with open(sources_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_ids.add(json.loads(line).get("source_id"))

if SOURCE_ID not in existing_ids:
    with open(sources_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(SOURCE_METADATA) + "\n")
    print(f"[+] Source {SOURCE_ID} logged to {sources_file}")
else:
    print(f"[*] Source {SOURCE_ID} was already recorded in {sources_file}")

# Step 5: Frame Extraction at 1 fps
print(f"[+] Extracting frames at 1 fps from {video_out_path}...")
cap = cv2.VideoCapture(video_out_path)
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

frame_interval = int(round(fps))
frame_idx = 0
extracted_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx % frame_interval == 0:
        timestamp_sec = round(frame_idx / fps, 2)
        frame_filename = f"{SOURCE_ID}_{extracted_count:06d}_t{timestamp_sec:05.1f}s.jpg"
        frame_path = os.path.join(RAW_IMAGES_DIR, frame_filename)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted_count += 1
        
    frame_idx += 1

cap.release()
print(f"[✓] Extraction complete. {extracted_count} frames saved to {RAW_IMAGES_DIR}")