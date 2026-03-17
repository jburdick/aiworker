import time
import requests
import psycopg2
from megadetector.detection import run_detector
import os
import json
from dotenv import load_dotenv
from PIL import Image

# =========================
# LOAD ENV
# =========================

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# =========================
# CONFIG
# =========================

MODEL_PATH = "models/md_v5a.0.0.pt"
TEMP_IMAGE = "temp.jpg"

# =========================
# GLOBAL DETECTOR
# =========================

detector = None

# =========================
# DB CONNECTION
# =========================

conn = psycopg2.connect(DB_URL)
conn.autocommit = True

print("🚀 Worker started...")

# =========================
# MAIN LOOP
# =========================

while True:
    try:
        cur = conn.cursor()

        # Get next unprocessed media
        cur.execute("""
            SELECT id, file_url
            FROM media
            WHERE ai_processed = false
            ORDER BY id ASC
            LIMIT 1;
        """)

        row = cur.fetchone()

        if not row:
            print("😴 No jobs. Sleeping...")
            time.sleep(5)
            continue

        media_id, file_url = row
        print(f"📸 Processing media ID: {media_id}")

        # =========================
        # DOWNLOAD IMAGE
        # =========================

        response = requests.get(file_url, timeout=10)

        if response.status_code != 200:
            print(f"❌ Failed to download: {file_url}")
            time.sleep(2)
            continue

# Save raw image
with open(TEMP_IMAGE, "wb") as f:
    f.write(response.content)

# 🔥 CLEAN + NORMALIZE IMAGE (CRITICAL FIX)
try:
    img = Image.open(TEMP_IMAGE)

    # Convert to RGB (fix grayscale / weird formats)
    img = img.convert("RGB")

    # Resize to safe dimensions (prevents model crashes)
    img.thumbnail((1280, 1280))

    # Re-save clean image
    img.save(TEMP_IMAGE, format="JPEG")

except Exception as e:
    print("🔥 Image preprocessing failed:", e)

        # =========================
        # LOAD DETECTOR (ONCE)
        # =========================

        if detector is None:
            print("🧠 Loading MegaDetector model...")
            detector = run_detector.load_detector(MODEL_PATH)

        # =========================
        # RUN DETECTION
        # =========================

        results = detector.generate_detections_one_image(
            TEMP_IMAGE,
            detection_threshold=0.2
        )

        if not results:
            raw_detections = []
        else:
            raw_detections = results.get("detections")

        if raw_detections is None:
            raw_detections = []

        # =========================
        # FILTER: ANIMALS ONLY
        # =========================

        print("RAW DETECTIONS:", raw_detections)

        animal_detections = raw_detections or []
        # =========================
        # DERIVED INTELLIGENCE
        # =========================

        animal_detected = len(animal_detections) > 0
        detection_count = len(animal_detections)
        max_confidence = max(
            [d["conf"] for d in animal_detections],
            default=0
        )

        # =========================
        # SAVE TO DB
        # =========================

        cur.execute("""
            UPDATE media
            SET detections = %s,
                ai_processed = true,
                animal_detected = %s,
                detection_count = %s,
                max_confidence = %s
            WHERE id = %s;
        """, (
            json.dumps(animal_detections),
            animal_detected,
            detection_count,
            max_confidence,
            media_id
        ))

        print(f"✅ Done: {media_id} | Animals: {detection_count}")

    except Exception as e:
        print("🔥 ERROR:", e)
        time.sleep(5)