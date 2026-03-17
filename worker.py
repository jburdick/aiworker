import time
import requests
import psycopg2
from megadetector.detection import run_detector
import os
import json

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

        # Get unprocessed media
        cur.execute("""
            SELECT id, file_url
            FROM media
            WHERE ai_processed = false
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

        with open(TEMP_IMAGE, "wb") as f:
            f.write(response.content)

        # =========================
        # LOAD DETECTOR (ONCE)
        # =========================

        global detector
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

        detections = results.get("detections", [])

        # =========================
        # SAVE TO DB
        # =========================

        cur.execute("""
            UPDATE media
            SET detections = %s,
                ai_processed = true
            WHERE id = %s;
        """, (json.dumps(detections), media_id))

        print(f"✅ Done: {media_id}")

    except Exception as e:
        print("🔥 ERROR:", e)
        time.sleep(5)