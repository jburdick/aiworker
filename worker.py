import time
import requests
import psycopg2
from megadetector.detection import run_detector
import os
import json
from dotenv import load_dotenv
import cv2  # ✅ import at top (IMPORTANT)

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

        with open(TEMP_IMAGE, "wb") as f:
            f.write(response.content)

        # =========================
        # CLEAN IMAGE WITH OPENCV
        # =========================

        try:
            img = cv2.imread(TEMP_IMAGE)

            if img is None:
                raise Exception("cv2 failed to read image")

            h, w = img.shape[:2]
            max_size = 1280

            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            cv2.imwrite(TEMP_IMAGE, img)

        except Exception as e:
            print("⚠️ OpenCV preprocessing failed:", e)

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
            detection_threshold=0.05  # 🔥 lowered for sensitivity
        )

        if not results:
            raw_detections = []
        else:
            raw_detections = results.get("detections") or []

        print("RAW DETECTIONS:", raw_detections)

        # =========================
        # FILTER: ANIMALS ONLY
        # =========================

        animal_detections = [
            d for d in raw_detections
            if str(d.get("category")) == "1"
        ]

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