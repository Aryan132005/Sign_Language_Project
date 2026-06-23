"""
app.py — Flask backend API for Sign Language Recognition.
Usage: python app.py
Endpoints:
  POST /api/predict  — accepts base64 JPEG frame, returns predicted sign
  GET  /api/classes  — returns list of supported signs
  GET  /api/health   — health check
"""

import os
import json
import base64
import logging
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

MODEL_PATH = "models/sign_model.h5"
CLASS_PATH = "models/classes.json"

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

model   = None
classes = []


def load_model():
    global model, classes
    if not os.path.exists(MODEL_PATH):
        log.warning("Model file not found. Using demo mode.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        classes = json.load(f)
    log.info(f"Model loaded. Classes: {classes}")


def extract_landmarks(frame_bgr):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None, None, None

    lm_list = result.multi_hand_landmarks[0]
    coords  = []
    for lm in lm_list.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    # Draw landmarks on frame for feedback
    annotated = frame_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated, lm_list, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 200, 100), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(50, 150, 255), thickness=2),
    )
    return np.array(coords), annotated, lm_list


def frame_from_b64(b64_string):
    # Strip data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr   = np.frombuffer(img_bytes, np.uint8)
    frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def frame_to_b64(frame):
    _, buf    = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ── Demo predictions (used when model not yet trained) ──────────────────────
DEMO_SIGNS    = ["hello", "thank_you", "yes", "no", "help",
                 "please", "sorry", "good", "bad", "i_love_you"]
_demo_counter = 0


def demo_predict():
    global _demo_counter
    sign = DEMO_SIGNS[_demo_counter % len(DEMO_SIGNS)]
    _demo_counter += 1
    conf = float(np.random.uniform(0.82, 0.99))
    return sign, conf, {s: round(float(np.random.uniform(0, 0.1)), 3)
                        for s in DEMO_SIGNS}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": classes,
        "demo_mode": model is None,
    })


@app.route("/api/classes", methods=["GET"])
def get_classes():
    c = classes if classes else DEMO_SIGNS
    return jsonify({"classes": c, "count": len(c)})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' field (base64 JPEG)"}), 400

    frame = frame_from_b64(data["frame"])
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    landmarks, annotated_frame, _ = extract_landmarks(frame)

    if landmarks is None:
        return jsonify({
            "hand_detected": False,
            "sign": None,
            "confidence": 0,
            "annotated_frame": frame_to_b64(frame),
        })

    # Real model inference
    if model is not None:
        probs = model.predict(landmarks[np.newaxis, :], verbose=0)[0]
        idx   = int(np.argmax(probs))
        sign  = classes[idx]
        conf  = float(probs[idx])
        all_probs = {classes[i]: round(float(p), 4) for i, p in enumerate(probs)}
    else:
        # Demo mode
        sign, conf, all_probs = demo_predict()

    return jsonify({
        "hand_detected": True,
        "sign": sign,
        "confidence": round(conf, 4),
        "all_probabilities": all_probs,
        "annotated_frame": frame_to_b64(annotated_frame),
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
