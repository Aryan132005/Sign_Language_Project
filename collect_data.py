"""
collect_data.py — Run this FIRST to record your own hand gesture data.
Usage: python collect_data.py
Press 's' to start recording each class, 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(static_image_mode=False,
                             max_num_hands=1,
                             min_detection_confidence=0.7)

CLASSES = ["hello", "thank_you", "yes", "no", "help", "please", "sorry", "good", "bad", "i_love_you"]
NUM_SAMPLES = 200
DATA_DIR    = "data/landmarks"
SEQUENCE_LEN = 30   # frames per dynamic gesture sample

os.makedirs(DATA_DIR, exist_ok=True)


def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        coords = []
        for point in lm.landmark:
            coords.extend([point.x, point.y, point.z])
        return np.array(coords), result.multi_hand_landmarks[0]
    return None, None


def collect():
    cap = cv2.VideoCapture(0)
    print("\n=== Sign Language Data Collector ===")
    print("Classes to record:", CLASSES)
    print("Press 's' to start recording a class")
    print("Press 'q' to quit\n")

    for label in CLASSES:
        save_dir = os.path.join(DATA_DIR, label)
        os.makedirs(save_dir, exist_ok=True)
        count = 0

        print(f"\nReady to record: '{label.upper()}'")
        print("Show the gesture, then press 's' to start.")

        ready = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            landmarks, lm_obj = extract_landmarks(frame)

            if lm_obj:
                mp_drawing.draw_landmarks(frame, lm_obj,
                                          mp_hands.HAND_CONNECTIONS)

            status = f"Class: {label} | Samples: {count}/{NUM_SAMPLES}"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
            if not ready:
                cv2.putText(frame, "Press 's' to start recording",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 200, 0), 2)
            else:
                cv2.putText(frame, "RECORDING...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            cv2.imshow("Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                ready = True

            if ready and landmarks is not None:
                np.save(os.path.join(save_dir, f"{count}.npy"), landmarks)
                count += 1
                if count >= NUM_SAMPLES:
                    print(f"  Done: {NUM_SAMPLES} samples saved for '{label}'")
                    time.sleep(1)
                    break

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    print("\n All classes recorded! Run train_model.py next.")


if __name__ == "__main__":
    collect()
