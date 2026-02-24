import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

from mediapipe.tasks.python.vision import FaceLandmarker, PoseLandmarker
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ---- Landmark Indices ----
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# ---- EAR Calculation ----
def calculate_EAR(landmarks, eye_indices, frame_w, frame_h):
    points = []
    for i in eye_indices:
        x = landmarks[i].x * frame_w
        y = landmarks[i].y * frame_h
        points.append((x, y))
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return round((A + B) / (2.0 * C), 4)

# ---- MAR Calculation ----
def calculate_MAR(landmarks, mouth_indices, frame_w, frame_h):
    points = []
    for i in mouth_indices:
        x = landmarks[i].x * frame_w
        y = landmarks[i].y * frame_h
        points.append((x, y))
    A = np.linalg.norm(np.array(points[2]) - np.array(points[6]))
    B = np.linalg.norm(np.array(points[3]) - np.array(points[7]))
    C = np.linalg.norm(np.array(points[4]) - np.array(points[5]))
    D = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    return round((A + B + C) / (2.0 * D), 4)

# ---- SHA Calculation ----
def calculate_SHA(pose_landmarks, frame_w, frame_h):
    nose = pose_landmarks[0]
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    mid_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_w
    mid_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_h
    nose_x = nose.x * frame_w
    nose_y = nose.y * frame_h
    angle = np.degrees(np.arctan2(nose_x - mid_x, mid_y - nose_y))
    return round(angle, 4)

# ---- CSV Setup ----
csv_file = "driver_dataset.csv"

# Create file with header if it doesn't exist yet
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EAR", "MAR", "SHA", "label"])
    print(f"Created new file: {csv_file}")
else:
    print(f"Adding to existing file: {csv_file}")

# ---- Choose Your Label ----
print("\n==============================")
print("Choose what state you will ACT:")
print("1 - focus")
print("2 - drowsy")
print("3 - distracted")
print("==============================")

choice = input("Enter 1, 2, or 3: ").strip()

label_map = {"1": "focus", "2": "drowsy", "3": "distracted"}

if choice not in label_map:
    print("Invalid choice. Exiting.")
    exit()

label = label_map[choice]
print(f"\nLabel selected: {label.upper()}")
print("Press S to START recording")
print("Press Q to STOP and save\n")

# ---- Camera Start ----
cap = cv2.VideoCapture(1)
recording = False
saved_count = 0

# ---- Cache last known values ----
last_ear = None
last_mar = None
last_sha = None

# ---- Setup MediaPipe ----
import os
face_model_path = os.path.join(os.getcwd(), 'models', 'face_landmarker.task')
pose_model_path = os.path.join(os.getcwd(), 'models', 'pose_landmarker.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=vision.RunningMode.IMAGE,
    min_face_detection_confidence=0.2,
    min_face_presence_confidence=0.2,
    min_tracking_confidence=0.2
)
pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
    min_tracking_confidence=0.2
)
face_mesh = FaceLandmarker.create_from_options(face_options)
pose = PoseLandmarker.create_from_options(pose_options)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    face_results = face_mesh.detect(mp_image)
    pose_results = pose.detect(mp_image)

    # ---- Extract and cache EAR, MAR, SHA ----
    if face_results.face_landmarks:
        lm = face_results.face_landmarks[0]
        try:
            left_ear = calculate_EAR(lm, LEFT_EYE, w, h)
            right_ear = calculate_EAR(lm, RIGHT_EYE, w, h)
            last_ear = round((left_ear + right_ear) / 2, 4)
            last_mar = calculate_MAR(lm, MOUTH, w, h)
        except Exception:
            pass  # keep last known values

    if pose_results.pose_landmarks:
        try:
            last_sha = calculate_SHA(pose_results.pose_landmarks[0], w, h)
        except Exception:
            pass  # keep last known SHA

    # ---- Save to CSV if recording ----
    if recording and last_ear is not None and last_sha is not None:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([last_ear, last_mar, last_sha, label])
        saved_count += 1

    # ---- Display on screen ----
    status_color = (0, 0, 255) if recording else (200, 200, 200)
    status_text = "RECORDING" if recording else "READY - Press S to start"

    cv2.putText(frame, f"Label: {label.upper()}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Saved frames: {saved_count}", (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if last_ear is not None:
        cv2.putText(frame, f"EAR: {last_ear}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {last_mar}", (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if last_sha is not None:
        cv2.putText(frame, f"SHA: {last_sha}", (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not recording:
        recording = True
        print("Recording started!")

    elif key == ord('q'):
        print(f"\nStopped. Total frames saved: {saved_count}")
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
pose.close()
print(f"Data saved to {csv_file}")