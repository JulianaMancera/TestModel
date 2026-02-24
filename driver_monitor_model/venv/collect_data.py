import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os

# ---- Model Paths (DO NOT CHANGE) ----
face_model_path = 'c:/Users/juliana/OneDrive/Documents/GitHub/TestModel/models/face_landmarker.task'
pose_model_path = 'c:/Users/juliana/OneDrive/Documents/GitHub/TestModel/models/pose_landmarker.task'

# ---- Global Variables (DO NOT CHANGE) ----
latest_face_result = None
latest_pose_result = None

def face_callback(result, output_image, timestamp_ms):
    global latest_face_result
    latest_face_result = result

def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result

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
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_w
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_h
    nose_x = nose.x * frame_w
    nose_y = nose.y * frame_h
    delta_x = nose_x - mid_shoulder_x
    delta_y = mid_shoulder_y - nose_y
    angle = np.degrees(np.arctan2(delta_x, delta_y))
    return round(angle, 4)

# ---- CSV Setup ----
csv_file = "driver_dataset.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EAR", "MAR", "SHA", "label"])
    print(f"Created new file: {csv_file}")
else:
    print(f"Adding to existing file: {csv_file}")

# ---- Choose Label ----
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

# ---- Setup MediaPipe ----
face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=face_model_path),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.2,
    min_face_presence_confidence=0.2,
    min_tracking_confidence=0.2,
    result_callback=face_callback
)
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
    min_tracking_confidence=0.2,
    result_callback=pose_callback
)

# ---- Camera Start ----
cap = cv2.VideoCapture(1)
recording = False
saved_count = 0
timestamp = 0

last_ear = None
last_mar = None
last_sha = None

# ---- Baseline (calibration values) ----
baseline_ear = None
baseline_mar = None
baseline_sha = None

with mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
     mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:

    # ================================================
    # CALIBRATION PHASE
    # ================================================
    print("\n--- CALIBRATION ---")
    print("Sit naturally as if you are DRIVING and FOCUSING.")
    print("Look straight ahead (toward your road direction).")
    print("Keep mouth closed and relaxed.")
    print("Calibration starts in 3 seconds automatically...\n")

    import time
    time.sleep(3)

    calib_ears = []
    calib_mars = []
    calib_shas = []
    calib_frames = 0
    calib_target = 100  # collect 100 frames for baseline

    print("Calibrating... hold still!")

    while cap.isOpened() and calib_frames < calib_target:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_landmarker.detect_async(mp_image, timestamp)
        pose_landmarker.detect_async(mp_image, timestamp)
        timestamp += 1

        if latest_face_result and latest_face_result.face_landmarks:
            lm = latest_face_result.face_landmarks[0]
            try:
                le = calculate_EAR(lm, LEFT_EYE, w, h)
                re = calculate_EAR(lm, RIGHT_EYE, w, h)
                calib_ears.append((le + re) / 2)
                calib_mars.append(calculate_MAR(lm, MOUTH, w, h))
            except:
                pass

        if latest_pose_result and latest_pose_result.pose_landmarks:
            try:
                calib_shas.append(
                    calculate_SHA(latest_pose_result.pose_landmarks[0], w, h)
                )
            except:
                pass

        calib_frames += 1
        progress = int((calib_frames / calib_target) * 100)

        # Show calibration progress on screen
        cv2.putText(frame, "CALIBRATING - Look straight ahead", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Progress: {progress}%", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)

    # Save baseline averages
    if calib_ears:
        baseline_ear = round(np.mean(calib_ears), 4)
    if calib_mars:
        baseline_mar = round(np.mean(calib_mars), 4)
    if calib_shas:
        baseline_sha = round(np.mean(calib_shas), 4)

    print(f"\nCalibration complete!")
    print(f"Your baseline EAR: {baseline_ear}  (eyes open normal)")
    print(f"Your baseline MAR: {baseline_mar}  (mouth closed normal)")
    print(f"Your baseline SHA: {baseline_sha}  (head angle at your camera position)")
    print(f"\nNow act as: {label.upper()}")
    print("Press S to START recording")
    print("Press Q to STOP and save\n")

    # ================================================
    # RECORDING PHASE
    # ================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_landmarker.detect_async(mp_image, timestamp)
        pose_landmarker.detect_async(mp_image, timestamp)
        timestamp += 1

        if latest_face_result and latest_face_result.face_landmarks:
            lm = latest_face_result.face_landmarks[0]
            try:
                le = calculate_EAR(lm, LEFT_EYE, w, h)
                re = calculate_EAR(lm, RIGHT_EYE, w, h)
                last_ear = round((le + re) / 2, 4)
                last_mar = calculate_MAR(lm, MOUTH, w, h)
            except:
                pass

        if latest_pose_result and latest_pose_result.pose_landmarks:
            try:
                last_sha = calculate_SHA(
                    latest_pose_result.pose_landmarks[0], w, h
                )
            except:
                pass

        # ---- Normalize SHA against baseline (fixes side camera) ----
        # This makes SHA relative to YOUR natural camera position
        normalized_sha = None
        if last_sha is not None and baseline_sha is not None:
            normalized_sha = round(last_sha - baseline_sha, 4)

        # ---- Save normalized values to CSV ----
        if recording and last_ear is not None and normalized_sha is not None:
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([last_ear, last_mar, normalized_sha, label])
            saved_count += 1

        # ---- Display ----
        status_color = (0, 0, 255) if recording else (200, 200, 200)
        status_text = "RECORDING" if recording else "READY - Press S to start"

        cv2.putText(frame, f"Label: {label.upper()}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Saved: {saved_count}", (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if last_ear is not None:
            cv2.putText(frame, f"EAR: {last_ear} (base:{baseline_ear})", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {last_mar} (base:{baseline_mar})", (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if normalized_sha is not None:
            cv2.putText(frame, f"SHA: {normalized_sha} (0=straight)", (30, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

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
print(f"Data saved to {csv_file}")