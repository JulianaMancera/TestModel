import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import collections

# ------------------------------------------------
# STEP 1 — Load TFLite Model
# ------------------------------------------------
interpreter = tf.lite.Interpreter(model_path=r'c:\Users\juliana\OneDrive\Documents\GitHub\TestModel\driver_monitor_model\venv\assets\driver_monitor_tcn.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded!")

# ------------------------------------------------
# STEP 2 — Label + Color Map
# ------------------------------------------------
label_map  = {0: 'DISTRACTED', 1: 'DROWSY', 2: 'FOCUS'}
color_map  = {
    'FOCUS':      (0, 255, 0),    # green
    'DROWSY':     (0, 165, 255),  # orange
    'DISTRACTED': (0, 0, 255)     # red
}

# ------------------------------------------------
# STEP 3 — MediaPipe Setup
# ------------------------------------------------
face_model_path = 'c:/Users/juliana/OneDrive/Documents/GitHub/TestModel/models/face_landmarker.task'
pose_model_path = 'c:/Users/juliana/OneDrive/Documents/GitHub/TestModel/models/pose_landmarker.task'

latest_face_result = None
latest_pose_result = None

def face_callback(result, output_image, timestamp_ms):
    global latest_face_result
    latest_face_result = result

def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result

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

# ------------------------------------------------
# STEP 4 — Landmark Indices
# ------------------------------------------------
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

def calculate_EAR(landmarks, eye_indices, w, h):
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return round((A + B) / (2.0 * C), 4)

def calculate_MAR(landmarks, mouth_indices, w, h):
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    A = np.linalg.norm(np.array(points[2]) - np.array(points[6]))
    B = np.linalg.norm(np.array(points[3]) - np.array(points[7]))
    C = np.linalg.norm(np.array(points[4]) - np.array(points[5]))
    D = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    return round((A + B + C) / (2.0 * D), 4)

def calculate_SHA(pose_landmarks, w, h):
    nose = pose_landmarks[0]
    ls   = pose_landmarks[11]
    rs   = pose_landmarks[12]
    mid_x = (ls.x + rs.x) / 2 * w
    mid_y = (ls.y + rs.y) / 2 * h
    return round(np.degrees(np.arctan2(nose.x * w - mid_x, mid_y - nose.y * h)), 4)

def predict(sequence):
    input_data = np.array([sequence], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    idx = np.argmax(output[0])
    return label_map[idx], output[0][idx] * 100

# ------------------------------------------------
# STEP 5 — Calibration + Live Detection
# ------------------------------------------------
cap = cv2.VideoCapture(0)
timestamp  = 0
last_ear   = None
last_mar   = None
last_sha   = None
baseline_sha = None

# Sequence buffer — stores last 30 frames of features
SEQUENCE_LENGTH = 30
sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

current_label = "Calibrating..."
current_conf  = 0.0

with mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as face_lm, \
     mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_lm:

    # ---- Calibration Phase ----
    print("Calibrating... sit naturally and look straight ahead.")
    calib_shas = []
    calib_count = 0
    calib_target = 100

    while cap.isOpened() and calib_count < calib_target:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_lm.detect_async(mp_img, timestamp)
        pose_lm.detect_async(mp_img, timestamp)
        timestamp += 1

        if latest_pose_result and latest_pose_result.pose_landmarks:
            try:
                calib_shas.append(
                    calculate_SHA(latest_pose_result.pose_landmarks[0], w, h)
                )
            except:
                pass

        calib_count += 1
        progress = int((calib_count / calib_target) * 100)

        cv2.putText(frame, "CALIBRATING - Look straight ahead", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Progress: {progress}%", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Driver Monitor - Live Test", frame)
        cv2.waitKey(1)

    baseline_sha = round(np.mean(calib_shas), 4) if calib_shas else 0.0
    print(f"Calibration done! Baseline SHA: {baseline_sha}")
    print("Live detection starting... Press Q to quit.")

    # ---- Live Detection Phase ----
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_lm.detect_async(mp_img, timestamp)
        pose_lm.detect_async(mp_img, timestamp)
        timestamp += 1

        # Extract features
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
                raw_sha  = calculate_SHA(latest_pose_result.pose_landmarks[0], w, h)
                last_sha = round(raw_sha - baseline_sha, 4)
            except:
                pass

        # Add to sequence buffer
        if last_ear is not None and last_sha is not None:
            sequence_buffer.append([last_ear, last_mar, last_sha])

        # Predict when buffer is full
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            current_label, current_conf = predict(list(sequence_buffer))

        # ---- Display ----
        color = color_map.get(current_label, (255, 255, 255))

        # Background bar for label
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(frame, f"{current_label}  {current_conf:.1f}%", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Feature values
        if last_ear is not None:
            cv2.putText(frame, f"EAR: {last_ear}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {last_mar}", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if last_sha is not None:
            cv2.putText(frame, f"SHA: {last_sha}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # Buffer fill indicator
        buf_len = len(sequence_buffer)
        cv2.putText(frame, f"Buffer: {buf_len}/{SEQUENCE_LENGTH}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Driver Monitor - Live Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Live test ended.")