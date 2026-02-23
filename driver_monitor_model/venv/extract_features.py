import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Model paths
face_model_path = 'c:/Users/juliana/OneDrive/Documents/GitHub/TestModel/models/face_landmarker.task'
holistic_model_path = 'c:/Users/juliana/OneDrive/Documents/GitHub/TestModel/models/holistic_landmarker.task'

# Global variables for latest results
latest_face_result = None
latest_holistic_result = None

def face_callback(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_face_result
    latest_face_result = result

def holistic_callback(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_holistic_result
    latest_holistic_result = result

# Setup Face Landmarker
face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=face_model_path),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=face_callback
)

# Setup Pose Landmarker
holistic_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=holistic_model_path),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=holistic_callback
)

# ---- EAR Calculation ----
def calculate_EAR(landmarks, eye_indices, frame_w, frame_h):
    # Get the 6 eye landmark points
    points = []
    for i in eye_indices:
        x = landmarks[i].x * frame_w
        y = landmarks[i].y * frame_h
        points.append((x, y))

    # Vertical distances
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    EAR = (A + B) / (2.0 * C)
    return round(EAR, 4)

# ---- MAR Calculation ----
def calculate_MAR(landmarks, mouth_indices, frame_w, frame_h):
    points = []
    for i in mouth_indices:
        x = landmarks[i].x * frame_w
        y = landmarks[i].y * frame_h
        points.append((x, y))

    # Vertical distances
    A = np.linalg.norm(np.array(points[2]) - np.array(points[6]))
    B = np.linalg.norm(np.array(points[3]) - np.array(points[7]))
    C = np.linalg.norm(np.array(points[4]) - np.array(points[5]))
    # Horizontal distance
    D = np.linalg.norm(np.array(points[0]) - np.array(points[1]))

    MAR = (A + B + C) / (2.0 * D)
    return round(MAR, 4)

# ---- SHA Calculation ----
def calculate_SHA(pose_landmarks, frame_w, frame_h):
    # Get nose, left shoulder, right shoulder
    nose = pose_landmarks[0]
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    # Midpoint of shoulders
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_w
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_h
    nose_x = nose.x * frame_w
    nose_y = nose.y * frame_h

    # Angle of head relative to shoulder midpoint
    delta_x = nose_x - mid_shoulder_x
    delta_y = mid_shoulder_y - nose_y  # inverted because y goes down in image

    angle = np.degrees(np.arctan2(delta_x, delta_y))
    return round(angle, 4)

# ---- Landmark Indices ----
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# ---- Start Camera ----
cap = cv2.VideoCapture(1)

with mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
     mp.tasks.vision.PoseLandmarker.create_from_options(holistic_options) as holistic_landmarker:

    timestamp = 0
    frame_count = 0
    while cap.isOpened() and frame_count < 300:  # Limit frames
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect asynchronously
        face_landmarker.detect_async(mp_image, timestamp)
        holistic_landmarker.detect_async(mp_image, timestamp)
        timestamp += 1

        ear = "N/A"
        mar = "N/A"
        sha = "N/A"

        # ---- Extract EAR and MAR ----
        if latest_face_result and latest_face_result.face_landmarks:
            landmarks = latest_face_result.face_landmarks[0]

            left_ear = calculate_EAR(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_EAR(landmarks, RIGHT_EYE, w, h)
            ear = round((left_ear + right_ear) / 2, 4)  # average both eyes

            mar = calculate_MAR(landmarks, MOUTH, w, h)

        # ---- Extract SHA ----
        if latest_holistic_result and latest_holistic_result.pose_landmarks:
            sha = calculate_SHA(latest_holistic_result.pose_landmarks[0], w, h)

        # ---- Display values on screen ----
        cv2.putText(frame, f"EAR: {ear}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"SHA: {sha}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

        cv2.imshow("EAR / MAR / SHA Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()