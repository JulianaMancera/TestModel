import tensorflow as tf
import numpy as np
import cv2
import os
import json
import urllib.request
import tempfile
from collections import deque
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe is required.  pip install mediapipe")
    raise SystemExit(1)

# CONFIG
TFLITE_MODEL_PATH    = r"C:\Users\juliana\OneDrive\Documents\GitHub\TestModel\driver_monitor_model\test_model\dms_hybridnet_v3_float32.tflite"
NORM_PARAMS_PATH     = r"C:\Users\juliana\OneDrive\Documents\GitHub\TestModel\driver_monitor_model\test_model\norm_params.json"
PHONE_STREAM_URL     = "http://192.168.8.155:8080/video"
CONFIDENCE_THRESHOLD = 0.4
FLIP_CAMERA          = False
DISPLAY_WIDTH        = 720
DISPLAY_HEIGHT       = 480
SEQ_LEN              = 30
SENTINEL             = -999.0
MAR_YAWN_THRESH      = 0.5
IMG_SIZE             = (224, 224)

BEHAVIOR_CLASSES = [
    "safe_driving",
    "talking",
    "texting",
    "phone",
    "radio",
    "drinking",
    "body_turn",
    "grooming",
    "smoking",
    "yawning",
    "yawning_occ",
    "fatigue",
    "microsleep",
]
PARENT_CLASSES = ["NATURAL", "DISTRACTED", "DROWSY"]
PARENT_MAP     = {0: 0, 1: 0,
                  2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
                  9: 2, 10: 2, 11: 2, 12: 2}
GAZE_ZONES = ["ROAD", "LAP", "LEFT", "L.MIR", "RIGHT", "R.MIR", "WHEEL", "N/A"]

LABEL_KEYS = {ord('1'): 0, ord('2'): 1, ord('3'): 2}

COLORS = {
    0: (50, 205,  50),
    1: (0,  130, 255),
    2: (0,   60, 220),
}

# MEDIAPIPE LANDMARK INDICES (same indices work in Tasks API)
_L_EYE = [33, 160, 158, 133, 153, 144]
_R_EYE = [362, 385, 387, 263, 373, 380]
_M_TOP, _M_BOT, _M_L, _M_R = 13, 14, 61, 291
_L_IRIS, _R_IRIS = 468, 473
_POSE_LMKS = [1, 199, 33, 263, 61, 291]
_POSE_3D   = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)

# STARTUP
print("=" * 60)
print("   TFLITE LIVE CAMERA -- DRIVER MONITOR V3")
print("=" * 60)

print(f"\n[1/4] Loading TFLite model...")
if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"Model not found: {TFLITE_MODEL_PATH}"); raise SystemExit(1)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
_inp = interpreter.get_input_details()
_out = interpreter.get_output_details()

def _tensor_by_name(details, kw):
    for d in details:
        if kw in d['name']:
            return d['index']
    return details[0]['index']

def _tensor_by_shape(details, size):
    for d in details:
        if d['shape'][-1] == size:
            return d['index']
    return details[0]['index']

_sp_idx  = _tensor_by_name(_inp,  'spatial')
_tm_idx  = _tensor_by_name(_inp,  'temporal')
_beh_idx = _tensor_by_shape(_out, 13)
_par_idx = _tensor_by_shape(_out,  3)
_gaz_idx = _tensor_by_shape(_out,  8)
print(f"Inputs: {len(_inp)}  Outputs: {len(_out)}")

print(f"\n[2/4] Loading norm params...")
if not os.path.exists(NORM_PARAMS_PATH):
    print(f"Not found: {NORM_PARAMS_PATH}"); raise SystemExit(1)
with open(NORM_PARAMS_PATH) as fh:
    _np_data = json.load(fh)
NORM_MEAN  = np.array(_np_data['mean'],  dtype=np.float32)
NORM_SCALE = np.array(_np_data['scale'], dtype=np.float32)
print("Norm params loaded.")

# MEDIAPIPE INIT -- Tasks API (MediaPipe 0.10.13+ / Python 3.12 compatible)
print("\n[3/4] Initializing MediaPipe Tasks API...")

def _download_task_model(url, fname):
    cache_dir = os.path.join(tempfile.gettempdir(), "mp_models")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        print(f"  Downloading {fname} (one-time setup)...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved: {path}")
    else:
        print(f"  Found cached: {fname}")
    return path

_BASE = "https://storage.googleapis.com/mediapipe-models"
_face_model = _download_task_model(
    f"{_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "face_landmarker.task")
_pose_model = _download_task_model(
    f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_lite.task")
_hand_model = _download_task_model(
    f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "hand_landmarker.task")

from mediapipe.tasks import python as _mp_tasks                         # type: ignore
from mediapipe.tasks.python import vision as _mp_vision                 # type: ignore
from mediapipe.tasks.python.core.base_options import BaseOptions        # type: ignore

_IMAGE_MODE = _mp_vision.RunningMode.IMAGE

_face_landmarker = _mp_vision.FaceLandmarker.create_from_options(
    _mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_face_model),
        running_mode=_IMAGE_MODE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    ))

_pose_landmarker = _mp_vision.PoseLandmarker.create_from_options(
    _mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_pose_model),
        running_mode=_IMAGE_MODE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

_hand_landmarker = _mp_vision.HandLandmarker.create_from_options(
    _mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_hand_model),
        running_mode=_IMAGE_MODE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

print(f"  MediaPipe {mp.__version__} Tasks API ready.")

print(f"\n[4/4] Connecting to {PHONE_STREAM_URL}...")
cap = cv2.VideoCapture(PHONE_STREAM_URL)
if not cap.isOpened():
    print("Could not connect. Verify IP Webcam is running."); raise SystemExit(1)
print("Connected!\n")

# FEATURE EXTRACTION
def _d(a, b):
    return float(np.linalg.norm(np.subtract(a, b)))

def _ear_from_coords(p):
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-6)

def _head_pose_from_pts(pts2d, w, h):
    f_len = float(w)
    cam   = np.array([[f_len, 0, w/2], [0, f_len, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_POSE_3D, pts2d, cam, np.zeros((4, 1)))
    if not ok:
        return SENTINEL, SENTINEL, SENTINEL
    rmat, _ = cv2.Rodrigues(rvec)
    ang, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(ang[0]), float(ang[1]), float(ang[2])


def extract_features(rgb):
    """Extract 20 base features from one RGB frame using Tasks API."""
    h, w  = rgb.shape[:2]
    feats = [SENTINEL] * 20
    feats[18] = 0.0
    feats[19] = 0.0
    face_box  = None

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Face landmarks
    face_result = _face_landmarker.detect(mp_img)
    if face_result.face_landmarks:
        lm = face_result.face_landmarks[0]  # list of NormalizedLandmark

        def lmxy(i):
            return (lm[i].x * w, lm[i].y * h)

        ear_l = _ear_from_coords([lmxy(i) for i in _L_EYE])
        ear_r = _ear_from_coords([lmxy(i) for i in _R_EYE])
        feats[0] = ear_l
        feats[1] = ear_r
        feats[2] = (ear_l + ear_r) / 2.0
        feats[3] = min(ear_l, ear_r)

        # MAR
        t, b = lmxy(_M_TOP), lmxy(_M_BOT)
        ml, mr = lmxy(_M_L), lmxy(_M_R)
        feats[4] = _d(t, b) / (_d(ml, mr) + 1e-6)

        # Head pose
        pts2d = np.array([lmxy(i) for i in _POSE_LMKS], dtype=np.float64)
        feats[5], feats[6], feats[7] = _head_pose_from_pts(pts2d, w, h)

        # Iris
        feats[8]  = lm[_L_IRIS].x
        feats[9]  = lm[_L_IRIS].y
        feats[10] = lm[_R_IRIS].x
        feats[11] = lm[_R_IRIS].y

        xs = [lk.x * w for lk in lm]
        ys = [lk.y * h for lk in lm]
        face_box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    # Pose landmarks
    pose_result = _pose_landmarker.detect(mp_img)
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]
        ls, rs = lm[11], lm[12]
        lw_lm, rw_lm = lm[15], lm[16]
        span = abs(ls.x - rs.x) * w
        if span > 1e-6:
            feats[12] = (lw_lm.x * w - ls.x * w) / span
            feats[13] = (lw_lm.y * h - ls.y * h) / span
            feats[14] = (rw_lm.x * w - rs.x * w) / span
            feats[15] = (rw_lm.y * h - rs.y * h) / span
        feats[16] = (ls.x + rs.x) / 2.0
        feats[17] = (ls.y + rs.y) / 2.0

    # Hand landmarks
    hand_result = _hand_landmarker.detect(mp_img)
    if hand_result.hand_landmarks and face_box is not None:
        fx, fy, fw, fh = face_box
        mx1, mx2 = fx + fw * 0.2, fx + fw * 0.8
        my1, my2 = fy + fh * 0.55, fy + fh * 0.9
        for hlm in hand_result.hand_landmarks:
            for lk in hlm:
                hx, hy = lk.x * w, lk.y * h
                if fx < hx < fx + fw and fy < hy < fy + fh:
                    feats[18] = 1.0
                if mx1 < hx < mx2 and my1 < hy < my2:
                    feats[19] = 1.0

    return [float(v) for v in feats]


def _window_stats(buf):
    ear = [b[2] for b in buf if b[2] != SENTINEL and not np.isnan(b[2])]
    mar = [b[4] for b in buf if b[4] != SENTINEL and not np.isnan(b[4])]
    ear_mean  = float(np.mean(ear)) if ear else 0.0
    ear_min   = float(np.min(ear))  if ear else 0.0
    mar_max   = float(np.max(mar))  if mar else 0.0
    mar_above = (sum(1 for v in mar if v > MAR_YAWN_THRESH) / len(mar) if mar else 0.0)
    ear_trend = 0.0
    if len(ear) >= 2:
        vi = [i for i, b in enumerate(buf)
              if b[2] != SENTINEL and not np.isnan(b[2])]
        xs = np.array(vi,  dtype=np.float64)
        ys = np.array(ear, dtype=np.float64)
        n, sx, sy = float(len(xs)), xs.sum(), ys.sum()
        denom = n * (xs * xs).sum() - sx * sx
        if abs(denom) > 1e-9:
            ear_trend = float((n * (xs * ys).sum() - sx * sy) / denom)
    return [ear_mean, ear_min, mar_max, float(mar_above), ear_trend]


def build_temporal(buf):
    stats = _window_stats(list(buf))
    arr   = np.zeros((1, SEQ_LEN, 25), dtype=np.float32)
    for t, fv in enumerate(buf):
        for f in range(20):
            arr[0, t, f] = (fv[f] - NORM_MEAN[f]) / (NORM_SCALE[f] + 1e-8)
        for f in range(5):
            arr[0, t, 20 + f] = (stats[f] - NORM_MEAN[20 + f]) / (NORM_SCALE[20 + f] + 1e-8)
    return arr


def build_spatial(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE).astype(np.float32)
    return (rgb / 127.5 - 1.0)[np.newaxis, ...]


def infer(spatial, temporal):
    interpreter.set_tensor(_sp_idx, spatial)
    interpreter.set_tensor(_tm_idx, temporal)
    interpreter.invoke()
    beh = interpreter.get_tensor(_beh_idx)[0].copy()
    par = interpreter.get_tensor(_par_idx)[0].copy()
    gaz = interpreter.get_tensor(_gaz_idx)[0].copy()
    return beh, par, gaz

# METRICS
all_true_labels = []
all_pred_labels = []

def _metrics_block(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1


def print_live_metrics():
    if not all_true_labels:
        print("   (No labeled samples yet -- press 1/2/3 to log ground truth)")
        return
    y_true, y_pred = np.array(all_true_labels), np.array(all_pred_labels)
    acc, prec, rec, f1 = _metrics_block(y_true, y_pred)
    print(f"\n{'-'*50}")
    print(f"  LIVE METRICS  (n={len(y_true)}, parent-class)")
    print(f"{'-'*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"{'-'*50}")


def print_final_metrics():
    if not all_true_labels:
        print("\n  No labeled samples -- press 1/2/3 during session to log ground truth.")
        return
    y_true, y_pred = np.array(all_true_labels), np.array(all_pred_labels)
    acc, prec, rec, f1 = _metrics_block(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"   FINAL EVALUATION RESULTS  (n={len(y_true)}, parent-class)")
    print(f"{'='*60}")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec*100:.2f}%")
    print(f"   Recall    : {rec*100:.2f}%")
    print(f"   F1-Score  : {f1*100:.2f}%")
    print(f"\n   PER-CLASS REPORT:")
    print("-" * 50)
    print(classification_report(y_true, y_pred,
                                target_names=PARENT_CLASSES, zero_division=0))
    print("   CONFUSION MATRIX  (rows=true, cols=predicted)")
    print("-" * 50)
    cm = confusion_matrix(y_true, y_pred)
    print(f"{'':>13}" + "".join(f"{c:>13}" for c in PARENT_CLASSES))
    for i, row in enumerate(cm):
        print(f"{PARENT_CLASSES[i]:>13}" + "".join(f"{v:>13}" for v in row))
    print(f"\n{'='*60}")
    if acc >= 0.90:
        print(f"   Looking solid at {acc*100:.1f}%!")
    elif acc >= 0.70:
        print(f"   {acc*100:.1f}% -- might need more training data.")
    else:
        print(f"   {acc*100:.1f}% -- consider retraining.")
    print(f"{'='*60}\n")

# OVERLAY
def draw_overlay(frame, b_cls, b_conf, p_cls, g_cls, par_probs, buf_len, last_true):
    h, w  = frame.shape[:2]
    col   = COLORS.get(p_cls, (200, 200, 200))

    cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (w, 65), col, 2)
    label = BEHAVIOR_CLASSES[b_cls].upper()
    if b_conf < CONFIDENCE_THRESHOLD:
        label = f"? {label}"
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 2, cv2.LINE_AA)
    info = f"{b_conf*100:.1f}%  |  {PARENT_CLASSES[p_cls]}  |  {GAZE_ZONES[g_cls]}"
    cv2.putText(frame, info, (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

    n = len(all_true_labels)
    cv2.putText(frame, f"Buf:{buf_len}/{SEQ_LEN}", (w - 135, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Logged:{n}", (w - 135, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    if last_true is not None:
        cv2.putText(frame, f"True:{PARENT_CLASSES[last_true]}", (w - 135, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    COLORS.get(last_true, (200, 200, 200)), 1, cv2.LINE_AA)

    bx, bw_max = 12, 200
    by0 = h - (3 * 32 + 28)
    cv2.rectangle(frame, (bx - 5, by0 - 6),
                  (bx + bw_max + 5, by0 + 3 * 32 + 6), (20, 20, 20), -1)
    for i, (pn, pp) in enumerate(zip(PARENT_CLASSES, par_probs)):
        by   = by0 + i * 32
        blen = int(pp * bw_max)
        c    = COLORS.get(i, (200, 200, 200))
        cv2.rectangle(frame, (bx, by + 14), (bx + bw_max, by + 26), (50, 50, 50), -1)
        if blen > 0:
            cv2.rectangle(frame, (bx, by + 14), (bx + blen, by + 26), c, -1)
        cv2.putText(frame, f"{pn}: {pp*100:.1f}%", (bx, by + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    c if i == p_cls else (180, 180, 180),
                    2 if i == p_cls else 1, cv2.LINE_AA)

    if buf_len < SEQ_LEN:
        cv2.putText(frame, f"Warming up  ({buf_len}/{SEQ_LEN} frames)",
                    (w // 2 - 130, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, "1=NATURAL  2=DISTRACTED  3=DROWSY  Q=quit+metrics",
                (bx, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
    return frame

# MAIN LOOP
print("[RUN] Controls in the camera window:")
print("      1=NATURAL  2=DISTRACTED  3=DROWSY  Q=quit\n")

frame_buffer   = deque(maxlen=SEQ_LEN)
frame_count    = 0
skip_frames    = 5
last_beh_probs = np.ones(13) / 13
last_par_probs = np.ones(3)  / 3
last_gaz_probs = np.ones(8)  / 8
last_beh_cls   = 0
last_par_cls   = 0
last_gaz_cls   = 0
last_conf      = 0.0
last_true      = None
log_interval   = 30

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.open(PHONE_STREAM_URL)
        continue

    if FLIP_CAMERA:
        frame = cv2.flip(frame, 1)

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_buffer.append(extract_features(rgb))

    if frame_count % skip_frames == 0 and len(frame_buffer) == SEQ_LEN:
        spatial  = build_spatial(frame)
        temporal = build_temporal(frame_buffer)
        last_beh_probs, last_par_probs, last_gaz_probs = infer(spatial, temporal)
        last_beh_cls = int(np.argmax(last_beh_probs))
        last_par_cls = int(np.argmax(last_par_probs))
        last_gaz_cls = int(np.argmax(last_gaz_probs))
        last_conf    = float(last_beh_probs[last_beh_cls])

    disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    disp = draw_overlay(disp, last_beh_cls, last_conf, last_par_cls, last_gaz_cls,
                        last_par_probs, len(frame_buffer), last_true)
    cv2.imshow("Driver Monitor V3 -- 1/2/3 to log | Q to quit", disp)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nQuitting...")
        break

    elif key in LABEL_KEYS:
        true_lbl  = LABEL_KEYS[key]
        last_true = true_lbl
        all_true_labels.append(true_lbl)
        all_pred_labels.append(last_par_cls)
        match = "OK" if true_lbl == last_par_cls else "XX"
        print(f"  [{match}]  True: {PARENT_CLASSES[true_lbl]:>12}  |  "
              f"Pred: {PARENT_CLASSES[last_par_cls]:>12}  |  "
              f"Behavior: {BEHAVIOR_CLASSES[last_beh_cls]:>15}  |  "
              f"Conf: {last_conf*100:.1f}%  |  "
              f"Total: {len(all_true_labels)}")
        if len(all_true_labels) % log_interval == 0:
            print_live_metrics()

cap.release()
cv2.destroyAllWindows()
print_final_metrics()