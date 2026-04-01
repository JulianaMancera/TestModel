import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

#  CONFIGURATION
KERAS_MODEL_PATH     = "model name"
IMG_SIZE             = (224, 224)
CLASS_NAMES          = ["neutral", "drowsy", "distracted"]
PHONE_STREAM_URL     = "http://192.168.1.175:8080/video" # change it to your IP Webcam URL (check the app for the correct IP and port) 
CONFIDENCE_THRESHOLD = 0.5
FLIP_CAMERA          = False

# Display window size — adjust to your screen
DISPLAY_WIDTH  = 720
DISPLAY_HEIGHT = 480

#  KEY BINDINGS  (press while window is focused to log label)
#  1 = neutral   2 = drowsy   3 = distracted   Q = quit
LABEL_KEYS = {
    ord('1'): 0,   # neutral
    ord('2'): 1,   # drowsy
    ord('3'): 2,   # distracted
}

#  LOAD MODEL
print("=" * 60)
print("   📱 KERAS LIVE PHONE CAMERA — DRIVER MONITORING")
print("=" * 60)

print(f"\n[1/3] Loading model from '{KERAS_MODEL_PATH}'...")
if not os.path.exists(KERAS_MODEL_PATH):
    print(f"❌ Model not found: '{KERAS_MODEL_PATH}'")
    exit(1)

model = tf.keras.models.load_model(KERAS_MODEL_PATH)
num_classes = model.output_shape[-1]
print(f"✅ Model loaded! Classes: {CLASS_NAMES}")

#  CONNECT TO PHONE
print(f"\n[2/3] Connecting to {PHONE_STREAM_URL}...")
cap = cv2.VideoCapture(PHONE_STREAM_URL)
if not cap.isOpened():
    print(f"❌ Could not connect. Check that IP Webcam is running and the IP is correct.")
    exit(1)
print("✅ Connected!\n")

#  METRICS TRACKER
all_true_labels = []
all_pred_labels = []

def print_live_metrics():
    if len(all_true_labels) == 0:
        print("   (No labeled samples yet — press 1/2/3 to log ground truth)")
        return

    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)
    n      = len(y_true)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  📊 LIVE METRICS  (n={n} labeled frames)")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"{'─'*50}")

def print_final_metrics():
    if len(all_true_labels) == 0:
        print("\n⚠️  No labeled samples were recorded.")
        print("   Tip: Press 1, 2, or 3 next time to log the true class per frame.")
        return

    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)
    n      = len(y_true)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"   ✅ FINAL EVALUATION RESULTS  (n={n} labeled frames)")
    print(f"{'=' * 60}")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec*100:.2f}%")
    print(f"   Recall    : {rec*100:.2f}%")
    print(f"   F1-Score  : {f1*100:.2f}%")
    print(f"\n   📋 PER-CLASS REPORT:")
    print("─" * 50)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print(f"   🔲 CONFUSION MATRIX  (rows=true, cols=predicted)")
    print("─" * 50)
    cm = confusion_matrix(y_true, y_pred)

    header = f"{'':>13}" + "".join(f"{c:>13}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{CLASS_NAMES[i]:>13}" + "".join(f"{v:>13}" for v in row)
        print(row_str)

    print(f"\n{'=' * 60}")
    if acc >= 0.90:
        print(f"   🎉 Looking solid at {acc*100:.1f}%!")
    elif acc >= 0.70:
        print(f"   ⚠️  {acc*100:.1f}% — might need more training data.")
    else:
        print(f"   ❌ {acc*100:.1f}% — consider retraining.")
    print(f"{'=' * 60}\n")

#  DRAW OVERLAY
COLORS = {
    0: (50, 205, 50),    # neutral    — green
    1: (0, 165, 255),    # drowsy     — orange
    2: (0, 60, 220),     # distracted — red
}

def draw_overlay(frame, class_name, confidence, class_idx, all_probs, last_true_label):
    h, w = frame.shape[:2]
    color = COLORS.get(class_idx, (200, 200, 200))

    # Top banner 
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (w, 60), color, 2)
    cv2.putText(frame, class_name.upper(), (12, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence*100:.1f}% confidence", (12, 57),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    # Sample counter top-right 
    n = len(all_true_labels)
    cv2.putText(frame, f"Logged: {n}", (w - 130, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    if last_true_label is not None:
        true_name = CLASS_NAMES[last_true_label]
        cv2.putText(frame, f"True: {true_name}", (w - 130, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    COLORS.get(last_true_label, (200,200,200)), 1, cv2.LINE_AA)

    #  Confidence bars
    bar_x = 12
    bar_y_start = h - (num_classes * 32 + 25)
    panel_w = 240

    cv2.rectangle(frame,
                  (bar_x - 6, bar_y_start - 8),
                  (bar_x + panel_w + 6, bar_y_start + num_classes * 32 + 8),
                  (20, 20, 20), -1)

    for i, (cname, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
        by    = bar_y_start + i * 32
        blen  = int(prob * panel_w)
        c     = COLORS.get(i, (200, 200, 200))

        cv2.rectangle(frame, (bar_x, by + 14), (bar_x + panel_w, by + 26), (50,50,50), -1)
        if blen > 0:
            cv2.rectangle(frame, (bar_x, by + 14), (bar_x + blen, by + 26), c, -1)
        cv2.putText(frame, f"{cname}: {prob*100:.1f}%",
                    (bar_x, by + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    c if i == class_idx else (200, 200, 200),
                    2 if i == class_idx else 1, cv2.LINE_AA)

    # Key hint at bottom
    hint = "1=neutral  2=drowsy  3=distracted  Q=quit+metrics"
    cv2.putText(frame, hint, (bar_x, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    return frame

#  [3/3] MAIN LOOP
print("[3/3] Running! Controls in the camera window:")
print("      1 = neutral   2 = drowsy   3 = distracted")
print("      Q = quit and print final metrics\n")

frame_count  = 0
skip_frames  = 3
last_probs   = np.ones(num_classes) / num_classes
last_class   = 0
last_conf    = 0.0
last_true    = None
log_interval = 30   # Print running metrics to terminal every N labeled frames

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.open(PHONE_STREAM_URL)
        continue

    if FLIP_CAMERA:
        frame = cv2.flip(frame, 1)

    frame_count += 1

    # Inference 
    if frame_count % skip_frames == 0:
        img = cv2.resize(frame, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        probs      = model.predict(img, verbose=0)[0]
        last_class = int(np.argmax(probs))
        last_conf  = float(probs[last_class])
        last_probs = probs

    # Overlay + display
    display_name = CLASS_NAMES[last_class]
    if last_conf < CONFIDENCE_THRESHOLD:
        display_name = f"? {CLASS_NAMES[last_class]}"

    frame_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    frame_display = draw_overlay(frame_display, display_name, last_conf,
                                 last_class, last_probs, last_true)

    cv2.imshow("Driver Monitor — 1/2/3 to log | Q to quit", frame_display)

    # Key handling
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n👋 Quitting...")
        break

    elif key in LABEL_KEYS:
        true_label = LABEL_KEYS[key]
        last_true  = true_label
        all_true_labels.append(true_label)
        all_pred_labels.append(last_class)

        match = "✅" if true_label == last_class else "❌"
        print(f"  {match}  True: {CLASS_NAMES[true_label]:>12}  |  "
              f"Pred: {CLASS_NAMES[last_class]:>12}  |  "
              f"Conf: {last_conf*100:.1f}%  |  "
              f"Total logged: {len(all_true_labels)}")

        if len(all_true_labels) % log_interval == 0:
            print_live_metrics()

cap.release()
cv2.destroyAllWindows()
print_final_metrics()