import numpy as np
import tensorflow as tf

# ------------------------------------------------
# STEP 1 — Load the TFLite model
# ------------------------------------------------
interpreter = tf.lite.Interpreter(model_path='assets/driver_monitor_tcn.tflite')
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded!")
print(f"Input shape:  {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# ------------------------------------------------
# STEP 2 — Label mapping (from your Colab training)
# ------------------------------------------------
label_map = {0: 'distracted', 1: 'drowsy', 2: 'focus'}

# ------------------------------------------------
# STEP 3 — Helper to run prediction
# ------------------------------------------------
def predict(sequence):
    # sequence shape must be (1, 30, 3)
    input_data = np.array([sequence], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output[0])
    confidence = output[0][predicted_index] * 100
    return label_map[predicted_index], confidence, output[0]

# ------------------------------------------------
# STEP 4 — Simulate test sequences
# ------------------------------------------------
print("\n--- Simulated Tests ---")

# Simulate FOCUS: EAR high, MAR low, SHA near 0
focus_sequence = [[0.35, 0.30, 0.5]] * 30
label, conf, raw = predict(focus_sequence)
print(f"Focus test     → Predicted: {label} ({conf:.1f}%)")
print(f"  Raw output: distracted={raw[0]:.3f}, drowsy={raw[1]:.3f}, focus={raw[2]:.3f}")

# Simulate DROWSY: EAR low, MAR high, SHA slight
drowsy_sequence = [[0.18, 0.65, 3.0]] * 30
label, conf, raw = predict(drowsy_sequence)
print(f"Drowsy test    → Predicted: {label} ({conf:.1f}%)")
print(f"  Raw output: distracted={raw[0]:.3f}, drowsy={raw[1]:.3f}, focus={raw[2]:.3f}")

# Simulate DISTRACTED: EAR normal, MAR normal, SHA high
distracted_sequence = [[0.35, 0.30, 20.0]] * 30
label, conf, raw = predict(distracted_sequence)
print(f"Distracted test → Predicted: {label} ({conf:.1f}%)")
print(f"  Raw output: distracted={raw[0]:.3f}, drowsy={raw[1]:.3f}, focus={raw[2]:.3f}")