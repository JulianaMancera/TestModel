import tensorflow as tf
import numpy as np

KERAS_MODEL_PATH = "model_name.keras"   
OUTPUT_PATH      = "dms_hybridnet.tflite"

print(f"[1/5] Loading Keras model from '{KERAS_MODEL_PATH}'...")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
model.summary()

# Get input shape
input_shape = model.input_shape
print(f"\n      Input shape: {input_shape}")
fixed_shape = [1] + [d if d is not None else 224 for d in list(input_shape[1:])]
print(f"      Fixed shape: {fixed_shape}")

print(f"\n[2/5] Freezing all layers...")
for layer in model.layers:
    layer.trainable = False

print(f"\n[3/5] Quick sanity check with Keras model...")
test_input = np.random.rand(*fixed_shape).astype(np.float32)
keras_output = model(test_input, training=False).numpy()
print(f"      Keras output: {keras_output}")
if np.any(np.isnan(keras_output)):
    print("⚠️  WARNING: Keras model itself outputs NaN — issue is in the model weights!")
    print("   Ask your partner to check if the model was saved correctly.")
else:
    print("✅ Keras model output is valid!")

print(f"\n[4/5] Converting to TFLite (FP32 — avoids NaN from FP16)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ✅ NO FP16 — use FP32 to avoid NaN numerical issues
# ✅ NO quantization — keep full precision
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
print("✅ FP32 conversion successful!")

print(f"\n[5/5] Saving and verifying '{OUTPUT_PATH}'...")
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
print(f"✅ Saved '{OUTPUT_PATH}' ({size_mb:.1f} MB)")

# Verify
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"\n📥 Input  shape: {input_details[0]['shape']}")
print(f"📤 Output shape: {output_details[0]['shape']}")

# Test with random input
test_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"🧪 Test output:  {output}")

if np.any(np.isnan(output)):
    print("\n❌ Still NaN! The issue is in the model weights — ask partner to re-save the .keras file.")
else:
    print(f"\n✅ Model is valid! Values sum to ~{output.sum():.3f} (should be ~1.0 for softmax)")
    print(f"   Copy '{OUTPUT_PATH}' to Flutter assets/ and run:")
    print("   flutter clean && flutter pub get && flutter run")