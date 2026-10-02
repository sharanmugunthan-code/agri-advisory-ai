import tensorflow as tf
from tensorflow.keras.models import load_model

print("TensorFlow version:", tf.__version__)
print("Loading model...")

model = load_model(
    "agri_model.keras",
    compile=False
)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)