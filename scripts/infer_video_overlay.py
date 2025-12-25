#!/usr/bin/env python3
"""
Segmentation overlay on a video.

- Loads a Keras model from `model.keras`.
- Reads `test.mp4` frame‑by‑frame.
- Runs the model on each frame (after resizing to the model’s input size).
- Overlays the predicted binary mask on the original frame with 50 % transparency.
- Writes the result to `video_overlay.mp4`.

Prerequisites:
    pip install tensorflow opencv-python
"""

import cv2
import numpy as np
import tensorflow as tf
import os


MODEL_LABEL = "model_mobilenet.py"

models_path = "../results/"
model_folder = models_path + MODEL_LABEL + '/'


# ----------------------------------------------------------------------
# 1️⃣  Load the Keras model
# ----------------------------------------------------------------------
MODEL_PATH = model_folder + "model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Cannot find {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Assume the model takes a single image input and outputs a single channel mask.
# Get the expected input size (height, width) from the model's first layer.
input_shape = model.input_shape  # e.g. (None, H, W, 3)
if len(input_shape) != 4:
    raise ValueError("Model input shape is not 4‑D (batch, height, width, channels)")

_, H, W, C = input_shape
if C not in (1, 3):
    raise ValueError("Model input must have 1 or 3 channels")

# ----------------------------------------------------------------------
# 2️⃣  Set up video reading and writing
# ----------------------------------------------------------------------
VIDEO_INPUT = "../data/test.mp4"
VIDEO_OUTPUT = model_folder + "video_overlay.mp4"

if not os.path.exists(VIDEO_INPUT):
    raise FileNotFoundError(f"Cannot find {VIDEO_INPUT}")

cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    raise RuntimeError("Failed to open input video")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # or 'XVID', 'H264', etc.
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (frame_width, frame_height))

# ----------------------------------------------------------------------
# 3️⃣  Helper: run inference on a single frame
# ----------------------------------------------------------------------
def predict_mask(frame: np.ndarray) -> np.ndarray:
    """
    Takes a BGR frame (H, W, 3), resizes to model input, runs inference,
    and returns a binary mask of the original frame size.
    """
    # Resize to model input size
    resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

    # Convert to RGB (Keras models are usually trained on RGB)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1] if the model expects float inputs
    resized_rgb = resized_rgb.astype(np.float32) / 255.0

    # Add batch dimension
    input_tensor = np.expand_dims(resized_rgb, axis=0)  # shape (1, H, W, 3)

    # Predict
    pred = model.predict(input_tensor, verbose=0)  # shape (1, H, W, 1) or (1, H, W)

    # Remove batch dimension
    pred = np.squeeze(pred, axis=0)

    # If the output is not already a single channel, take the first channel
    if pred.ndim == 3 and pred.shape[2] == 1:
        pred = pred[:, :, 0]

    # Threshold to get binary mask (you can adjust the threshold if needed)
    mask = (pred > 0.5).astype(np.uint8) * 255  # 255 for white mask

    # Resize mask back to original frame size
    mask_resized = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    return mask_resized

# ----------------------------------------------------------------------
# 4️⃣  Process each frame
# ----------------------------------------------------------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Predict mask for this frame
    mask = predict_mask(frame)  # shape (H, W), single channel

    # Create a colored mask (e.g. red) for overlay
    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, 2] = mask  # set the red channel; 0=black, 255=red

    # Overlay with 50% transparency
    overlay = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)

    # Write the frame to the output video
    out.write(overlay)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx} frames...")

print(f"Finished processing {frame_idx} frames.")
print(f"Output video saved to {VIDEO_OUTPUT}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()