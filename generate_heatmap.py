import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

# -----------------------------
# Paths and model loading
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'tb_densenet_model.keras')

# Load model once at startup
model = load_model(MODEL_PATH)
model.trainable = False  # Ensure model is inference-only

# -----------------------------
# Preprocess image (resize + normalize)
# -----------------------------
def load_and_preprocess_image(img_path, target_size=(160, 160)):
    """
    Load image, resize for faster Grad-CAM, normalize to [0,1], return both PIL img and array.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img, img_array

# -----------------------------
# Generate Grad-CAM heatmap (optimized)
# -----------------------------
@tf.function  # Compiled for faster execution
def generate_gradcam_heatmap(img_array, layer_name="conv5_block16_2_conv"):
    """
    Returns normalized Grad-CAM heatmap for a given image array.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.output, model.get_layer(layer_name).output]
    )
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions, conv_output = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[0][class_idx]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.cond(max_val > 0, lambda: heatmap / max_val, lambda: heatmap)
    return heatmap.numpy()

# -----------------------------
# Overlay heatmap on original image
# -----------------------------
def overlay_heatmap(heatmap, img_path, alpha=0.4, output_size=(500, 500)):
    """
    Overlay Grad-CAM heatmap on original image.
    Resize output to a fixed width for faster mobile loading.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_rgb, alpha, 0)

    # Resize for mobile-friendly width
    ratio = output_size[0] / overlayed_img.shape[1]
    new_height = int(overlayed_img.shape[0] * ratio)
    overlayed_img = cv2.resize(overlayed_img, (output_size[0], new_height))

    return overlayed_img

# -----------------------------
# Main function
# -----------------------------
def process_image_and_generate_heatmap(img_path):
    """
    Fast pipeline: load image, generate heatmap, overlay heatmap.
    """
    _, img_array = load_and_preprocess_image(img_path)
    heatmap = generate_gradcam_heatmap(img_array)
    overlayed_img = overlay_heatmap(heatmap, img_path)
    return overlayed_img
