import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

# Load the pre-trained model (DenseNet model for TB classification)
model = load_model(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\model\tb_densenet_model.keras')

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to [0, 1]
    return img, img_array

# Function to compute Grad-CAM heatmap
def generate_gradcam_heatmap(model, img_array, layer_name="conv5_block16_2_conv"):
    """Generate Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, model.get_layer(layer_name).output])

    # Ensure the input image is in TensorFlow format
    img_tensor = tf.convert_to_tensor(img_array)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        # Get model predictions and output of the target layer
        predictions, conv_output = grad_model(img_tensor)
        # Assuming we're interested in the class with the highest prediction
        class_idx = np.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    # Compute the gradient of the class output with respect to the feature map
    grads = tape.gradient(class_output, conv_output)

    # Global average pooling to get the pooled gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Convert the pooled gradients and the convolutional layer output to numpy arrays
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()

    # Multiply the gradients with the output feature map to get a weighted feature map
    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    # Generate heatmap by averaging the weighted feature map
    heatmap = np.mean(conv_output, axis=-1)  # Average over the channels
    
    # Normalize the heatmap to the range [0, 1]
    heatmap = np.maximum(heatmap, 0)  # Remove negative values
    heatmap /= np.max(heatmap)  # Normalize the heatmap

    return heatmap

# Function to overlay the heatmap on the original image
def overlay_heatmap(heatmap, img_path, alpha=0.4):
    """Overlay heatmap on the original image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize the heatmap to the size of the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert the heatmap to a color map (using Jet colormap)
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to [0, 255]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply Jet colormap
    
    # Superimpose the heatmap onto the image
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    
    return superimposed_img

# Main execution function for image processing
def process_image_and_generate_heatmap(img_path, model_path):
    """Process the uploaded image and generate heatmap."""
    # Load the image and preprocess it
    img, img_array = load_and_preprocess_image(img_path)

    # Load the pre-trained model using the provided model path
    model = load_model(model_path)  # The model path is now passed as an argument

    # Generate the heatmap
    heatmap = generate_gradcam_heatmap(model, img_array)

    # Overlay the heatmap on the original image
    superimposed_img = overlay_heatmap(heatmap, img_path)

    return img, superimposed_img

