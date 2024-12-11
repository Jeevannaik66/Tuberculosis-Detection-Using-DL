import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from keras.preprocessing.image import load_img, img_to_array
import cv2
from generate_heatmap import generate_gradcam_heatmap, overlay_heatmap  # Import heatmap functions

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Load the pre-trained models
tb_classification_model_path = os.path.join('model', 'tb_classification_model.h5')
tb_densenet_model_path = os.path.join('model', 'tb_densenet_model.keras')
precheck_model_path = os.path.join('model', 'precheck_model.h5')

tb_classification_model = tf.keras.models.load_model(tb_classification_model_path)
tb_densenet_model = tf.keras.models.load_model(tb_densenet_model_path)
precheck_model = tf.keras.models.load_model(precheck_model_path)

# Define the image size your model expects
IMG_SIZE = (224, 224)  # Adjust according to your model's input size

# Ensure uploads directory exists within static
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the image is a valid chest X-ray
def is_chest_xray(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (128, 128))  # Resize to match input shape of precheck model
    img = img / 255.0  # Normalize if required by the model
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict with the precheck model
    prediction = precheck_model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]  # 0 for non-xray, 1 for chest-xray

    return class_index == 1  # Returns True if chest-xray


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if a file is provided in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # Check if the user actually selected a file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Check if the file is an image
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            flash('File type not supported. Please upload a PNG or JPG image.')
            return redirect(request.url)

        try:
            # Save the uploaded file to the static/uploads directory
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

                       # Check if the image is a valid chest X-ray
            if not is_chest_xray(image_path):
                flash('Uploaded image is not a valid chest X-ray. Please upload a valid X-ray image.')
                return redirect(request.url)


            # Render the upload page with the uploaded image
            return render_template('upload.html', image_file=file.filename, show_predict_button=True)

        except Exception as e:
            flash(f'An error occurred while saving the file: {str(e)}')
            return redirect(request.url)

    # If GET request, simply render the upload page
    return render_template('upload.html')

# Route to make predictions using the first model (tb_classification_model.h5)
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.form['image_file']
    image_path = os.path.join(UPLOAD_FOLDER, image_file)

    try:
        # Prepare the image for prediction
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Step 1: Make prediction using TB classification model (tb_classification_model.h5)
        tb_classification_prediction = tb_classification_model.predict(img_array)
        tb_predicted_class = np.argmax(tb_classification_prediction, axis=1)[0]
        tb_accuracy = np.max(tb_classification_prediction) * 100  # Get accuracy as a percentage

        # Step 2: Check if TB is detected
        if tb_predicted_class == 0:
            result = "NO"
            heatmap_url = None  # No heatmap if no TB detected
        else:
            result = "YES"
            # Step 3: If TB is detected, pass the image to tb_densenet_model.keras for heatmap generation
            heatmap_url = url_for('generate_heatmap', image_file=image_file)  # URL to generate heatmap

        # Step 4: Render result in the template, including the uploaded image
        return render_template('upload.html', result=result, accuracy=tb_accuracy, image_file=image_file)

    except Exception as e:
        flash(f'An error occurred during prediction: {str(e)}')
        return redirect(request.url)

# Route to generate heatmap using the second model (tb_densenet_model.keras)
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    try:
        # Get the uploaded image file path from the form data
        image_file = request.form['image_file']
        img_path = os.path.join(UPLOAD_FOLDER, image_file)  # Full path to the uploaded image

        # Load and preprocess the image
        _, img_array = load_and_preprocess_image(img_path, target_size=IMG_SIZE)

        # Generate the Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(tb_densenet_model, img_array)

        # Overlay the heatmap on the original image
        overlayed_img = overlay_heatmap(heatmap, img_path)

        # Save the heatmap image to the same directory
        heatmap_path = os.path.join(UPLOAD_FOLDER, f'heatmap_{image_file}')
        cv2.imwrite(heatmap_path, overlayed_img)

        # Create the heatmap URL for displaying in the web app
        heatmap_url = url_for('static', filename=f'uploads/heatmap_{image_file}')

        # Render the template with the heatmap URL
        return render_template('upload.html', heatmap_url=heatmap_url, image_file=image_file)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction.
    Args:
    - image_path: str, path to the image file.
    - target_size: tuple, size to resize the image (default is 224x224).
    
    Returns:
    - img: The processed image (with channel values between 0 and 1).
    - img_array: The image converted to a numpy array with batch dimension.
    """
    # Load the image and resize it to the target size
    img = load_img(image_path, target_size=target_size)
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    
    # Expand dimensions to create a batch (model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image (assuming the model expects pixel values between 0 and 1)
    img_array /= 255.0
    
    return img, img_array

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
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)

    return heatmap

def overlay_heatmap(heatmap, original_img_path, alpha=0.4):
    """Overlay the heatmap on the original image."""
    # Load the original image
    img = cv2.imread(original_img_path)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_rgb, alpha, 0)

    return overlayed_img

@app.route('/about')
def about_us():
    return render_template('about.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

if __name__ == '__main__':
    app.run(debug=True)
