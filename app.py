import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from keras.preprocessing.image import load_img, img_to_array
import cv2
import gc

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set base directory for models and uploads
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# DON'T load models at startup - declare as None
tb_classification_model = None
tb_densenet_model = None
precheck_model = None

IMG_SIZE = (224, 224)

# Function to load precheck model only when needed
def load_precheck_model():
    global precheck_model
    if precheck_model is None:
        precheck_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'precheck_model.h5'))

# Function to load classification model only when needed
def load_classification_model():
    global tb_classification_model
    if tb_classification_model is None:
        tb_classification_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'tb_classification_model.h5'))

# Function to load densenet model only when needed
def load_densenet_model():
    global tb_densenet_model
    if tb_densenet_model is None:
        tb_densenet_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'tb_densenet_model.keras'))

# Function to check if image is a valid chest X-ray
def is_chest_xray(image_path):
    # Load precheck model only when checking image type
    load_precheck_model()
    
    img = cv2.imread(image_path)
    if img is None:
        return False
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = precheck_model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    return class_index == 1

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            flash('File type not supported')
            return redirect(request.url)
        try:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
            if not is_chest_xray(image_path):
                flash('Uploaded image is not a valid chest X-ray.')
                return redirect(request.url)
            return render_template('upload.html', image_file=file.filename, show_predict_button=True)
        except Exception as e:
            flash(f'Error saving file: {str(e)}')
            return redirect(request.url)
    return render_template('upload.html')

# Predict route - Load classification model only when predicting
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.form['image_file']
    image_path = os.path.join(UPLOAD_FOLDER, image_file)
    try:
        # Load classification model only when user clicks predict
        load_classification_model()
        
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        tb_classification_prediction = tb_classification_model.predict(img_array)
        tb_predicted_class = np.argmax(tb_classification_prediction, axis=1)[0]
        tb_accuracy = np.max(tb_classification_prediction) * 100
        if tb_predicted_class == 0:
            result = "NO"
            heatmap_url = None
        else:
            result = "YES"
            heatmap_url = url_for('generate_heatmap', image_file=image_file)
        return render_template('upload.html', result=result, accuracy=tb_accuracy, image_file=image_file)
    except Exception as e:
        flash(f'Error during prediction: {str(e)}')
        return redirect(request.url)

# Generate heatmap route - Load densenet model only when generating heatmap
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    try:
        image_file = request.form['image_file']
        img_path = os.path.join(UPLOAD_FOLDER, image_file)
        
        # Load densenet model only when user clicks heatmap button
        load_densenet_model()
        
        _, img_array = load_and_preprocess_image(img_path, target_size=IMG_SIZE)
        heatmap = generate_gradcam_heatmap(tb_densenet_model, img_array)
        overlayed_img = overlay_heatmap(heatmap, img_path)
        heatmap_path = os.path.join(UPLOAD_FOLDER, f'heatmap_{image_file}')
        cv2.imwrite(heatmap_path, overlayed_img)
        heatmap_url = url_for('static', filename=f'uploads/heatmap_{image_file}')
        return render_template('upload.html', heatmap_url=heatmap_url, image_file=image_file)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Image preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array

# Grad-CAM heatmap generation (fixed indexing)
def generate_gradcam_heatmap(model, img_array, layer_name="conv5_block16_2_conv"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.output, model.get_layer(layer_name).output]
    )
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions, conv_output = grad_model(img_tensor)
        # ✅ Convert tensor to int before indexing
        class_idx = int(tf.argmax(predictions[0]))
        class_output = predictions[0][class_idx]  # scalar
    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap /= max_val
    return heatmap.numpy()

# Overlay heatmap
def overlay_heatmap(heatmap, original_img_path, alpha=0.4):
    img = cv2.imread(original_img_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlayed_img

# About and prevention pages
@app.route('/about')
def about_us():
    return render_template('about.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/health')
def health_check():
    return 'OK', 200

if __name__ == "__main__":
    # Initial cleanup
    gc.collect()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)