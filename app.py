import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
try:
    from keras.preprocessing.image import load_img, img_to_array
except ImportError:
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import gc

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# Set base directory for models and uploads
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# DON'T load models at startup - declare as None
tb_classification_model = None
tb_densenet_model = None
precheck_model = None

IMG_SIZE = (224, 224)

# Function to load precheck model only when needed
def load_precheck_model():
    global precheck_model
    if precheck_model is None:
        model_path = os.path.join(MODEL_DIR, 'precheck_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Pre-check model is not available. Please contact the administrator.")
        precheck_model = tf.keras.models.load_model(model_path)

def load_classification_model():
    global tb_classification_model
    if tb_classification_model is None:
        model_path = os.path.join(MODEL_DIR, 'tb_classification_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Classification model is not available. Please contact the administrator.")
        tb_classification_model = tf.keras.models.load_model(model_path)

def load_densenet_model():
    global tb_densenet_model
    if tb_densenet_model is None:
        model_path = os.path.join(MODEL_DIR, 'tb_densenet_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Heatmap model is not available. Please contact the administrator.")
        tb_densenet_model = tf.keras.models.load_model(model_path)

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
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    if request.method == 'POST':
        if 'file' not in request.files:
            if is_ajax:
                return jsonify(error='No file part'), 400
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            if is_ajax:
                return jsonify(error='No selected file'), 400
            flash('No selected file')
            return redirect(request.url)
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if is_ajax:
                return jsonify(error='File type not supported. Use JPG, JPEG, or PNG.'), 400
            flash('File type not supported')
            return redirect(request.url)
        try:
            safe_name = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, safe_name)
            file.save(image_path)
            if not is_chest_xray(image_path):
                if is_ajax:
                    return jsonify(error='Uploaded image is not a valid chest X-ray.'), 400
                flash('Uploaded image is not a valid chest X-ray.')
                return redirect(request.url)
            if is_ajax:
                image_url = url_for('static', filename=f'uploads/{safe_name}')
                return jsonify(image_file=safe_name, image_url=image_url)
            return render_template('upload.html', image_file=safe_name, show_predict_button=True)
        except FileNotFoundError as e:
            if is_ajax:
                return jsonify(error=str(e)), 503
            flash(str(e))
            return redirect(request.url)
        except Exception:
            if is_ajax:
                return jsonify(error='Something went wrong while uploading. Please try again.'), 500
            flash('Something went wrong while uploading. Please try again.')
            return redirect(request.url)
    return render_template('upload.html')

# Predict route - Load classification model only when predicting
@app.route('/predict', methods=['POST'])
def predict():
    image_file = secure_filename(request.form['image_file'])
    image_path = os.path.join(UPLOAD_FOLDER, image_file)
    try:
        load_classification_model()
        _, img_array = load_and_preprocess_image(image_path, target_size=IMG_SIZE)
        tb_classification_prediction = tb_classification_model.predict(img_array)
        tb_predicted_class = np.argmax(tb_classification_prediction, axis=1)[0]
        tb_accuracy = round(float(np.max(tb_classification_prediction) * 100), 2)
        result = "NO" if tb_predicted_class == 0 else "YES"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(result=result, accuracy=tb_accuracy, image_file=image_file)
        return render_template('upload.html', result=result, accuracy=tb_accuracy, image_file=image_file)
    except FileNotFoundError as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(error=str(e)), 503
        flash(str(e))
        return redirect(url_for('upload'))
    except Exception:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(error='Something went wrong during prediction. Please try again.'), 500
        flash('Something went wrong during prediction. Please try again.')
        return redirect(url_for('upload'))

# Generate heatmap route - Load densenet model only when generating heatmap
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    try:
        image_file = secure_filename(request.form['image_file'])
        img_path = os.path.join(UPLOAD_FOLDER, image_file)

        load_densenet_model()

        _, img_array = load_and_preprocess_image(img_path, target_size=IMG_SIZE)
        heatmap = generate_gradcam_heatmap(tb_densenet_model, img_array)
        overlayed_img = overlay_heatmap(heatmap, img_path)
        heatmap_path = os.path.join(UPLOAD_FOLDER, f'heatmap_{image_file}')
        cv2.imwrite(heatmap_path, overlayed_img)
        heatmap_url = url_for('static', filename=f'uploads/heatmap_{image_file}')
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(heatmap_url=heatmap_url, image_file=image_file)
        return render_template('upload.html', heatmap_url=heatmap_url, image_file=image_file, result="YES")
    except FileNotFoundError as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(error=str(e)), 503
        flash(str(e))
        return redirect(url_for('upload'))
    except Exception:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(error='Something went wrong while generating heatmap. Please try again.'), 500
        flash('Something went wrong while generating heatmap. Please try again.')
        return redirect(url_for('upload'))

# Image preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img, img_array

# Grad-CAM heatmap generation
def generate_gradcam_heatmap(model, img_array, layer_name="conv5_block16_2_conv"):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_output, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[0][class_idx]
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