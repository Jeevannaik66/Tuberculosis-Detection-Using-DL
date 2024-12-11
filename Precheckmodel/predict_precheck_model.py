import cv2
import tensorflow as tf
import numpy as np

# Load the precheck model to verify if the image is a chest X-ray
try:
    precheck_model = tf.keras.models.load_model(
        r'C:/Users/Jeevan/Documents/Major Project/TB_Chest_Radiography_Database/Precheckmodel/precheck_model.h5'
    )
    print("Precheck model loaded successfully.")
except Exception as e:
    print(f"Error loading precheck model: {e}")

# Load the TB classification model
try:
    tb_model = tf.keras.models.load_model(
        r'C:/Users/Jeevan/Documents/Major Project/TB_Chest_Radiography_Database/model/tb_classification_model.h5'
    )
    print("TB classification model loaded successfully.")
except Exception as e:
    print(f"Error loading TB classification model: {e}")

# Path to the uploaded image
image_path = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\uploads\example_image.jpg'

# Function to check if the image is a chest X-ray
def is_chest_xray(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not load the image. Please check if the file exists.")
        return False
    
    # Resize image to match input shape of precheck model (assumed 128x128 for this example)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize if required by the model
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict with the precheck model
    prediction = precheck_model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]  # Assume 0 for non-xray, 1 for chest-xray

    return class_index == 1  # Returns True if chest-xray

# Function to predict TB if the image is a valid chest X-ray
def predict_tb(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not load the image. Please check if the file exists.")
        return "Error"

    # Resize image to model's expected input size
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize pixel values if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict TB
    prediction = tb_model.predict(img)
    return "TB Positive" if prediction[0][0] > 0.5 else "TB Negative"

# Execute the precheck and prediction
if is_chest_xray(image_path):
    print("Valid Chest X-ray detected. Proceeding with TB prediction.")
    result = predict_tb(image_path)
    if result != "Error":
        print("TB Prediction:", result)
else:
    print("Please upload a valid chest X-ray image.")
