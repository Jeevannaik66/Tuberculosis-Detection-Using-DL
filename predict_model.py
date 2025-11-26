import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'tb_classification_model.h5')
model = load_model(model_path)

# Define constants
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Adjust according to your model's input size
TEST_IMAGES_DIR = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\test_images'  # Path to test images

# Function to load and preprocess images
def load_and_preprocess_images(directory):
    # Get all image paths
    image_paths = glob.glob(os.path.join(directory, '*.*'))  # Adjust this pattern for specific formats like *.jpg, *.png etc.
    images = []
    
    for img_path in image_paths:
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='rgb')  # Load image
        img_array = img_to_array(img)  # Convert to array
        img_array = img_array / 255.0  # Normalize the image
        images.append(img_array)
    
    return np.array(images)  # Convert list to NumPy array

# Function to visualize predictions
def visualize_predictions(images):
    predictions = model.predict(images)  # Get predictions
    predicted_classes = np.argmax(predictions, axis=-1)  # Convert probabilities to class labels
    return predicted_classes

# Main execution
if __name__ == "__main__":
    print(f"Loading images from: {TEST_IMAGES_DIR}")
    images = load_and_preprocess_images(TEST_IMAGES_DIR)
    
    # Check if any images were loaded
    if images.size == 0:
        print("No images found in the specified directory.")
    else:
        predicted_classes = visualize_predictions(images)
        for i, predicted_class in enumerate(predicted_classes):
            print(f"Image {i + 1}: Predicted class: {predicted_class}")
