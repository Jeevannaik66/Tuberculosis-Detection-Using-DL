import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

# Define the paths for your dataset
dataset_dir = r"C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\dataset"
chest_xray_dir = os.path.join(dataset_dir, "chest_xrays")
non_xray_dir = os.path.join(dataset_dir, "non_xrays")

# Define image size to resize the images to
image_size = (128, 128)  # Resize all images to 128x128

# Initialize lists for images and labels
images = []
labels = []

# Function to load and preprocess images
def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        # Check if the file is an image (you can check for extension if needed)
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)  # Resize to desired size
            img = img_to_array(img)  # Convert to a NumPy array
            images.append(img)
            labels.append(label)

# Load chest X-rays (label 1 for TB detection)
load_images_from_folder(chest_xray_dir, label=1)

# Load non-X-rays (label 0 for non-X-ray images)
load_images_from_folder(non_xray_dir, label=0)

# Convert images and labels to NumPy arrays
images = np.array(images, dtype="float32")
labels = np.array(labels)

# Normalize pixel values to the range [0, 1]
images /= 255.0

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# One-hot encode the labels (convert labels to categorical format)
y_train = to_categorical(y_train, 2)  # 2 classes: 0 (non-X-ray), 1 (X-ray)
y_test = to_categorical(y_test, 2)

# Save the preprocessed data to files for future use (optional)
np.save(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\X_train.npy', X_train)
np.save(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\X_test.npy', X_test)
np.save(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\y_train.npy', y_train)
np.save(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\y_test.npy', y_test)

print("Dataset preparation complete. Training and testing data saved.")
