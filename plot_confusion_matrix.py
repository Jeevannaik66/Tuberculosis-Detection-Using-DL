import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set up directory paths
BASE_DIR = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database'
MODEL_PATH = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\model\tb_classification_model.h5'

# Define image size and batch size
image_size = (224, 224)
batch_size = 32

# Load the trained model
model = load_model(MODEL_PATH)
print(f"Model expects input shape: {model.input_shape}")

# Set up test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load the test dataset
test_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Processed/testing'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for consistent predictions
)

# Predict the classes
print("Generating predictions...")
predictions = model.predict(test_generator, verbose=1)

# Convert predictions and true labels to their class indices
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot confusion matrix with labels
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)
plt.tight_layout()
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up directory paths
BASE_DIR = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database'
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'tb_classification_model.h5')

# Load the trained model
model = load_model(MODEL_PATH)

# Set up image size and batch size
image_size = (224, 224)
batch_size = 32

# Define ImageDataGenerator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Create test generator (assuming testing data is in the 'testing' folder)
test_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Processed', 'testing'),  # Ensure this path is correct
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification, 'categorical' is used
    shuffle=False  # Important: Do not shuffle when evaluating, as we need consistent labels for prediction comparison
)

# Generate predictions from the model
y_pred = np.argmax(model.predict(test_generator), axis=-1)
y_true = test_generator.classes  # True labels from the generator

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Classification report (precision, recall, f1-score)
report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())

# Print the classification report
print(report)

# Plot confusion matrix with annotations
def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(cm, labels=test_generator.class_indices.keys())

# Optionally, visualize the confusion matrix as percentages
def plot_percentage_confusion_matrix(cm, labels):
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title('Confusion Matrix (Percentage)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plot percentage confusion matrix
plot_percentage_confusion_matrix(cm, labels=test_generator.class_indices.keys())
