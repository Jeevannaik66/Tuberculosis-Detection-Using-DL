import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Set up directory paths
BASE_DIR = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database'

# Define batch size and image size
batch_size = 32
image_size = (224, 224)

# Use ImageDataGenerator for loading images in batches
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Split the data for training and validation
)

# Create train and validation generators
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Ensure this matches the output shape
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Ensure this matches the output shape
    subset='validation'
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(9, activation='softmax')  # Change this line for 9 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using the generators
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[early_stopping]
)

# Save the model
model.save(os.path.join(BASE_DIR, 'model', 'tb_classification_model.h5'))

# Evaluate on the validation dataset
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Generate classification report
y_pred = np.argmax(model.predict(validation_generator), axis=-1)
y_true = validation_generator.classes
print(classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys()))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize some predictions
def visualize_predictions(model, generator):
    class_labels = list(generator.class_indices.keys())
    images, labels = next(generator)
    predictions = np.argmax(model.predict(images), axis=-1)

    plt.figure(figsize=(12, 12))
    for i in range(9):  # Show 9 predictions
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Predicted: {class_labels[predictions[i]]}\nTrue: {class_labels[np.argmax(labels[i])]}")  # Display true label
        plt.axis('off')
    plt.show()

visualize_predictions(model, validation_generator)
