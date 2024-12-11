import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Set up directory paths
BASE_DIR = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Train_dataset'

# Define image size and batch size (reduce size and batch size for less complexity)
image_size = (224, 224)  # Use 224x224 to save memory
batch_size = 8  # Use smaller batch size

# ImageDataGenerator for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 20% for validation

train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    subset='training'  # Specify that this is the training subset
)

validation_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    subset='validation'  # Specify that this is the validation subset
)

# Load DenseNet121 as base model without top layers (exclude fully connected layers)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of DenseNet
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers to prevent retraining during initial epochs
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks
checkpoint_path = "./tb_classification_densenet_model.keras"  # Save as .keras
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[checkpoint, early_stopping]
)

# Optionally, save the final model
model.save('./tb_classification_densenet_model_final.keras')
