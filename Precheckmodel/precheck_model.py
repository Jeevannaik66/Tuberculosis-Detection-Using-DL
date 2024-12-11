from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load preprocessed dataset (from Step 1)
X_train = np.load(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\X_train.npy')
X_test = np.load(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\X_test.npy')
y_train = np.load(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\y_train.npy')
y_test = np.load(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\y_test.npy')

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: X-ray vs Non-X-ray
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model to a file
model.save(r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Precheckmodel\precheck_model.h5')

print("Precheck model training complete and saved.")
