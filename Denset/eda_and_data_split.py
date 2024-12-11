# eda_and_data_split.py

# Import required libraries
import os
import shutil
import random
import matplotlib.pyplot as plt

# Define paths
DATA_DIR = "C:/Users/Jeevan/Documents/Major Project/TB_Chest_Radiography_Database/Train_dataset"
NORMAL_DIR = os.path.join(DATA_DIR, "Normal")
TB_DIR = os.path.join(DATA_DIR, "Tuberculosis")
OUTPUT_DIR = "C:/Users/Jeevan/Documents/Major Project/TB_Chest_Radiography_Database/Processed"

# Check dataset information
print(f"There are {len(os.listdir(NORMAL_DIR))} images of Normal.")
print(f"There are {len(os.listdir(TB_DIR))} images of Tuberculosis.")

# Function to create train, validation, and test directories
def create_dirs(base_dir):
    os.makedirs(os.path.join(base_dir, "training/Normal"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "training/Tuberculosis"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "validation/Normal"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "validation/Tuberculosis"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "testing/Normal"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "testing/Tuberculosis"), exist_ok=True)
    print("Directories created successfully.")

# Create directories for processed dataset
create_dirs(OUTPUT_DIR)

# Function to split data into train, validation, and test sets
def split_data(SOURCE, TRAINING, VALIDATION, TESTING, SPLIT_TRAIN=0.8, SPLIT_VAL=0.1):
    files = [f for f in os.listdir(SOURCE) if os.path.isfile(os.path.join(SOURCE, f))]
    random.shuffle(files)

    train_size = int(len(files) * SPLIT_TRAIN)
    val_size = int(len(files) * SPLIT_VAL)

    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(SOURCE, file), TRAINING)
    for file in val_files:
        shutil.copy(os.path.join(SOURCE, file), VALIDATION)
    for file in test_files:
        shutil.copy(os.path.join(SOURCE, file), TESTING)

    print(f"Data split complete. {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

# Split Normal images
split_data(
    NORMAL_DIR,
    os.path.join(OUTPUT_DIR, "training/Normal"),
    os.path.join(OUTPUT_DIR, "validation/Normal"),
    os.path.join(OUTPUT_DIR, "testing/Normal"),
)

# Split Tuberculosis images
split_data(
    TB_DIR,
    os.path.join(OUTPUT_DIR, "training/Tuberculosis"),
    os.path.join(OUTPUT_DIR, "validation/Tuberculosis"),
    os.path.join(OUTPUT_DIR, "testing/Tuberculosis"),
)

# Visualize sample images from training dataset
def display_sample_images(category_dir, category_label):
    sample_files = random.sample(os.listdir(category_dir), 4)
    plt.figure(figsize=(10, 5))
    for idx, file in enumerate(sample_files):
        img = plt.imread(os.path.join(category_dir, file))
        plt.subplot(1, 4, idx + 1)
        plt.imshow(img, cmap="gray")
        plt.title(category_label)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Display sample images from training sets
print("Displaying Normal samples:")
display_sample_images(os.path.join(OUTPUT_DIR, "training/Normal"), "Normal")

print("Displaying Tuberculosis samples:")
display_sample_images(os.path.join(OUTPUT_DIR, "training/Tuberculosis"), "Tuberculosis")
