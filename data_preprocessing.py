import os
from PIL import Image
import numpy as np

# Define absolute directories
normal_directory = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Normal'
tuberculosis_directory = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Tuberculosis'
processed_directory = r'C:\Users\Jeevan\Documents\Major Project\TB_Chest_Radiography_Database\Processed'

# Create the processed directory if it doesn't exist
os.makedirs(processed_directory, exist_ok=True)

# Set desired image size
desired_size = (128, 128)  # Resize images to 128x128 pixels

def process_images(directory, label):
    # Print the directory being processed
    print(f"Processing images in: {directory}")
    
    """
    Process images in the given directory, resize them, normalize pixel values,
    and save them with a specified label.
    """
    # Loop through all files in the specified directory
    for image_name in os.listdir(directory):
        # Check for valid image file extensions
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(directory, image_name)  # Full image path
            
            try:
                # Open the image
                with Image.open(image_path) as img:
                    # Resize the image to the desired size
                    img = img.resize(desired_size)

                    # Normalize the image
                    img_array = np.array(img) / 255.0  # Scale pixel values to [0, 1]

                    # Convert back to image format after normalization
                    img_normalized = Image.fromarray((img_array * 255).astype('uint8'))

                    # Construct the new file name with label
                    save_path = os.path.join(processed_directory, f"{label}_{image_name}")
                    
                    # Save the processed image
                    img_normalized.save(save_path)
                    print(f"Processed and saved: {save_path}")

            except Exception as e:
                print(f"Error processing {image_name}: {e}")

# Process Normal images
process_images(normal_directory, 'normal')

# Process Tuberculosis images
process_images(tuberculosis_directory, 'tuberculosis')

print(f"All processed images are saved in the '{processed_directory}' directory.")
