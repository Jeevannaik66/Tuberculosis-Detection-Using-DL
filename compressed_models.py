import tensorflow as tf
import os

def compress_model(original_model_path, compressed_model_path):
    """Compress model by removing unnecessary data"""
    try:
        print(f"🔄 Compressing: {original_model_path}")
        
        # Load original model
        model = tf.keras.models.load_model(original_model_path)
        
        # Save as compressed H5 (removes optimizer state, etc.)
        model.save(compressed_model_path, save_format='h5')
        
        # Get file sizes
        original_size = os.path.getsize(original_model_path) / (1024 * 1024)  # MB
        compressed_size = os.path.getsize(compressed_model_path) / (1024 * 1024)  # MB
        
        print(f"✅ Compressed: {original_model_path}")
        print(f"   Original: {original_size:.1f}MB → Compressed: {compressed_size:.1f}MB")
        print(f"   Reduction: {((original_size - compressed_size) / original_size) * 100:.1f}%")
        print()
        
    except Exception as e:
        print(f"❌ Error compressing {original_model_path}: {e}")

# Compress your models
if __name__ == "__main__":
    MODEL_DIR = 'model'  # Add this line
    
    models_to_compress = [
        ('tb_classification_model.h5', 'tb_classification_model_compressed.h5'),
        ('tb_densenet_model.keras', 'tb_densenet_model_compressed.h5'),
    ]
    
    print("🚀 Starting model compression...")
    
    for original, compressed in models_to_compress:
        original_path = os.path.join(MODEL_DIR, original)
        compressed_path = os.path.join(MODEL_DIR, compressed)
        
        # Check if original file exists
        if not os.path.exists(original_path):
            print(f"❌ File not found: {original_path}")
            continue
            
        compress_model(original_path, compressed_path)
    
    print("🎉 Compression completed!")