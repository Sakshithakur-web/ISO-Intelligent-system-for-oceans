# src/preprocessing.py
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

class CoralPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def resize_image(self, image):
        """Resize image to target dimensions"""
        return cv2.resize(image, self.target_size)
    
    def normalize_image(self, image):
        """Normalize pixel values to [0,1]"""
        return image.astype(np.float32) / 255.0
    
    def apply_noise_reduction(self, image):
        """Apply Gaussian blur for noise reduction"""
        return cv2.GaussianBlur(image, (3, 3), 0)
    
    def preprocess(self, image_path):
        """Complete preprocessing pipeline"""
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing steps
        img = self.resize_image(img)
        img = self.apply_noise_reduction(img)
        img = self.normalize_image(img)
        
        return img
    
    def load_dataset(self, data_dir):
        """Load and preprocess entire dataset"""
        images = []
        labels = []
        class_names = ['healthy_corals', 'bleached_corals']
        class_map = {name: idx for idx, name in enumerate(class_names)}
        
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
                
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        processed_img = self.preprocess(img_path)
                        images.append(processed_img)
                        labels.append(class_map[class_name])
                    except Exception as e:
                        print(f"Warning: Could not process {img_path}: {str(e)}")
                        continue
        
        if len(images) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        return np.array(images), np.array(labels), class_names

# Usage example
if __name__ == "__main__":
    preprocessor = CoralPreprocessor()
    X, y, class_names = preprocessor.load_dataset('data/raw/')
    print(f"Loaded {len(X)} images")
    print(f"Classes: {class_names}")