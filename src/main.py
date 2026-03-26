# main.py
import os
import sys
import cv2
import numpy as np

# Ensure src path is available regardless of current working directory
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from preprocessing import CoralPreprocessor
from enhancement import UnderwaterImageEnhancer
from classification import CoralClassifier

class CoralHealthMonitoringSystem:
    def __init__(self):
        self.preprocessor = CoralPreprocessor()
        self.enhancer = UnderwaterImageEnhancer()
        self.classifier = None
        
    def load_enhanced_data(self, data_path, split='train'):
        """Load enhanced data from pre-organized directories"""
        print(f"Loading {split} dataset from {data_path}...")
        images = []
        labels = []
        class_names = ['healthy_corals', 'bleached_corals']
        class_map = {name: idx for idx, name in enumerate(class_names)}
        
        split_dir = os.path.join(data_path, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Path not found: {split_dir}")
        
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
                
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Resize to standard size
                        img = cv2.resize(img, (224, 224))
                        # Normalize to [0, 1]
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                        labels.append(class_map[class_name])
                    except Exception as e:
                        print(f"Warning: Could not process {img_path}: {str(e)}")
                        continue
        
        if len(images) == 0:
            raise ValueError(f"No images found in {split_dir}")
        
        X = np.array(images)
        y = np.array(labels)
        print(f"Loaded {len(X)} {split} images from {len(class_names)} classes")
        
        return X, y, class_names

    def train_system(self, X_train, y_train, X_val, y_val, epochs=20):
        """Train the classification model with augmentation and optional fine-tuning"""
        print("Training classifier with data augmentation...")
        self.classifier = CoralClassifier(num_classes=len(np.unique(y_train)))

        # Stage 1: Data-augmented training on frozen base weights
        history = self.classifier.train_with_augmentation(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32)

        # Stage 2: Unfreeze top layers for fine tuning
        print("\nFine-tuning after initial augmented training...")
        self.classifier.fine_tune(X_train, y_train, X_val, y_val, num_layers_to_unfreeze=40, learning_rate=1e-5, epochs=10, batch_size=16)

        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the system"""
        print("Evaluating on test set...")
        results = self.classifier.evaluate(X_test, y_test)
        
        metrics = {
            'loss': results['loss'],
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        }
        
        print(f"\n=== Final Results ===")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def predict_single(self, image_path):
        """Predict health of a single coral image"""
        # Read and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        
        # Enhance the image
        img = self.enhancer.enhance_image(img)
        
        # Predict
        class_idx, confidence = self.classifier.predict(img)
        class_names = ['healthy_corals', 'bleached_corals']
        
        print(f"\nPrediction for {image_path}")
        print(f"Class: {class_names[class_idx]}")
        print(f"Confidence: {confidence:.4f}")
        
        return class_names[class_idx], confidence

def main():
    """Main execution pipeline"""
    print("="*60)
    print("Coral Reef Health Monitoring System - MobileNetV2 CNN")
    print("="*60)
    
    # Initialize system
    system = CoralHealthMonitoringSystem()
    
    # Data paths
    enhanced_data_path = 'data/enhanced'
    
    # Load pre-organized enhanced data
    try:
        X_train, y_train, class_names = system.load_enhanced_data(enhanced_data_path, split='train')
        X_val, y_val, _ = system.load_enhanced_data(enhanced_data_path, split='val')
        X_test, y_test, _ = system.load_enhanced_data(enhanced_data_path, split='test')
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print(f"Make sure enhanced images are in {enhanced_data_path}/train, /val, /test directories")
        return
    
    # Train the model
    print("\n" + "="*60)
    print("Phase 1: Training")
    print("="*60)
    system.train_system(X_train, y_train, X_val, y_val, epochs=20)
    
    # Evaluate the model
    print("\n" + "="*60)
    print("Phase 2: Evaluation")
    print("="*60)
    metrics = system.evaluate(X_test, y_test)
    
    # Save the trained model
    print("\n" + "="*60)
    print("Phase 3: Saving Model")
    print("="*60)
    os.makedirs('models', exist_ok=True)
    system.classifier.save_model('models/coral_classifier.h5')
    
    # Save metrics to file
    metrics_file = 'results/metrics.txt'
    os.makedirs('results', exist_ok=True)
    with open(metrics_file, 'w') as f:
        f.write("Coral Reef Health Monitoring - MobileNetV2 CNN Results\n")
        f.write("="*50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Metrics saved to {metrics_file}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()