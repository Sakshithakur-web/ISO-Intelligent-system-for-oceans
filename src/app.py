# app.py - Simple command-line interface
import os
import sys
sys.path.append('src')

from preprocessing import CoralPreprocessor
from enhancement import UnderwaterImageEnhancer
from classification import CoralClassifier
import numpy as np

def main():
    print("=" * 50)
    print("Coral Reef Health Monitoring System")
    print("=" * 50)
    
    # Load trained model (you need to have trained models saved)
    classifier = CoralClassifier()
    
    # Try to load existing model weights
    if os.path.exists('models/coral_classifier.pkl'):
        classifier.load_model('models/coral_classifier.h5')
        print("Loaded existing model")
    else:
        print(" No trained model found. Please train the model first.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Predict coral health from image")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1/2): ")
        
        if choice == '1':
            image_path = input("Enter image path: ")
            
            if os.path.exists(image_path):
                preprocessor = CoralPreprocessor()
                enhancer = UnderwaterImageEnhancer()
                
                # Process image
                img = preprocessor.preprocess(image_path)
                enhanced = enhancer.enhance_image(img)
                class_idx, confidence = classifier.predict(enhanced)
            
                class_names = ['Healthy', 'Bleached']
                
                print(f"\n Prediction: {class_names[class_idx]}")
                print(f"Confidence: {confidence:.2%}")
            else:
                print(" Image not found!")
        
        elif choice == '2':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()