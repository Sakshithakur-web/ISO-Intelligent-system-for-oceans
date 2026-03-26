"""
Create a basic MobileNetV2 model for testing
"""
import tensorflow as tf
from tensorflow import keras
import os

def create_test_model():
    """Create a simple MobileNetV2-based model for coral classification"""
    print("Creating test model...")
    
    # Build a simple MobileNetV2 model
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom top layers
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2, activation='softmax')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/coral_classifier.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path

if __name__ == '__main__':
    create_test_model()
