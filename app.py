from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
import cv2
import os
from src.enhancement import UnderwaterImageEnhancer

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
try:
    model = tf.keras.models.load_model('models/coral_classifier.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model - {str(e)}")
    model = None

# Initialize image enhancer
enhancer = UnderwaterImageEnhancer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the original image
        file_content = file.read()
        img = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Store original image for display
        original_img = img.copy()
        
        # Resize for model input
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run prediction on original image
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        # Enhance the image
        enhanced_array = enhancer.enhance_image(img_array[0])
        enhanced_array = np.expand_dims(enhanced_array, axis=0)
        
        # Run prediction on enhanced image
        enhanced_prediction = model.predict(enhanced_array)
        enhanced_class_idx = np.argmax(enhanced_prediction[0])
        enhanced_confidence = float(enhanced_prediction[0][enhanced_class_idx])
        
        # Convert images to base64 for frontend display
        # Original image
        original_buffer = io.BytesIO()
        original_img.save(original_buffer, format='JPEG')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
        
        # Enhanced image
        enhanced_pil = Image.fromarray((enhanced_array[0] * 255).astype(np.uint8))
        enhanced_buffer = io.BytesIO()
        enhanced_pil.save(enhanced_buffer, format='JPEG')
        enhanced_base64 = base64.b64encode(enhanced_buffer.getvalue()).decode('utf-8')
        
        classes = ['Healthy', 'Bleached']
        return jsonify({
            'original_prediction': classes[class_idx],
            'original_confidence': confidence,
            'enhanced_prediction': classes[enhanced_class_idx],
            'enhanced_confidence': enhanced_confidence,
            'original_image': f'data:image/jpeg;base64,{original_base64}',
            'enhanced_image': f'data:image/jpeg;base64,{enhanced_base64}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)