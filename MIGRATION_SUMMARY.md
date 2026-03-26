# Coral Reef Monitoring: RandomForest → MobileNetV2 CNN Migration

## Summary of Changes

Successfully replaced RandomForest classifier with a **MobileNetV2-based CNN** using TensorFlow/Keras transfer learning.

---

## 📋 Files Modified

### 1. **requirements.txt**
Added TensorFlow and Keras dependencies:
```
tensorflow==2.15.0
keras==3.0.0
```

### 2. **src/classification.py**
Complete rewrite from scikit-learn RandomForest to TensorFlow/Keras:

**Architecture:**
- **Base Model:** MobileNetV2 (pre-trained on ImageNet, frozen weights)
- **Top Layers:**
  - Global Average Pooling
  - Dense(256, ReLU) + Dropout(0.5)
  - Dense(128, ReLU) + Dropout(0.3)
  - Dense(num_classes, Softmax)

**Training:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Sparse Categorical Crossentropy
- Callbacks:
  - EarlyStopping (monitor val_loss, patience=5)
  - ReduceLROnPlateau (factor=0.5, patience=3)

**Inference:**
- Input: 224×224×3 RGB images, normalized to [0, 1]
- Output: Class prediction with confidence score

### 3. **src/main.py**
Refactored pipeline to use pre-organized enhanced data:

**Data Loading:**
- `load_enhanced_data()` loads from: `data/enhanced/{train,val,test}/{class}/`
- Classes: `healthy_corals`, `bleached_corals`
- Automatic resizing to 224×224 and normalization

**Execution Pipeline:**
1. Load train/val/test splits from `data/enhanced/`
2. Create MobileNetV2 classifier
3. Train for up to 20 epochs with validation
4. Evaluate on test set
5. Save model to `models/coral_classifier.h5`
6. Save metrics to `results/metrics.txt`

**Output Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Loss

---

## 🚀 Usage

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```bash
python src/main.py
```

### Expected Directory Structure
```
coral-reef-monitoring/
├── data/
│   └── enhanced/
│       ├── train/
│       │   ├── healthy_corals/ (images...)
│       │   └── bleached_corals/ (images...)
│       ├── val/
│       │   ├── healthy_corals/ (images...)
│       │   └── bleached_corals/ (images...)
│       └── test/
│           ├── healthy_corals/ (images...)
│           └── bleached_corals/ (images...)
├── models/
│   └── coral_classifier.h5 (output)
├── results/
│   └── metrics.txt (output)
└── src/
    ├── main.py
    ├── classification.py
    ├── preprocessing.py
    ├── enhancement.py
    └── utils.py
```

---

## 📊 Key Improvements

| Aspect | RandomForest | MobileNetV2 CNN |
|--------|-------------|-----------------|
| **Architecture** | Scikit-learn ensemble | Deep CNN with transfer learning |
| **Input Processing** | Flattened vectors (lossy) | 2D spatial data preserved |
| **Feature Learning** | Hand-crafted features | Learned hierarchical features |
| **Scalability** | Limited by memory | GPU-accelerated training |
| **Accuracy Potential** | Moderate | High (image classification) |
| **Fine-tuning** | Difficult | Easy (layer unfreezing) |
| **Model Size** | ~100MB | ~90MB |

---

## 🔧 Advanced Usage

### Fine-tuning MobileNetV2
To fine-tune base model layers (in `classification.py`):
```python
# Unfreeze last N layers of base model
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False
```

### Predict on Single Image
```python
from src.classification import CoralClassifier
import numpy as np
import cv2

# Load and preprocess image
img = cv2.imread('image.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0

# Predict
classifier = CoralClassifier(num_classes=2)
classifier.load_model('models/coral_classifier.h5')
class_idx, confidence = classifier.predict(img)
print(f"Predicted: {['healthy', 'bleached'][class_idx]} ({confidence:.2%})")
```

### Load Pre-trained Model
```python
from src.main import CoralHealthMonitoringSystem

system = CoralHealthMonitoringSystem()
system.classifier.load_model('models/coral_classifier.h5')
```

---

## ✅ Performance Tracking

The pipeline automatically saves:
1. **Model:** `models/coral_classifier.h5` (Keras format)
2. **Metrics:** `results/metrics.txt` with:
   - Test Accuracy
   - Test Precision
   - Test Recall
   - Test F1-Score
   - Test Loss

---

## 📝 Notes

- **Transfer Learning:** Base MobileNetV2 weights are frozen during initial training for stability. These can be unfrozen for fine-tuning with additional epochs.
- **Image Normalization:** All images are normalized to [0,1] float32 format as expected by MobileNetV2.
- **Class Names:** Automatically detected from directory structure (`healthy_corals`, `bleached_corals`).
- **GPU Support:** Training will automatically use GPU if available (CUDA/cuDNN).

---

## 🐛 Troubleshooting

**Issue:** `No images found in data/enhanced/...`
- Ensure enhanced images are organized in the directory structure above

**Issue:** GPU memory error
- Reduce `batch_size` in `main.py` train_system call

**Issue:** Poor accuracy
- Ensure data quality in enhanced directories
- Increase epochs (modify in `main()` function)
- Check image preprocessing (224×224 size requirement)

---

## 📚 References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras Sequential Model](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
