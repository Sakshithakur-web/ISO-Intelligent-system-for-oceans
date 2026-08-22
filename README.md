<<<<<<< HEAD
# Coral Reef Health Monitoring System using machine learning

A machine learning system for monitoring coral reef health using underwater image analysis.

## Features

- **Image Preprocessing**: Resize, normalize, and apply noise reduction to underwater images
- **Image Enhancement**: Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for better underwater image quality
- **Coral Classification**: Random Forest classifier to distinguish between healthy and bleached corals
- **Command-line Interface**: Easy-to-use CLI for single image predictions
## Project Structure

```
coral-reef-monitoring/
├── data/
│   ├── raw/
│   │   ├── healthy_corals/     # Healthy coral images
│   │   └── bleached_corals/    # Bleached coral images
│   ├── processed/              # Processed images
│   └── enhanced/               # Enhanced images
├── models/                     # Saved trained models
├── results/                    # Training results and plots
├── src/
│   ├── preprocessing.py        # Image preprocessing utilities
│   ├── enhancement.py          # Image enhancement using CLAHE
│   ├── classification.py       # Random Forest classifier
│   ├── main.py                 # Main training pipeline
│   ├── app.py                  # Command-line interface
│   └── utils.py                # Utility functions for plotting
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Dataset

The system expects images organized in the following structure:
```
data/raw/
├── healthy_corals/    # Images of healthy corals
└── bleached_corals/   # Images of bleached corals
```

## Model Performance

The current Random Forest model achieves approximately 74% accuracy on the test set.

## Technologies Used

- **Python**: Core programming language
- **OpenCV**: Image processing
- **scikit-learn**: Machine learning (Random Forest)
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Joblib**: Model serialization

## Future Improvements

- Implement deep learning models (CNNs) for better accuracy
- Add more coral health categories
- Implement real-time video analysis

## Note on Model & Data

⚠️ **Large Files Not Included**: 
- Model weights (`*.h5` files) - typically 100+ MB
- Raw dataset - can be GBs in size

These are excluded from the repository to keep it lightweight and performant. Follow the setup instructions above to obtain or generate these files before running the application.

## Repository Size

This lean repository setup (~5-10 MB) allows for:
- ✅ Fast cloning
- ✅ Easy collaboration 
- ✅ CI/CD pipelines without bloat
- ✅ Efficient storage
- Add web interface for easier use
- Include more advanced image enhancement techniques

