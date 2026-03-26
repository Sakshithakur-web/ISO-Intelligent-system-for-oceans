// Coral Reef Health Monitoring System - Frontend Script
// Handles image upload, preview, classification, and result display

document.addEventListener('DOMContentLoaded', function() {
    // DOM element references
    const fileInput = document.getElementById('file-input');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const classifyBtn = document.getElementById('classify-btn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const originalImageDisplay = document.getElementById('original-image-display');
    const enhancedImageDisplay = document.getElementById('enhanced-image-display');
    const originalPredictionSpan = document.getElementById('original-prediction');
    const enhancedPredictionSpan = document.getElementById('enhanced-prediction');
    const originalConfidenceSpan = document.getElementById('original-confidence');
    const enhancedConfidenceSpan = document.getElementById('enhanced-confidence');
    const originalConfidenceFill = document.getElementById('original-confidence-fill');
    const enhancedConfidenceFill = document.getElementById('enhanced-confidence-fill');
    const originalConfidenceText = document.getElementById('original-confidence-text');
    const enhancedConfidenceText = document.getElementById('enhanced-confidence-text');
    const errorMessage = document.getElementById('error-message');
    const errorText = errorMessage.querySelector('p');

    // State management
    let isProcessing = false;

    /**
     * Handles file selection and displays image preview
     */
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];

        // Reset previous states
        hideError();
        hideResult();

        if (file) {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file.');
                classifyBtn.disabled = true;
                return;
            }

            // Validate file size (max 10MB)
            if (file.size > 10 * 1024 * 1024) {
                showError('File size must be less than 10MB.');
                classifyBtn.disabled = true;
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewContainer.style.display = 'block';
                classifyBtn.disabled = false;

                // Add fade-in animation
                imagePreviewContainer.style.animation = 'none';
                setTimeout(() => {
                    imagePreviewContainer.style.animation = 'fadeIn 0.5s ease-out';
                }, 10);
            };
            reader.onerror = function() {
                showError('Error reading the selected file.');
                classifyBtn.disabled = true;
            };
            reader.readAsDataURL(file);
        } else {
            imagePreviewContainer.style.display = 'none';
            classifyBtn.disabled = true;
        }
    });

    /**
     * Handles classification button click
     */
    classifyBtn.addEventListener('click', async function() {
        const file = fileInput.files[0];

        if (!file) {
            showError('Please select an image file first.');
            return;
        }

        if (isProcessing) {
            return; // Prevent multiple simultaneous requests
        }

        // Update UI state
        isProcessing = true;
        classifyBtn.disabled = true;
        classifyBtn.textContent = 'Processing...';
        hideError();
        hideResult();
        showLoading();

        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);

            // Send request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Display successful result
            displayResult(data);

        } catch (error) {
            console.error('Classification error:', error);
            showError(error.message || 'An unexpected error occurred. Please try again.');
        } finally {
            // Reset UI state
            isProcessing = false;
            classifyBtn.disabled = false;
            classifyBtn.textContent = 'Classify';
            hideLoading();
        }
    });

    /**
     * Displays the classification result
     */
    function displayResult(data) {
        // Set original image and prediction
        originalImageDisplay.src = data.original_image;
        originalPredictionSpan.textContent = data.original_prediction;
        originalPredictionSpan.className = 'value prediction-result';
        if (data.original_prediction.toLowerCase() === 'healthy') {
            originalPredictionSpan.classList.add('healthy');
        } else if (data.original_prediction.toLowerCase() === 'bleached') {
            originalPredictionSpan.classList.add('bleached');
        }

        // Set enhanced image and prediction
        enhancedImageDisplay.src = data.enhanced_image;
        enhancedPredictionSpan.textContent = data.enhanced_prediction;
        enhancedPredictionSpan.className = 'value prediction-result';
        if (data.enhanced_prediction.toLowerCase() === 'healthy') {
            enhancedPredictionSpan.classList.add('healthy');
        } else if (data.enhanced_prediction.toLowerCase() === 'bleached') {
            enhancedPredictionSpan.classList.add('bleached');
        }

        // Set confidence values
        const originalConfidencePercentage = (data.original_confidence * 100).toFixed(2);
        const enhancedConfidencePercentage = (data.enhanced_confidence * 100).toFixed(2);
        
        originalConfidenceSpan.textContent = originalConfidencePercentage + '%';
        enhancedConfidenceSpan.textContent = enhancedConfidencePercentage + '%';
        originalConfidenceText.textContent = originalConfidencePercentage + '%';
        enhancedConfidenceText.textContent = enhancedConfidencePercentage + '%';

        // Show result with animation
        result.style.display = 'block';
        result.style.animation = 'none';
        setTimeout(() => {
            result.style.animation = 'slideInUp 0.6s ease-out';
        }, 10);

        // Animate confidence bars
        setTimeout(() => {
            originalConfidenceFill.style.width = originalConfidencePercentage + '%';
            enhancedConfidenceFill.style.width = enhancedConfidencePercentage + '%';
        }, 300);
    }

    /**
     * Shows loading indicator
     */
    function showLoading() {
        loading.style.display = 'flex';
        loading.style.animation = 'none';
        setTimeout(() => {
            loading.style.animation = 'fadeIn 0.3s ease-out';
        }, 10);
    }

    /**
     * Hides loading indicator
     */
    function hideLoading() {
        loading.style.display = 'none';
    }

    /**
     * Shows error message
     */
    function showError(message) {
        errorText.textContent = message;
        errorMessage.style.display = 'block';
        errorMessage.style.animation = 'none';
        setTimeout(() => {
            errorMessage.style.animation = 'fadeIn 0.3s ease-out';
        }, 10);
    }

    /**
     * Hides error message
     */
    function hideError() {
        errorMessage.style.display = 'none';
    }

    /**
     * Hides result display
     */
    function hideResult() {
        result.style.display = 'none';
        originalConfidenceFill.style.width = '0%';
        enhancedConfidenceFill.style.width = '0%';
    }

    // Initialize the application
    console.log('Coral Reef Health Monitoring System initialized');
});