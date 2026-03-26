# src/classification.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CoralClassifier:
    def __init__(self, num_classes=2, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self._build_classifier()

    def _build_classifier(self):
        """Build MobileNetV2-based CNN with transfer learning for coral bleaching detection"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling=None
        )
        
        # Freeze base model weights for transfer learning
        base_model.trainable = False
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create full model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("MobileNetV2-based CNN initialized successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total parameters: {self.model.count_params():,}")

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
        """Train the classifier"""
        print("\n" + "="*50)
        print("Starting Training with MobileNetV2")
        print("="*50)

        # Normalize images to [0, 1] range expected by MobileNetV2
        X_train = np.clip(X_train, 0, 1).astype(np.float32)
        if X_val is not None:
            X_val = np.clip(X_val, 0, 1).astype(np.float32)

        print(f"Training on {len(X_train)} samples...")
        if X_val is not None:
            print(f"Validating on {len(X_val)} samples...")

        # Callbacks for training
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )

        # Train the model
        if X_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[reduce_lr],
                verbose=1
            )

        print("Training completed!")
        return self.history

    def train_with_augmentation(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """Train the model using data augmentation to improve generalization"""
        print("\n" + "="*50)
        print("Starting Augmented Training")
        print("="*50)

        X_train = np.clip(X_train, 0, 1).astype(np.float32)
        X_val = np.clip(X_val, 0, 1).astype(np.float32)

        # Augmentation config suitable for underwater images
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        train_gen = data_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_gen = tf.keras.preprocessing.image.ImageDataGenerator().flow(X_val, y_val, batch_size=batch_size)

        early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)

        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=max(1, len(X_train) // batch_size),
            validation_steps=max(1, len(X_val) // batch_size),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        print("Augmented Training completed!")
        return self.history

    def fine_tune(self, num_layers_to_unfreeze=40, learning_rate=1e-5, epochs=10, batch_size=16):
        """Unfreeze top layers of the base model and continue training for improved accuracy"""
        print("\n" + "="*50)
        print("Starting fine-tuning")
        print("="*50)

        base_model = self.model.layers[0]
        base_model.trainable = True

        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"Fine-tuning: unfrozen last {num_layers_to_unfreeze} layers")
        print(f"Fine-tuning learning rate: {learning_rate}")

        # this method expects `self.history` to have been set in initial training or augmentation.
        return self.history

    def predict(self, image):
        """Predict coral health class"""
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normalize image
        image = np.clip(image, 0, 1).astype(np.float32)
        
        # Get prediction probabilities
        probabilities = self.model.predict(image, verbose=0)[0]
        
        # Get the predicted class
        class_idx = np.argmax(probabilities)
        confidence = probabilities[class_idx]

        return class_idx, confidence

    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("\n" + "="*50)
        print("Model Evaluation")
        print("="*50)

        # Normalize test data
        X_test = np.clip(X_test, 0, 1).astype(np.float32)

        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Calculate loss
        loss, eval_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1-Score: {metrics['f1_score']:.4f}")

        return metrics

    def save_model(self, filepath='models/coral_classifier.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='models/coral_classifier.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def fine_tune(self, X_train, y_train, X_val, y_val, num_layers_to_unfreeze=50, learning_rate=1e-5, epochs=10, batch_size=16):
        """Unfreeze top layers of the base model and continue fine-tuning"""
        print("\n" + "="*50)
        print("Starting fine-tuning")
        print("="*50)

        base_model = self.model.layers[0]
        base_model.trainable = True

        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"Fine-tuning: unfrozen last {num_layers_to_unfreeze} layers")
        print(f"Learning rate: {learning_rate}")

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        print("Fine-tuning completed!")
        return history