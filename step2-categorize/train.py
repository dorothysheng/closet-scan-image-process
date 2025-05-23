import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import traceback
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Set TensorFlow log level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define class mapping (only keeping 5 classes)
class_mapping = {
    0: 0,  # T-shirt/top
    1: 1,  # Trouser
    3: 2,  # Dress
    4: 3,  # Coat
    6: 4,  # Shirt
}

# Class names (for display)
class_names = [
    "T-shirt/top",
    "Trouser",
    "Dress",
    "Coat",
    "Shirt"
]

def filter_classes(images, labels):
    """Filter dataset to include only the target clothing categories."""
    mask = np.isin(labels, list(class_mapping.keys()))
    filtered_images = images[mask]
    filtered_labels = np.array([class_mapping[label] for label in labels[mask]])
    
    print(f"Original dataset size: {len(labels)} samples")
    print(f"Filtered dataset size: {len(filtered_labels)} samples")
    print(f"Removed {len(labels) - len(filtered_labels)} samples from excluded classes")
    
    # Print class distribution
    unique, counts = np.unique(filtered_labels, return_counts=True)
    print("\nClass distribution after filtering:")
    for class_id, count in zip(unique, counts):
        print(f"  {class_names[class_id]}: {count} samples")
    
    return filtered_images, filtered_labels

def create_model():
    """Initialize CNN architecture for clothing classification."""
    model = keras.Sequential([
        # First Convolutional Block
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Second Convolutional Block
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Third Convolutional Block
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Classifier
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    try:
        # Create timestamp for model versioning
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        print("=" * 50)
        print("CLOTHING CLASSIFICATION MODEL TRAINING")
        print("=" * 50)
        
        # Load Fashion MNIST dataset
        print("\nLoading Fashion MNIST dataset...")
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        
        # Filter and normalize training data
        print("\nProcessing training data...")
        train_images, train_labels = filter_classes(train_images, train_labels)
        train_images = train_images / 255.0  # Normalize to [0,1]
        
        print("\nProcessing test data...")
        test_images, test_labels = filter_classes(test_images, test_labels)
        test_images = test_images / 255.0  # Normalize to [0,1]
        
        # Split training data into train and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        print(f"\nFinal dataset sizes:")
        print(f"  Training: {len(train_images)} samples")
        print(f"  Validation: {len(val_images)} samples")
        print(f"  Testing: {len(test_images)} samples")
        
        # Create model
        print("\nCreating model...")
        model = create_model()
        model.summary()
        
        # Reshape images for CNN input (add channel dimension)
        train_images = train_images[..., np.newaxis]
        val_images = val_images[..., np.newaxis]
        test_images = test_images[..., np.newaxis]
        
        # Data augmentation configuration
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Model training callbacks
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduler
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=f'models/best_model_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=f'logs/run_{timestamp}',
                histogram_freq=1
            )
        ]
        
        # Prepare data generator for training
        datagen.fit(train_images)
        
        # Create training generator
        train_generator = datagen.flow(
            train_images,
            train_labels,
            batch_size=64
        )
        
        # Calculate steps per epoch
        steps_per_epoch = len(train_images) // 64
        
        print("\nTraining model...")
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=30,
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Save the final model
        final_model_path = f'models/fashion_model_{timestamp}.h5'
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Load and evaluate the best model
        best_model_path = f'models/best_model_{timestamp}.h5'
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from {best_model_path}...")
            best_model = keras.models.load_model(best_model_path)
            
            # Evaluate best model
            print("Evaluating best model on test set...")
            best_loss, best_acc = best_model.evaluate(test_images, test_labels, verbose=2)
            print(f"Best model test accuracy: {best_acc:.4f}")
        
        # Save training history plot
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = f'models/training_history_{timestamp}.png'
        plt.savefig(plot_path)
        print(f"Training history plot saved to {plot_path}")
        
        # Save class mapping and metadata
        metadata = {
            'class_mapping': class_mapping,
            'class_names': class_names,
            'timestamp': timestamp,
            'model_path': final_model_path,
            'best_model_path': best_model_path,
            'training_samples': len(train_images),
            'validation_samples': len(val_images),
            'test_samples': len(test_images),
            'test_accuracy': float(test_acc),
            'best_accuracy': float(best_acc) if os.path.exists(best_model_path) else float(test_acc)
        }
        
        # Save metadata to JSON file
        metadata_path = f'models/model_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Model metadata saved to {metadata_path}")
        
        # Print summary
        print("\nTraining completed!")
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"Final test accuracy: {test_acc:.4f}")
        if os.path.exists(best_model_path):
            print(f"Best model test accuracy: {best_acc:.4f}")
            
        return 0
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()
