# Step 2: Clothing Classification Model

This module contains the clothing classification model that identifies the category of clothing items from images.

## Overview

The classification model is a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset to identify 5 categories of clothing:

1. T-shirt/top
2. Trouser
3. Dress
4. Coat
5. Shirt

## Files

- `train.py`: Script to train the classification model
- `predict.py`: Script to classify new clothing images
- `models/`: Directory containing trained models
- `logs/`: Directory containing training logs

## Usage

### Training the Model

```bash
# Navigate to the step2-categorize directory
cd step2-categorize

# Train the model
python train.py
```

The training process:
- Loads the Fashion MNIST dataset
- Filters to include only the 5 selected clothing categories
- Splits data into training, validation, and test sets
- Trains a CNN with data augmentation
- Saves the best model and training history

### Making Predictions

```bash
# Classify a clothing image
python predict.py path/to/your/image.jpg
```

## Model Architecture

The CNN architecture includes:
- Multiple convolutional layers with batch normalization
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

## Integration

This model is used in the clothing classification pipeline:

1. The model receives preprocessed images (with backgrounds removed)
2. It classifies the clothing item into one of the 5 categories
3. The classification result is passed to the 3D model selection module
4. The appropriate 3D model is selected based on the classification

## Performance

The model typically achieves:
- ~85-90% accuracy on the filtered Fashion MNIST test set
- ~70-80% accuracy on real-world clothing images

## Future Improvements

- Fine-tuning on real-world clothing images
- Adding more clothing categories
- Implementing ensemble methods for better accuracy
