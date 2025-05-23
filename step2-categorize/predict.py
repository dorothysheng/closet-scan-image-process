import os
import sys
import numpy as np
import json
from tensorflow import keras
from PIL import Image
import glob

# Set TensorFlow log level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Class names for our filtered 5-class model
class_names = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Dress",        # 2
    "Coat",         # 3
    "Shirt"         # 4
]

def load_and_prepare_image(image_path):
    """Preprocess input image for model inference."""
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Channel dimension
    return img_array

def predict_image(model, image_path):
    """Classify clothing item in the input image."""
    img_array = load_and_prepare_image(image_path)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return {
        'class': class_names[predicted_class],
        'class_id': int(predicted_class),
        'confidence': float(confidence),
        'all_predictions': {name: float(prob) for name, prob in zip(class_names, predictions[0])}
    }

def find_latest_model():
    """Locate the most recent model artifact."""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return None
        
    model_files = glob.glob(os.path.join(models_dir, 'fashion_model_*.h5'))
    
    if not model_files:
        model_files = glob.glob(os.path.join(models_dir, 'best_model_*.h5'))
        
    if not model_files:
        return None
        
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_files[0]

def main():
    # Load classification model
    model_path = find_latest_model()
    
    if not model_path:
        print("Error: No trained model found in the models directory")
        print("Please run train.py first to train and save the model")
        return 1
    
    print(f"Using model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Check if image path was provided as command line argument
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        print(f"Using provided image: {test_image_path}")
    else:
        # If no argument provided, prompt for input
        test_image_path = input("Enter path to an image file: ")
    
    if not test_image_path or not os.path.exists(test_image_path):
        print("Error: No valid image path provided")
        print("Please provide a path to an image file")
        return 1
    
    try:
        # Make prediction
        result = predict_image(model, test_image_path)
        
        # Display results
        print("\nPrediction Results:")
        print(f"Predicted class: {result['class']} (ID: {result['class_id']})")
        print(f"Confidence: {result['confidence']:.2%}")
        
        print("\nAll class probabilities:")
        for class_name, prob in result['all_predictions'].items():
            print(f"{class_name}: {prob:.2%}")
            
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    main()
