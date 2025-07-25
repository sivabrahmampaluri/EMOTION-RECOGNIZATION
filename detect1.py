import numpy as np
import cv2
import os
import keras
import tensorflow as tf
model_from_json=tf.keras.models.model_from_json
preprocess_input=tf.keras.applications.vgg16.preprocess_input
# Load the saved model architecture
with open("model/model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("model/model.weights.h5")

# Compile the model (same optimizer, loss, and metrics as before)
loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                     loss="sparse_categorical_crossentropy", 
                     metrics=["accuracy"])

# Directory for test images (replace with your test image folder)
test_directory = 'images'

# List of class labels (same as used during training)
class_labels = ['surprise', 'fearful', 'angry', 'neutral', 'sad', 'disgust', 'happy']  # Adjust this list as per your classes

# Function to preprocess and predict on a single image
def test_single_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Resize to 224x224 (input size for VGG16)
    img = cv2.resize(img, (224, 224))
    
    # Convert to float32 and preprocess for VGG16
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess using VGG16's preprocessing
    
    # Predict the class probabilities
    predictions = loaded_model.predict(img)
    
    # Get the index of the highest probability
    predicted_class = np.argmax(predictions)
    
    # Map the predicted index to the corresponding class label
    predicted_label = class_labels[predicted_class]
    
    # Print prediction
    print(f"Predicted Emotion for {os.path.basename(image_path)}: {predicted_label}")
    return predicted_label

# Test on a set of test images
test_images = os.listdir(test_directory)
for test_img in test_images:
    test_path = os.path.join(test_directory, test_img)
    test_single_image(test_path)
