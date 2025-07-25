import numpy as np
import cv2
import keras
import tensorflow as tf
model_from_json = tf.keras.models.model_from_json
preprocess_input = tf.keras.applications.vgg16.preprocess_input

# Load the saved model architecture
with open("model1/model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("model1/model.weights.h5")

# Compile the model (same optimizer, loss, and metrics as before)
loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                     loss="sparse_categorical_crossentropy", 
                     metrics=["accuracy"])

# List of class labels (same as used during training)
class_labels = ['surprise', 'fearful', 'angry', 'neutral', 'sad', 'disgust', 'happy']  # Adjust this list as per your classes

# Function to preprocess and predict on a single frame with emotion percentages
def predict_emotion(frame):
    # Resize to 224x224 (input size for VGG16)
    img = cv2.resize(frame, (224, 224))
    
    # Convert to float32 and preprocess for VGG16
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess using VGG16's preprocessing
    
    # Predict the class probabilities
    predictions = loaded_model.predict(img)[0]  # Get the probabilities
    
    # Convert probabilities to percentages
    percentages = [f"{class_labels[i]}: {prob * 1:.2f}%" for i, prob in enumerate(predictions)]
    
    return percentages

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from the webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break
    
    # Predict emotion percentages on the current frame
    emotion_percentages = predict_emotion(frame)
    
    # Display the resulting frame with the emotion percentages
    y0, dy = 50, 30
    for i, emotion_text in enumerate(emotion_percentages):
        y = y0 + i * dy
        cv2.putText(frame, emotion_text, (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
