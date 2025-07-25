import cv2
import keras
import numpy as np
import pandas as pd
import time
from openpyxl import Workbook

# Load the model
model_from_json = keras.models.model_from_json
emotion_dict = {0: 'angry', 1: 'disgust', 2: 'Fear', 3: 'happy', 4: 'Neutral', 5: 'sad', 6: 'surprise'}

# Load the model architecture and weights
with open(r"C:\\Users\\mahes\\Desktop\\DeepLearningProject\\model1\\model.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(r"C:\\Users\\mahes\\Desktop\\DeepLearningProject\\model1\\model.weights.h5")
print("Loaded model from disk")

# Initialize video capture
cap = cv2.VideoCapture(0)

emotion_indices = []
start_time = time.time()

# Load Haar Cascade face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # Extract face region and convert to 3 channels (for VGG16)
        roi_color_frame = frame[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_color_frame, (224, 224))
        
        # Preprocess image (expand dimensions and normalize as VGG16 expects)
        cropped_img = np.expand_dims(cropped_img, axis=0)  # Add batch dimension
        cropped_img = cropped_img.astype('float32') / 255.0  # Normalize pixel values

        # Predict emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]

        # Append the detected emotion index
        emotion_indices.append(maxindex)

        # Display the predicted emotion on the frame
        cv2.putText(frame, emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Show the video frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop after 35 seconds or if 'q' is pressed
    if time.time() - start_time > 35:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()

# Process and save results
if emotion_indices:
    avg_emotion_index = int(round(np.mean(emotion_indices)))
    avg_emotion = emotion_dict[avg_emotion_index]
else:
    avg_emotion = "No Emotion Detected"

emotion_result_df = pd.DataFrame([{'Person': 'Person1', 'Average Emotion': avg_emotion}])
emotion_result_df.to_excel(r"C:\\Users\\mahes\\Desktop\\DeepLearningProject\\emotion_data.xlsx", index=False)
print("Emotion data saved to Excel file.")
