import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
ResNet50=tf.keras.applications.ResNet50
Dense=tf.keras.layers.Dense
Flatten=tf.keras.layers.Flatten
Model=tf.keras.models.Model
# Function to load and preprocess the dataset
def load_data(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # Get class labels from folder names
    print(f"Classes found: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                try:
                    img_path = os.path.join(class_path, img_name)
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")
                    continue

    images = np.array(images)
    labels = np.array(labels)

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    return images, labels, class_names

# Data loading
data_dir = 'train'  # Update with the correct path to your dataset
images, labels, class_names = load_data(data_dir)

# Encode the labels to integers (0 to n_classes-1)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # Transform to numerical labels

print("Unique labels after encoding:", np.unique(labels))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create a transfer learning model using ResNet50
def build_model(input_shape, n_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Build the model
input_shape = (224, 224, 3)
n_classes = len(class_names)
transfer_resnet = build_model(input_shape, n_classes)

# Compile the model
transfer_resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define a simple callback for early stopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Train the model
try:
    transfer_resnet.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=callbacks)
except Exception as e:
    print(f"Training error: {e}")

# Save model architecture to JSON
model_json = transfer_resnet.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)
    print("Model architecture saved as model_architecture.json")

# Save weights to HDF5
transfer_resnet.save_weights("model.weights.h5")
print("Model weights saved as model.weights.h5")
