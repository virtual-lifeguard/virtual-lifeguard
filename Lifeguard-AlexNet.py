import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("AlexNet model script execution has started.")

# Datasets
train_dir = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\train"  
val_dir = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\test" 
logging.info(f"Training dataset directory: {train_dir}")
logging.info(f"Validation dataset directory: {val_dir}")

try:
    assert os.path.exists(train_dir), "Training directory does not exist!"
    assert os.path.exists(val_dir), "Validation directory does not exist!"
    logging.info("Dataset directories have been verified.")
except AssertionError as e:
    logging.error(f"Directory Error: {e}")
    exit()

# Define the AlexNet architecture as a function
def AlexNet(input_shape=(224, 224, 3), num_classes=3):
   
    model = models.Sequential()
    # 1. Convolutional Layer
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2. Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3., 4., 5. Convolutional Layers
    model.add(layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5)) 
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))  
    model.add(layers.Dense(num_classes, activation='softmax'))  
    return model

# Num of classes
num_classes = len(os.listdir(train_dir))
logging.info(f"Number of classes in the training set: {num_classes}")

# Build and compile the model
model = AlexNet(input_shape=(224, 224, 3), num_classes=num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
logging.info("AlexNet model has been compiled successfully.")

# Display the model architecture
logging.info("AlexNet architecture:")
print(model.summary())

# Data augmentation 
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    shear_range=0.2,  
    zoom_range=0.2, 
    horizontal_flip=True  
)
val_datagen = ImageDataGenerator(rescale=1./255)  

# Load datasets 
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define training parameters
EPOCHS = 15  
logging.info(f"Model training will start for {EPOCHS} epochs.")

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)
logging.info("Model training has been completed.")

# Save the model
model.save("alexnet_model.h5")
logging.info("Trained AlexNet model has been saved as 'alexnet_model.h5'.")

# Function to plot training accuracy and loss curves
def plot_training_history(history):
    
    plt.figure(figsize=(10, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

# Plot the training history
plot_training_history(history)

# Evaluate the model using the confusion matrix
def evaluate_model():
    logging.info("Evaluating model and generating confusion matrix...")
    val_steps = len(val_generator)
    predictions = model.predict(val_generator, steps=val_steps)
    y_true = val_generator.classes
    y_pred = np.argmax(predictions, axis=1)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys())
    logging.info("Classification Report:\n")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Evaluate the model
evaluate_model()

# Start the webcam 
cap = cv2.VideoCapture(1)  
logging.info("Webcam stream started for real-time object detection.")

class_names = list(train_generator.class_indices.keys())

while True:
    ret, frame = cap.read()  
    if not ret:
        break

    # Preprocess the frame for prediction
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Perform prediction
    predictions = model.predict(input_frame)
    class_idx = np.argmax(predictions)
    class_label = class_names[class_idx]

    # Display the prediction result on the frame
    cv2.putText(frame, f"Class: {class_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit the loop when 'r' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

# Release resources and close the webcam feed
cap.release()
cv2.destroyAllWindows()
logging.info("Webcam stream stopped.")
        break

cap.release()
cv2.destroyAllWindows()
