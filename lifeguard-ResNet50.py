import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from ultralytics import YOLO  
import time  
import logging  
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Execution of the code has started.")

# YOLO Model initialization
yolo_model = YOLO("yolov8n.pt")  
logging.info("YOLO model successfully loaded.")

# Datasets
IMAGE_SIZE = [224, 224]  
trainingFolder = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\train"
testFolder = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\test"
EPOCHS = 20  

# Check if training and test folders exist
try:
    assert os.path.exists(trainingFolder), "Training folder does not exist!"
    assert os.path.exists(testFolder), "Test folder does not exist!"
    logging.info("Training and test dataset folders are correctly located.")
except AssertionError as e:
    logging.error(f"Error: {e}")
    exit()

# Load the pretrained ResNet50 model
logging.info("Loading the ResNet50 model with pre-trained weights...")
myResnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)
for layer in myResnet.layers:
    layer.trainable = False  
logging.info("ResNet50 model has been successfully loaded and configured.")

# Adding layers 
Classes = os.listdir(trainingFolder)  
numOfClasses = len(Classes)  
global_avg_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(myResnet.output)
PlusFlattenLayer = Flatten()(global_avg_pooling_layer)
predictionLayer = Dense(numOfClasses, activation='softmax')(PlusFlattenLayer)
model = Model(inputs=myResnet.input, outputs=predictionLayer)

# Compiling the model
model.compile(
    loss='categorical_crossentropy',  
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  
    metrics=['accuracy']  
)
logging.info("Model compilation completed successfully.")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True  
)
test_datagen = ImageDataGenerator(rescale=1./255)  

# Loading datasets
training_set = train_datagen.flow_from_directory(
    trainingFolder, 
    target_size=IMAGE_SIZE,  
    batch_size=32,  
    class_mode='categorical'  
)
test_set = test_datagen.flow_from_directory(
    testFolder,  
    target_size=IMAGE_SIZE, 
    batch_size=32, 
    class_mode='categorical'
)

def train_and_evaluate_model():
    logging.info("Initiating model training...")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
    ) 
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )  
    history = model.fit(
        training_set,
        validation_data=test_set,
        epochs=EPOCHS,
        steps_per_epoch=len(training_set),  
        validation_steps=len(test_set),  
        callbacks=[checkpoint, early_stop]
    )
    logging.info("Model training completed successfully.")
    return history

# Visualizing training performance
def plot_training(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label="Training Accuracy")  
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")  
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()  

# Evaluating the model on test data
def evaluate_model():
    logging.info("Evaluating model performance...")
    test_steps = len(test_set)
    predictions = model.predict(test_set, steps=test_steps)  
    y_true = test_set.classes  
    y_pred = np.argmax(predictions, axis=1)  
    report = classification_report(y_true, y_pred, target_names=test_set.class_indices.keys())
    logging.info("Evaluation Report:")
    logging.info(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_set.class_indices.keys(), yticklabels=test_set.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Convert model to TensorFlow Lite format
def save_tflite_model():
    logging.info("Converting model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    logging.info("Model successfully converted to TFLite format.")

# Real-time detection using webcam and YOLO
def webcam_inference():
    cap = cv2.VideoCapture(1)  
    fps_start = time.time()  
    frame_count = 0  
    class_map = {v: k for k, v in training_set.class_indices.items()}
    logging.info("Starting webcam stream for real-time detection.")

    while True:
        ret, frame = cap.read()  
        if not ret:
            break

        frame_count += 1
        if frame_count % 24 == 0:  # Calculate FPS every 24 frames
            fps_end = time.time()
            fps = frame_count / (fps_end - fps_start)
            logging.info(f"Current FPS: {fps:.2f}")
            fps_start = fps_end
            frame_count = 0

        results = yolo_model.predict(frame, save=False) 
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue  

            cropped_object = frame[y1:y2, x1:x2]  
            if cropped_object.size > 0:
                resized_cropped_object = cv2.resize(cropped_object, (224, 224))  
                normalized_object = resized_cropped_object / 255.0 
                input_object = np.expand_dims(normalized_object, axis=0)  
                predictions = model.predict(input_object)  
                class_idx = np.argmax(predictions)  
                class_label = class_map[class_idx]  
                cv2.putText(frame, f"{class_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw bounding box

        cv2.imshow("Real-Time Detection with YOLO + ResNet50", frame)  
        if cv2.waitKey(1) & 0xFF == ord('r'):  
            break

    cap.release() 
    cv2.destroyAllWindows() 
    logging.info("Webcam stream ended.")

# Execute all processes
history = train_and_evaluate_model()
plot_training(history)
evaluate_model()
save_tflite_model()
webcam_inference()
