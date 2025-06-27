import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Data augmentation and normalization for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, 
    shear_range=0.2,  
    zoom_range=0.2, 
    horizontal_flip=True
)

# Normalization for the validation set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_dir = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\train"
val_dir = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\test"

# Load training dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load validation dataset
val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

shufflenet_url = "https://tfhub.dev/google/imagenet/shufflenet_v2_1.0_224/classification/5"
base_model = hub.KerasLayer(shufflenet_url, input_shape=(224, 224, 3), trainable=False)

num_classes = len(train_generator.class_indices)

# Building the model
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Model Checkpointing
checkpoint = ModelCheckpoint(
    filepath='model_weights.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Training
EPOCHS = 18

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[checkpoint]
)

# Enhanced Visualization of Training Results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].legend()
axes[0].set_title('Training vs. Validation Accuracy')

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].legend()
axes[1].set_title('Training vs. Validation Loss')

plt.tight_layout()
plt.show()

# Confusion Matrix and Classification Report with Seaborn Visualization
val_labels = val_generator.classes
val_predictions = model.predict(val_generator)
val_pred_classes = np.argmax(val_predictions, axis=1)

conf_matrix = confusion_matrix(val_labels, val_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

report = classification_report(val_labels, val_pred_classes, target_names=list(train_generator.class_indices.keys()))
print("Classification Report:\n", report)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(train_generator.class_indices.keys()), yticklabels=list(train_generator.class_indices.keys()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Starting the webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Normalizing the image
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Perform prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_label = list(train_generator.class_indices.keys())[predicted_class]
    confidence = predictions[0][predicted_class]

    # Display the classification result on the frame
    cv2.putText(frame, f"Class: {class_label}, Confidence: {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save classification result
    with open('classification_log.txt', 'a') as log_file:
        log_file.write(f"Class: {class_label}, Confidence: {confidence:.2f}\n")

    cv2.imshow('Real-Time ShuffleNet Classification', frame)

    # Exit when the 'r' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()
