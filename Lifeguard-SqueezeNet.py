import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Datasets
trainingFolder = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\train"  
testFolder = "C:\\Users\\ruyaa\\OneDrive\\Desktop\\DATASET\\test"       

# Define the SqueezeNet architecture
def SqueezeNet(input_shape=(224, 224, 3), num_classes=3):
    input_layer = layers.Input(shape=input_shape)

    # Conv1
    x = layers.Conv2D(96, (7, 7), strides=(2, 2), activation='relu')(input_layer)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Fire Modules
    def fire_module(x, squeeze_filters, expand_filters):
        squeeze = layers.Conv2D(squeeze_filters, (1, 1), activation='relu')(x)
        expand1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu')(squeeze)
        expand3x3 = layers.Conv2D(expand_filters, (3, 3), padding='same', activation='relu')(squeeze)
        return layers.Concatenate()([expand1x1, expand3x3])

    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 32, 128)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 64, 256)

    # Final Conv Layer
    x = layers.Conv2D(num_classes, (1, 1), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    output_layer = layers.Activation('softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)

# Build the model
IMAGE_SIZE = [224, 224]
NUM_CLASSES = len(os.listdir(trainingFolder))  
model = SqueezeNet(input_shape=IMAGE_SIZE + [3], num_classes=NUM_CLASSES)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print(model.summary())

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset
training_set = train_datagen.flow_from_directory(
    trainingFolder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load test dataset
test_set = test_datagen.flow_from_directory(
    testFolder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print class names
class_names = list(training_set.class_indices.keys())
print("Class Names:", class_names)


def plot_training(history):
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Learning rate scheduler function
def lr_schedule(epoch, lr):
    if epoch > 10:
        return lr * 0.1  
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Training
EPOCHS = 20
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=[lr_scheduler]
)

# Visualize training metrics
plot_training(history)

# Evaluate the model and plot confusion matrix
def evaluate_model(model, test_set):
    test_steps = len(test_set)
    predictions = model.predict(test_set, steps=test_steps)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_set.classes

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

evaluate_model(model, test_set)

# Convert the model 
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("squeezenet_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format!")

# Start the webcam 
cap = cv2.VideoCapture(1) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture webcam frame. Exiting...")
        break

    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Perform prediction
    predictions = model.predict(input_frame)
    class_idx = np.argmax(predictions)
    class_label = class_names[class_idx]

    # Display classification result on the frame
    cv2.putText(frame, f"Class: {class_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('r'):  
        break

cap.release()
cv2.destroyAllWindows()
