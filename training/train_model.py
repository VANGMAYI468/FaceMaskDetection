import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
train_dir = "C:/Users/vangm/OneDrive/Desktop/FaceMaskDetection/dataset/train"
test_dir = "C:/Users/vangm/OneDrive/Desktop/FaceMaskDetection/dataset/test"

# Image preprocessing
img_size = 100
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save model automatically during training
checkpoint = ModelCheckpoint(
    "C:/Users/vangm/OneDrive/Desktop/FaceMaskDetection/model/mask_detector_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

try:
    model.fit(train_data, epochs=10, validation_data=test_data, callbacks=[checkpoint], verbose=2)

    print("Training complete.")
except KeyboardInterrupt:
    print("Training interrupted. Best model saved if val_accuracy improved.")
