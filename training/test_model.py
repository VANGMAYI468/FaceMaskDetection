import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("C:/Users/vangm/OneDrive/Desktop/FaceMaskDetection/model/mask_detector_model.h5")

# Load and preprocess a test image
img = cv2.imread("C:/Users/vangm/OneDrive/Desktop/FaceMaskDetection/test_image.jpg")
img = cv2.resize(img, (100, 100))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
print("Mask" if prediction[0][0] < 0.5 else "No Mask")
