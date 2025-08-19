import pygame
import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

pygame.mixer.init()

# ✅ Paste this line here
alert_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'alert.wav')


# Load trained model
model = load_model(os.path.join(os.path.dirname(__file__), '..', 'model', 'mask_detector_model.h5'))

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path to alert sound (WAV format recommended for pygame)
alert_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'alert.wav')

# Cooldown timer for alert sound
last_alert_time = 0

# Start webcam
cap = cv2.VideoCapture(0)
IMG_SIZE = 100

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0][0]
        print("Prediction score:", prediction)  # Debug output

        # Adjust threshold based on your model's output range
        label = "No Mask" if prediction < 0.2 else "Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Play alert sound if no mask and cooldown passed
        if label == "No Mask" and time.time() - last_alert_time > 3:
            try:
                pygame.mixer.music.load(alert_path)
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.play()
                last_alert_time = time.time()
            except Exception as e:
                print("Sound error:", e)


        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
