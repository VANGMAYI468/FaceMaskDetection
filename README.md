# FaceMaskDetection
Real-time face mask detection with alert system using OpenCV and TensorFlow.
📘 README.md — Face Mask Detection with Real-Time Alerts
markdown
# 😷 Face Mask Detection with Real-Time Alerts

A real-time computer vision project that detects whether a person is wearing a face mask using a webcam feed. If no mask is detected, an audible alert (`alert.wav`) is triggered. Built with Python, OpenCV, and a custom-trained CNN model.

----------------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 Features

- Real-time face detection via webcam
- Mask classification using a trained CNN model
- Audible alert system using `alert.wav`
- Organized folder structure for training, testing, and deployment
- Flask app integration (optional) for web-based interface

----------------------------------------------------------------------------------------------------------------------------------------------------

## 🧠 Tech Stack

| Component          |          Description                 |
|----------------    |--------------------------------------|
| Python             | Core programming language            |
| OpenCV             | Face detection and video processing  |
| Keras + TensorFlow | CNN model training and inference     |
| playsound / pygame | Sound alert playback                 |
| Flask (optional)   | Web interface                        |

----------------------------------------------------------------------------------------------------------------------------------------------------
## 📁 Project Structure

FaceMaskDetection/ │  ├── assets/ # Contains alert.wav and other media
                      ├── dataset/ # Organized training and testing images
                      ├── flask_app/ # Flask web interface (optional)
                      ├── model/ # Saved CNN model (.h5)
                      ├── raw_dataset/ # Original unprocessed images
                      ├── scripts/ # Python scripts for detection and testing
                      ├── training/ # Model training scripts and logs 
                      ├── venv/ # Virtual environment (ignored in Git)
                      ├── .gitignore # Git ignore rules
                      ├── README.md # Project documentation 
                      └── test_image.jpg # Sample image for testing


----------------------------------------------------------------------------------------------------------------------------------------------------

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/VANGMAYI468/FaceMaskDetection.git
cd FaceMaskDetection
2. Create Virtual Environment (Optional but Recommended)
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install -r requirements.txt
If requirements.txt is missing, install manually:

bash
pip install opencv-python keras tensorflow playsound
4. Run the Detection Script
bash
python scripts/detect_mask_alert.py
Make sure your webcam is connected and accessible.
----------------------------------------------------------------------------------------------------------------------------------------------------
### 🔊 Alert System
The script plays assets/alert.wav when a person is detected without a mask.
If sound doesn’t play, check:  File path: should be "assets/alert.wav"
Dependency: install playsound or use pygame as fallback
Local environment: GitHub hosts the file, but sound plays only locally
----------------------------------------------------------------------------------------------------------------------------------------------------
### 🧪 Testing
You can test the model using:

bash
python scripts/test_sound.py
python scripts/detect_mask_webcam.py
Or use test_image.jpg for static image detection.
-----------------------------------------------------------------------------------------------------------------------------------------------------
### 📊 Model Training
Training scripts are located in training/
Dataset is split into dataset/train/ and dataset/test/
Model is saved as model/mask_detector_model.h5
-----------------------------------------------------------------------------------------------------------------------------------------------------
### 🌐 Flask Web App (Optional)
To run the web interface:

bash
cd flask_app
python app.py
Then visit http://localhost:5000 in your browser.
