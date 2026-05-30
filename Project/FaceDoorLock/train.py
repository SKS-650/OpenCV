# train.py
import cv2
import numpy as np
import os

# Step 1: Initialize LBPH Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Prepare training data
faces = []
ids = []

dataset_path = 'dataset/'

for user in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user)
    for image_name in os.listdir(user_path):
        img_path = os.path.join(user_path, image_name)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(gray)
        ids.append(int(user))  # User ID as integer

# Step 3: Train the recognizer
recognizer.train(faces, np.array(ids))

# Step 4: Save the trained model
os.makedirs('trainer', exist_ok=True)
recognizer.save('trainer/trainer.yml')
print("Training completed and model saved!")
