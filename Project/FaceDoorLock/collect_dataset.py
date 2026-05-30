# collect_dataset.py
import cv2
import os

# Step 1: Enter User ID (integer)
user_id = input("Enter User ID (e.g., 1): ")
path = f"dataset/{user_id}"
os.makedirs(path, exist_ok=True)

# Step 2: Initialize webcam and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{path}/{count}.jpg", face_img)  # save face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    
    cv2.imshow("Collecting Faces", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Collected {count} images for user {user_id}")
