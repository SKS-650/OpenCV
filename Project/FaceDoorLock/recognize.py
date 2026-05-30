# recognize.py
import cv2
import pyttsx3  # Optional for voice feedback

# Step 1: Initialize voice engine
engine = pyttsx3.init()

# Step 2: Load recognizer & Haar cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 3: Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(face_img)
        
        # Step 4: Check confidence
        if conf < 50:  # Lower = better match
            cv2.putText(frame, f"User {id_} - UNLOCKED", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            engine.say("Access Granted. Door Unlocked!")
            engine.runAndWait()
        else:
            cv2.putText(frame, "Unknown - LOCKED", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            engine.say("Access Denied. Door Locked!")
            engine.runAndWait()
    
    cv2.imshow("Face Recognition Door Lock", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
