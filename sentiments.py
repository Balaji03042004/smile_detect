import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained face and smile cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion recognition model
emotion_model = load_model('/content/download (5).jpg')  # Replace with the actual path

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect faces, smiles, and emotions in a webcam feed
def detect_emotions_webcam():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Resize and preprocess image for emotion model
            resized_roi = cv2.resize(roi_gray, (48, 48))
            resized_roi = resized_roi / 255.0
            resized_roi = np.reshape(resized_roi, (1, 48, 48, 1))

            # Predict emotion using the model
            emotion_pred = emotion_model.predict(resized_roi)
            emotion_index = np.argmax(emotion_pred)
            emotion_label = emotion_labels[emotion_index]

            # Display emotion label on the frame
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to start webcam emotion detection
detect_emotions_webcam()