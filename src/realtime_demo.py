import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/emotion_cnn.h5")

# Emotion labels
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

print("Webcam started. Press 'q' to exit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48,48))

        face = face / 255.0

        face = np.reshape(face, (1,48,48,1))

        prediction = model.predict(face, verbose=0)

        emotion = emotion_labels[np.argmax(prediction)]

        # Draw rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Show emotion label
        cv2.putText(frame, emotion,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()