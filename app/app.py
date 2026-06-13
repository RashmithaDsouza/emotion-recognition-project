import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image

# Base folder for resolving relative paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion_cnn.h5"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Facial Emotion Recognition System",
    layout="wide"
)

# --------------------------------------------------
# UI STYLE
# --------------------------------------------------

st.markdown("""
<style>
html, body, [class*="css"] {
    font-size:16px !important;
}
.block-container{
    max-width:1200px;
    padding-top:1rem;
}
h1{
    text-align:center;
    font-size:34px !important;
}
.result-box{
    font-size:22px;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# EMOTION LABELS
# --------------------------------------------------

emotion_labels = [
    'angry','disgust','fear','happy','neutral','sad','surprise'
]

emoji = {
    "angry":"😠",
    "disgust":"🤢",
    "fear":"😨",
    "happy":"😄",
    "neutral":"😐",
    "sad":"😢",
    "surprise":"😲"
}

# --------------------------------------------------
# BUILD MODEL (NO load_model)
# --------------------------------------------------

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))

    return model

# --------------------------------------------------
# LOAD MODEL (SAFE)
# --------------------------------------------------

@st.cache_resource
def load_my_model():
    model = build_model()
    model.load_weights(str(MODEL_PATH))
    return model

model = load_my_model()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Facial Emotion Recognition System")

st.write("Upload an image or use webcam for emotion detection.")

# --------------------------------------------------
# MODE SELECTION
# --------------------------------------------------

mode = st.sidebar.radio("Select Mode", ["Upload Image", "Webcam"])

# ==================================================
# IMAGE MODE
# ==================================================

if mode == "Upload Image":

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file is not None:

        image = Image.open(file)
        img = np.array(image)

        img = cv2.resize(img,(500,500))

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(gray,1.1,3)

        if len(faces) == 0:
            st.warning("No face detected")

        for (x,y,w,h) in faces:

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            face = gray[y:y+h,x:x+w]
            face = cv2.resize(face,(48,48))
            face = face/255.0
            face = face.reshape(1,48,48,1)

            prediction = model.predict(face, verbose=0)

            emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.image(img)
            st.markdown(
                f"<div class='result-box'>"
                f"Emotion: {emotion.upper()} {emoji[emotion]}<br>"
                f"Confidence: {confidence:.2f}"
                f"</div>",
                unsafe_allow_html=True
            )

# ==================================================
# WEBCAM MODE
# ==================================================

elif mode == "Webcam":

    start = st.checkbox("Start Camera")

    frame_window = st.image([])
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while start:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.1,4)

        for (x,y,w,h) in faces:

            face = gray[y:y+h,x:x+w]
            face = cv2.resize(face,(48,48))
            face = face/255.0
            face = face.reshape(1,48,48,1)

            prediction = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,emotion,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cap.release()