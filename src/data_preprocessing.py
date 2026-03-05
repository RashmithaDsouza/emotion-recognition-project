import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/validation"

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_images_from_folder(folder):
    images = []
    labels = []

    for label, emotion in enumerate(emotion_labels):
        emotion_path = os.path.join(folder, emotion)

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = img / 255.0

            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    images = images.reshape(-1, 48, 48, 1)
    labels = to_categorical(labels, num_classes=7)

    return images, labels


print("Loading training data...")
X_train, y_train = load_images_from_folder(train_dir)

print("Loading validation data...")
X_val, y_val = load_images_from_folder(val_dir)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)