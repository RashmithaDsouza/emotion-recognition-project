import tensorflow as tf
from model import build_emotion_model
from data_preprocessing import load_images_from_folder
import os

# dataset paths
train_dir = "dataset/train"
val_dir = "dataset/validation"

print("Loading training data...")
X_train, y_train = load_images_from_folder(train_dir)

print("Loading validation data...")
X_val, y_val = load_images_from_folder(val_dir)

# build model
model = build_emotion_model()

# compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting training...")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# create models folder if not exists
os.makedirs("models", exist_ok=True)

# save trained model
model.save("models/emotion_cnn.h5")

print("Model saved to models/emotion_cnn.h5")