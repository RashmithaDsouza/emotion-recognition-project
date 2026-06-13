import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data_preprocessing import load_images_from_folder

# load validation dataset
print("Loading validation data...")
X_val, y_val = load_images_from_folder("dataset/validation")

# load trained model
print("Loading trained model...")
model = tf.keras.models.load_model("models/emotion_cnn.h5")

# predictions
print("Running predictions...")
y_pred = model.predict(X_val)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=emotion_labels))