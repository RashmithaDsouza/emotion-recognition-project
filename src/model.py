import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_emotion_model():

    model = Sequential()

    # First convolution block
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Second convolution block
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Third convolution block
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten feature maps
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer for 7 emotions
    model.add(Dense(7, activation='softmax'))

    return model


if __name__ == "__main__":

    model = build_emotion_model()

    model.summary()