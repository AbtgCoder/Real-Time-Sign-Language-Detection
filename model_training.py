from data_processing import actions

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 2172)))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(actions.shape[0], activation="softmax"))

    return model

def train_model(X_train, y_train, EPOCHS=1000):
    model = create_model()
    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )
    model.fit(X_train, y_train, epochs=EPOCHS)
    model.save("sign_gesture_model.h5")

def load_model():
    model = tf.keras.models.load_model("sign_gesture_model.h5")
    return model
