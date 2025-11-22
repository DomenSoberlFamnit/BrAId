import os

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_mlp(sample_size, architecture, X, Y):
    # Construct the model.
    model = keras.Sequential()
    model.add(layers.Input((sample_size,)))
    for layer_size in architecture:
        model.add(layers.Dense(layer_size, activation='relu'))
    model.add(layers.Dense(sample_size, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # Training parameters.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X, Y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    return model

def train_cnn(sample_size, X, Y):
    # Construct the model.
    model = keras.Sequential()
    model.add(layers.Input((sample_size, 1)))
    model.add(layers.Conv1D(filters=32, kernel_size=7, activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(sample_size, activation="relu"))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(sample_size, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # Training parameters.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X, Y,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    return model

