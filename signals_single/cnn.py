import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Input, Dropout
from tensorflow.keras.models import Model


def cnn_block(x, filters, kernel_size, dropout):
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    return x


def model(signal_length, cnn_filters=32, cnn_layers=(9, 7, 5), dropout=0.2):
    inputs = Input(shape=(signal_length, 1))
    x = inputs

    for kernel_size in cnn_layers:
        x = cnn_block(
            x,
            filters=cnn_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    x = Conv1D(filters=1, kernel_size=1, padding='same')(x)
    outputs = Activation('sigmoid')(x)

    return Model(inputs, outputs)
