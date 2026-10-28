import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout, Add, Input, Flatten
from tensorflow.keras.models import Model

def tcn_block(x, filters, kernel_size, dilation_rate, dropout):
    residual = x

    x = Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate
    )(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(dropout)(x)

    x = Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate
    )(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(dropout)(x)

    output = Add()([x, residual])
    #output = Activation('relu')(output)

    return output

def model(signal_length, tcn_filters, tcn_dilations, dropout):
    inputs = Input(shape=(signal_length, 1))
    x = inputs

    for d in tcn_dilations:
        x = tcn_block(
            x,
            filters=tcn_filters,
            kernel_size=5,
            dilation_rate=d,
            dropout=dropout
        )

    x = Conv1D(filters=1, kernel_size=1, padding="same", activation="sigmoid")(x)
    #x = Conv1D(filters=1, kernel_size=1, padding="same")(x)
    #x = Flatten()(x)
    outputs = x

    return Model(inputs, outputs)