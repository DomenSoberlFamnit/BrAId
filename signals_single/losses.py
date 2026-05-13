import tensorflow as tf
from tensorflow.keras.losses import Loss

class WeightedMSE(Loss):
    def __init__(self, positive_weight=1.0, negative_weight=1.0):
        super().__init__(name='wmse')
        self._positive_weight = positive_weight
        self._negative_weight = negative_weight

    def call(self, y_true, y_pred):
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(y_true, y_pred)
        weighted_loss = tf.where(tf.greater(y_true, 0), self._positive_weight * loss, self._negative_weight * loss)
        return tf.reduce_mean(weighted_loss)
