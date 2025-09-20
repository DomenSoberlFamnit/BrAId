import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

def save_image(name, image, norm=True):
    if norm:
        img = tf.keras.preprocessing.image.array_to_img(image[0])
    else:
        img = tf.keras.preprocessing.image.array_to_img(image[0].astype(np.uint8), scale=False)

    print(f"Saving {name}.png")
    img.save(f"{name}.png")

def test_map(image):
    feat_shape = tf.shape(image)
    Hf, Wf = feat_shape[1], feat_shape[2]

    image_cast = tf.cast(image, tf.float32) / 255.0

    save_image("1-input", image)

    dmin = tf.reduce_min(image_cast, axis=-1, keepdims=True)
    dmax = tf.reduce_max(image_cast, axis=-1, keepdims=True)

    lum = 1.0 - dmax

    save_image("2-lum", lum.numpy())

    sat = 1.0 - tf.where(tf.equal(dmax, 0), dmax, tf.divide(tf.subtract(dmax, dmin), dmax))
    
    save_image("3-sat", sat.numpy())

    mask = 0.9 * lum + 0.1 * sat
    save_image("4-mask", mask.numpy())

    resized = tf.image.resize(mask, size=[Hf, Wf], method='bilinear')    

    attn = tf.broadcast_to(resized, tf.shape(image))
    save_image("5-attn", attn.numpy())

    out = image * attn

    save_image("6-out", out.numpy(), norm=False)


    

class DarkAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
            trainable=True
        )
        self.gamma = self.add_weight(
            name="gamma", shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        features, image = inputs
        
        feat_shape = tf.shape(features)
        Hf, Wf = feat_shape[1], feat_shape[2]

        # Computer the attention map.
        image_cast = tf.cast(image, tf.float32) / 255.0

        dmin = tf.reduce_min(image_cast, axis=-1, keepdims=True)
        dmax = tf.reduce_max(image_cast, axis=-1, keepdims=True)

        lum = 1.0 - dmax
        sat = 1.0 - tf.where(tf.equal(dmax, 0), dmax, tf.divide(tf.subtract(dmax, dmin), dmax))
        mask = self.alpha * lum + (1.0 - self.alpha) * sat

        resized = tf.image.resize(mask, size=[Hf, Wf], method='bilinear')    
        attn = tf.broadcast_to(resized, tf.shape(features))

        # Apply the attention map.
        out = features * attn
        return out



class MaskLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        dmin = tf.reduce_min(inputs, axis=-1, keepdims=True)
        dmax = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        lum = 1.0 - dmax
        sat = 1.0 - tf.where(tf.equal(dmax, 0), dmax, tf.divide(tf.subtract(dmax, dmin), dmax))
        mask = self.alpha * lum + (1.0 - self.alpha) * sat

        return mask