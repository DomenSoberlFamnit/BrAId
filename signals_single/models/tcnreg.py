import os

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, losses
from progress.bar import Bar
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

from models.base import BraidModel
import dataset
import visualizations
import evaluation

class SelfAttentionBlock(layers.Layer):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn_output = self.mha(query=x, value=x)  # safer form
        return self.norm(x + attn_output)


class TCNReg(BraidModel):
    def tcn_block(x, filters, kernel_size, dilation_rate, dropout):
        residual = x

        x = layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            dilation_rate=dilation_rate
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Dropout(dropout)(x)

        x = layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            dilation_rate=dilation_rate
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return layers.Add()([x, residual])

    def __init__(self, dir_braid, sample_size, tcn_filters, tcn_dilations, dropout, y_neurons, s_neurons):
        name = f'tcnreg_{sample_size}'
        super().__init__(dir_braid, name)

        self._sample_size = sample_size

        inputs = layers.Input(shape=(sample_size, 1))
        x = inputs

        for d in tcn_dilations:
            x = TCNReg.tcn_block(
                x,
                filters=tcn_filters,
                kernel_size=3,
                dilation_rate=d,
                dropout=dropout
            )

        # Axle distances
        y = SelfAttentionBlock(num_heads=4, key_dim=64)(x)
        
        y = layers.Conv1D(filters=1, kernel_size=1, padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)

        y = layers.Flatten()(y)


        y = layers.Dense(256, activation="sigmoid")(y)
        y = layers.Dense(64, activation="sigmoid")(y)
        y = layers.Dense(10, name="distances")(y)

        # Output size
        l = layers.Conv1D(filters=1, kernel_size=1, padding="same")(x)
        l = layers.BatchNormalization()(l)
        l = layers.Activation('relu')(l)

        l = layers.Flatten()(l)


        l = layers.Dense(s_neurons, activation="sigmoid")(l)
        l = layers.Dense(10, name="size")(l)

        outputs = [y, l]
        self._model = models.Model(inputs, outputs)

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                "distances": "mse",
                "size": losses.SparseCategoricalCrossentropy(from_logits=True)
            },
            loss_weights={
                "distances": 1.0,
                "size": 1.0
            }
        )

        plot_model(self._model, to_file='tcnreg_model.png', show_shapes=True)
    
    def train(self, X, Y):
        super().train(X, Y)

        (Y_dist, Y_size) = Y

        # Start recording the results.
        file = open(f'{self._dir_training}loss.txt', 'w')
        file.write('epoch,train_loss,validation_loss\n')

        # Training parameters.
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        print('Evaluating the untrained model.')
        
        # Evaluate the untrained model
        [_, distances_loss, size_loss] = self._model.evaluate(X, Y, verbose=1)
        file.write(f'0,{distances_loss},{size_loss},{distances_loss},{size_loss}\n')

        print('Training the model.')

        # Train the model
        history = self._model.fit(
            X,
            {
                "distances": Y_dist,
                "size": Y_size
            },
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stop],
            verbose=1
        )

        for i, (distances_loss, size_loss, val_distances_loss, val_size_loss) in enumerate(zip(
            history.history['distances_loss'],
            history.history['size_loss'],
            history.history['val_distances_loss'],
            history.history['val_size_loss'])
        ):
            file.write(f'{i+1},{distances_loss},{size_loss},{val_distances_loss},{val_size_loss}\n')
        
        file.close()

    def classification_accuracy(self, kernel_size):
        filename = f'{self._dir_results}predictions.csv'

        if not os.path.exists(filename):
            return
        
        print("Computing the score.")

        cnt = 0
        cnt_correct = 0 
        cnt_sizes_correct = 0
        sum_mae = 0

        axle_confusion = {'tp': 0, 'fp': 0, 'fn': 0}

        f = open(filename)

        header = True
        for line in f:
            if header:
                header = False
                continue

            cols = line.strip().split(",")
            
            y_dist = []
            for i in range(0, 10):
                y_dist.append(float(cols[i]))
            
            y_size = int(cols[10]) + 1

            p_dist = []
            for i in range(11, 21):
                p_dist.append(float(cols[i]))
            
            p_size = int(cols[21]) + 1

            y_dist = np.array(y_dist)[0:y_size]
            p_dist = np.array(p_dist)[0:p_size]

            y_pulses = evaluation.distances_to_pulses(y_dist)
            p_pulses = evaluation.distances_to_pulses(p_dist)

            tp, fn, fp, mae = evaluation.sample_accuracy(y_pulses, p_pulses, 0.5, kernel_size)

            if y_size == p_size:
                cnt_sizes_correct += 1

            if y_size == p_size and fn + fp == 0:
                cnt_correct += 1
            
            cnt += 1

            axle_confusion['tp'] += tp
            axle_confusion['fn'] += fn
            axle_confusion['fp'] += fp

            tp = axle_confusion['tp']
            fn = axle_confusion['fn']
            fp = axle_confusion['fp']

            acc = tp / (tp + fn + fp)
            sum_mae += mae

            if cnt % 1000 == 0:
                print(f'cnt={cnt}, axle accuracy={(tp / (tp + fn + fp)):.4f}, sample accuracy={(cnt_correct / cnt):.4f}, sizes accuracy={(cnt_sizes_correct / cnt):.4f}')

        f.close()

        tp = axle_confusion['tp']
        fn = axle_confusion['fn']
        fp = axle_confusion['fp']

        print('Axle results:')
        print(f'TP: {tp}, FN: {fn}, FP: {fp}')
        print(f'Accuracy: {tp / (tp + fn + fp)}')
        print(f'Mae: {sum_mae / cnt}')
        print('Sample results:')
        print(f'Incorrect samples: {cnt - cnt_correct}')
        print(f'Accuracy: {cnt_correct / cnt}')

        ret_val = ([], tp, fn, fp, cnt_correct, cnt)

    def evaluate(self, X, Y, meta, kernel_size, vehicle_info=None):
        super().evaluate(X, Y, meta)

        (Y_dist, Y_size) = Y

        print('Evaluating the model.')
        (P_dist, P_size) = self._model.predict(X)

        f_predictions = open(f'{self._dir_results}predictions.csv', 'w')
        f_predictions.write('y_0,y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9,y_size,p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9,p_size\n')

        bar = Bar('Saving results', max=len(P_dist))
        
        for (y_dist, y_size, p_dist, p_size) in zip(Y_dist, Y_size, P_dist, P_size):
            first = True
            for value in y_dist:
                if first:
                    f_predictions.write(f'{value:.4f}')
                    first = False
                else:
                    f_predictions.write(f',{value:.4f}')
            
            f_predictions.write(f',{y_size}')

            for value in p_dist:
                f_predictions.write(f',{value:.4f}')
            
            f_predictions.write(f',{np.argmax(p_size)}\n')
            
            bar.next()

        bar.finish()
        f_predictions.close()

        return self.classification_accuracy(kernel_size=kernel_size)