import os

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from progress.bar import Bar
import matplotlib.pyplot as plt
import evaluation

class TCNMult:
    def tcn_block(x, filters, kernel_size, dilation_rate, dropout):
        (_, signals_count, signal_length, channels_count) = x.shape

        residual = x

        outputs = []
        for i in range(signals_count):
            signal = x[:,i,:,:]

            signal = layers.Conv1D(
                filters,
                kernel_size,
                padding="same",
                dilation_rate=dilation_rate
            )(signal)

            signal = layers.BatchNormalization()(signal)
            signal = layers.Activation('relu')(signal)

            signal = layers.Dropout(dropout)(signal)

            signal = layers.Conv1D(
                filters,
                kernel_size,
                padding="same",
                dilation_rate=dilation_rate
            )(signal)

            signal = layers.BatchNormalization()(signal)
            signal = layers.Activation('relu')(signal)
            
            signal = layers.Reshape((1, signal_length, filters))(signal)
            outputs.append(signal)

        x = layers.Concatenate(axis=1)(outputs)

        return layers.Add()([x, residual])

    def cnn_expand(x, filters, kernel_size):
        (_, signals_count, signal_length, channels_count) = x.shape

        outputs = []
        for i in range(signals_count):
            signal = x[:,i,:,:]

            signal = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same"
            )(signal)

            signal = layers.BatchNormalization()(signal)
            signal = layers.Activation('relu')(signal)
            
            signal = layers.Reshape((1, signal_length, filters))(signal)
            outputs.append(signal)

        x = layers.Concatenate(axis=1)(outputs)

        return x
    
    def cnn_reduce(x, kernel_size):
        (_, signals_count, signal_length, channels_count) = x.shape

        outputs = []
        for i in range(signals_count):
            signal = x[:,i,:,:]

            signal = layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same"
            )(signal)

            signal = layers.BatchNormalization()(signal)
            signal = layers.Activation('relu')(signal)
            
            signal = layers.Reshape((1, signal_length, 1))(signal)
            outputs.append(signal)

        x = layers.Concatenate(axis=1)(outputs)

        return x

    def head(x):
        (_, signals_count, signal_length, channels_count) = x.shape

        outputs = []
        for i in range(signal_length):
            location = x[:,:,i,:]
            location = layers.Flatten()(x)
            location = layers.Dense(1, activation='sigmoid')(location)
            outputs.append(location)
        
        x = layers.Concatenate(axis=1)(outputs)

        return x

    def __init__(self,
        dir_braid,
        signal_count,
        tcn_filters,
        tcn_dilations, 
        dropout
    ):
        self._dir_braid = dir_braid
        name = f'tcn_multiple'
        
        inputs = layers.Input(shape=(signal_count, 1500, 1))
        x = inputs

        x = TCNMult.cnn_expand(x, filters=tcn_filters, kernel_size=3)

        for d in tcn_dilations:
            x = TCNMult.tcn_block(
                x,
                filters=tcn_filters,
                kernel_size=3,
                dilation_rate=d,
                dropout=dropout
            )

        x = TCNMult.cnn_reduce(x, kernel_size=3)
        x = TCNMult.head(x)

        outputs = x
        self._model = models.Model(inputs, outputs)
        self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
        
        #self._model.summary()
        #plot_model(self._model, to_file=f'{dir_braid}tcn_multiple_model.png', show_shapes=True)
    
    def train(self, X, Y):
        dir_training = f'{self._dir_braid}training/'
        if not os.path.exists(dir_training):
            os.mkdir(dir_training)

        file = open(f'{dir_training}tcn_loss.txt', 'w')
        file.write('epoch,train_loss,validation_loss\n')

        print('Evaluating the untrained model.')
        loss, _ = self._model.evaluate(X, Y, verbose=1)
        file.write(f'0,{loss},{loss}\n')
        print(f'loss = {loss:.4f}')

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        print('Started training the model.')

        history = self._model.fit(
            X, Y,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            #class_weight={0: 0.7, 1: 5.5},
            callbacks=[early_stop],
            verbose=1
        )

        print('Training done.')

        for i, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            file.write(f'{i+1},{train_loss},{val_loss}\n')

        file.close()

    def evaluate(self, X, Y, meta, class_threshold, kernel_size):
        print('Evaluating the model.')

        dir_evaluation = f'{self._dir_braid}evaluation/'
        if not os.path.exists(dir_evaluation):
            os.mkdir(dir_evaluation)

        Yp = self._model.predict(X)

        cnt = 0
        cnt_correct = 0 
        sum_mae = 0

        axle_confusion = {'tp': 0, 'fp': 0, 'fn': 0}

        thrs = []

        bar = Bar('Evaluating model', max=len(Yp))
        for (signal, pulses, m, prediction) in zip(X, Y, meta, Yp):
            filtered_prediction = evaluation.max_filter(prediction, threshold=class_threshold, kernel_size=kernel_size)
            tp, fn, fp, mae = evaluation.sample_accuracy(pulses, filtered_prediction, class_threshold, kernel_size)

            for value in prediction:
                if value < 0.1:
                    continue
                thrs.append(value)

            cnt += 1

            if fn + fp == 0:
                cnt_correct += 1
            else:
                pass

            axle_confusion['tp'] += tp
            axle_confusion['fn'] += fn
            axle_confusion['fp'] += fp

            tp = axle_confusion['tp']
            fn = axle_confusion['fn']
            fp = axle_confusion['fp']

            acc = tp / (tp + fn + fp)
            sum_mae += mae

            if cnt % 10 == 0:
                bar.suffix = f'Complete: {bar.percent:3.0f}/100 | AC axles/samples: {(100 * acc):3.2f}/{(100 * cnt_correct / cnt):3.2f}'

            bar.next()
        bar.finish()

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

        plt.hist(thrs, bins=100, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'Confidence values ({len(thrs)})')
        plt.savefig(f'{dir_evaluation}hist_confidence.png')
