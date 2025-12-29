import os
import shutil

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from progress.bar import Bar
import matplotlib.pyplot as plt

from models.base import BraidModel
import dataset
import visualizations
import evaluation
import losses

class MlpFramesOne(BraidModel):
    def __init__(self, dir_braid, frame_size, architecture):
        name = f'mlp_frames_{frame_size}'
        for layer_size in architecture:
            name += f'_{layer_size}'

        super().__init__(dir_braid, name)

        self._frame_size = frame_size
        self._architecture = architecture

        # Construct the model.
        model = keras.Sequential()
        model.add(layers.Input((frame_size,)))
        for layer_size in architecture:
            model.add(layers.Dense(layer_size, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        # Use weighted loss.
        # wmse = losses.WeightedMSE(positive_weight=1000, negative_weight=1)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

        self._model = model
    
    def train(self, X, Y):
        bar = Bar('Generating frames', max=len(X))

        frames_X_pos = []
        frames_Y_pos = []
        frames_X_neg = []
        frames_Y_neg = []
        for (signal, pulses) in zip(X, Y):
            for (x, y) in dataset.frames_from_signal(signal, pulses, self._frame_size):
                if y >= 0.5: # Could also be 1.
                    frames_X_pos.append(x)
                    frames_Y_pos.append(y)
                else:
                    frames_X_neg.append(x)
                    frames_Y_neg.append(y)
            bar.next()
        bar.finish()

        print(f"Positive instances: {len(frames_X_pos)}.")
        print(f"Negative instances: {len(frames_X_neg)}.")

        print("Balancing the classes.")

        frames_X_pos = np.array(frames_X_pos)
        frames_Y_pos = np.array(frames_Y_pos)
        frames_X_neg = np.array(frames_X_neg)
        frames_Y_neg = np.array(frames_Y_neg)
        
        idx = np.random.permutation(np.arange(len(frames_X_pos)))
        idx = idx[0:len(frames_X_pos)]
        frames_X_neg = frames_X_neg[idx]
        frames_Y_neg = frames_Y_neg[idx]

        print(f"Positive instances: {len(frames_X_pos)}.")
        print(f"Negative instances: {len(frames_X_neg)}.")

        X = np.concatenate((frames_X_pos, frames_X_neg))
        Y = np.concatenate((frames_Y_pos, frames_Y_neg))

        idx = np.random.permutation(np.arange(len(X)))
        X = X[idx]
        Y = Y[idx]

        print(f"All instances: {len(X)}.")

        print("training the model.")

        # Use weighted loss.
        #wmse = losses.WeightedMSE(positive_weight=1000, negative_weight=1)

        # Start recording the results.
        file = open(f'{self._dir_training}{self._name}', 'w')
        file.write('epoch,train_loss,validation_loss\n')

        # Training parameters.
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        # Evaluate the untrained model
        #loss, _ = self._model.evaluate(X, Y, verbose=0)
        loss = 0
        file.write(f'0,{loss},{loss}\n')

        # Train the model
        history = self._model.fit(
            X, Y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        for i, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            file.write(f'{i+1},{train_loss},{val_loss}\n')
        
        file.close()

    def evaluate(self, X, Y, meta, class_threshold, kernel_size, plots=False):
        bar = Bar('Making predictions', max=len(X))

        predictions = []
        for i, (signal, pulses) in enumerate(zip(X, Y)):
            frames_X = []
            frames_Y = []
            for (x, y) in dataset.frames_from_signal(signal, pulses, self._frame_size, include_empty=True):
                frames_X.append(x)
                frames_Y.append(y)

            frames_X = np.array(frames_X)
            frames_Y = np.array(frames_Y)

            frames = [frame for frame in zip(frames_X, self._model.predict(frames_X))]
            (_, new_pulses) = dataset.signal_from_frames(frames)
            predictions.append(new_pulses)

            bar.next()
        bar.finish()

        dir_plots = f'{self._dir_plots}validation/'
        if not os.path.exists(dir_plots):
            os.makedirs(dir_plots)

        self.confusion = {'tp': 0, 'fp': 0, 'fn': 0}
        thrs_t = []
        thrs_f = []

        file = open(f'{self._dir_results}{self._name}', 'w')
        bar = Bar('Evaluating model', max=len(predictions))

        cnt = 0
        sum_mae = 0
        for (signal, pulses, prediction) in zip(X, Y, predictions):
            if plots and cnt % 10 == 0:
                visualizations.plot_sample(f'{dir_plots}{cnt}.png', signal, pulses, prediction)

            prediction = evaluation.sharpen_prediction(prediction)

            tp, fn, fp, mae = evaluation.sample_accuracy(pulses, prediction, class_threshold, kernel_size)
            cnt += 1

            self.confusion['tp'] += tp
            self.confusion['fn'] += fn
            self.confusion['fp'] += fp

            tp = self.confusion['tp']
            fn = self.confusion['fn']
            fp = self.confusion['fp']

            acc = tp / (tp + fn + fp)
            sum_mae += mae

            if cnt % 10 == 0:
                bar.suffix = f'Complete: {bar.percent:3.0f}/100 | AC/MAE: {(100 * acc):3.2f}/{(sum_mae / cnt):.6f}'

            thrs_t_1, thrs_f_1 = evaluation.sample_thresholds(pulses, prediction, kernel_size)

            thrs_t = thrs_t + thrs_t_1
            thrs_f = thrs_f + thrs_f_1

            bar.next()
        
        bar.finish()
        file.close()

        tp = self.confusion['tp']
        fn = self.confusion['fn']
        fp = self.confusion['fp']

        print(f'TP: {tp}, FN: {fn}, FP: {fp}')
        print(f'Accuracy: {tp / (tp + fn + fp)}')

        plt.hist(thrs_t, bins=100, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'True confidence values ({len(thrs_t)})')
        plt.savefig(f'{self._dir_braid}hist_confidence_true.png')

        plt.clf()

        plt.hist(thrs_f, bins=100, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'False confidence values ({len(thrs_f)})')
        plt.savefig(f'{self._dir_braid}hist_confidence_false.png')

        plt.clf()

        plt.hist(thrs_t + thrs_f, bins=100, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'Confidence values ({len(thrs_t) + len(thrs_f)})')
        plt.savefig(f'{self._dir_braid}hist_confidence.png')