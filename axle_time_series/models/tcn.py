import os

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from progress.bar import Bar
import matplotlib.pyplot as plt

from models.base import BraidModel
import dataset
import visualizations
import evaluation
import losses

class TCN(BraidModel):
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

    def __init__(self, dir_braid, sample_size, tcn_filters, tcn_dilations, dropout):
        name = f'tcn_{sample_size}'
        super().__init__(dir_braid, name)

        self._sample_size = sample_size

        inputs = layers.Input(shape=(sample_size, 1))
        x = inputs

        for d in tcn_dilations:
            x = TCN.tcn_block(
                x,
                filters=tcn_filters,
                kernel_size=3,
                dilation_rate=d,
                dropout=dropout
            )

        x = layers.Conv1D(filters=1, kernel_size=1, padding="same", activation="sigmoid")(x)
        x = layers.Flatten()(x)

        outputs = x
        
        self._model = models.Model(inputs, outputs)
        self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    
    def train(self, X, Y):
        super().train(X, Y)

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
        loss, _ = self._model.evaluate(X, Y, verbose=1)
        file.write(f'0,{loss},{loss}\n')

        print('Training the model.')

        # Train the model
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

        for i, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            file.write(f'{i+1},{train_loss},{val_loss}\n')
        
        file.close()

    def evaluate(self, X, Y, meta, class_threshold, kernel_size, plots=False, vehicle_info=None):
        super().evaluate(X, Y, meta)

        print('Evaluating the model.')
        predictions = self._model.predict(X)

        self.axle_confusion = {'tp': 0, 'fp': 0, 'fn': 0}
        self.groups_confusion = {}

        thrs_t = []
        thrs_f = []

        if plots:
            dir_plots_neg = f'{self._dir_plots}validation/negative/'
            if os.path.exists(dir_plots_neg):
                shutil.rmtree(dir_plots_neg)
            os.makedirs(dir_plots_neg)

        file = open(f'{self._dir_results}incorrect_classifications.txt', 'w')

        bar = Bar('Evaluating model', max=len(predictions))

        cnt = 0
        cnt_correct = 0 
        sum_mae = 0
        for (signal, pulses, m, prediction) in zip(X, Y, meta, predictions):
            ts = m[0]
            groups = m[2]

            if groups not in self.groups_confusion:
                self.groups_confusion[groups] = {'positive': 0, 'negative': 0}

            filtered_prediction = evaluation.max_filter(prediction, threshold=class_threshold, kernel_size=9)
            tp, fn, fp, mae = evaluation.sample_accuracy(pulses, filtered_prediction, class_threshold, kernel_size)
            
            if fn + fp == 0:
                self.groups_confusion[groups]['positive'] += 1
                cnt_correct += 1
            else:
                self.groups_confusion[groups]['negative'] += 1
                file.write(f'{ts},{groups},truth,')
                for pulse in pulses:
                    file.write(f',{pulse}')
                file.write(f'\n{ts},{groups},prediction,')
                for pulse in prediction:
                    file.write(f',{pulse}')
                file.write(f'\n{ts},{groups},maxpool,')
                for pulse in filtered_prediction:
                    file.write(f',{pulse}')
                file.write('\n\n')

            if plots:
                siwim_correct = vehicle_info[ts]['siwim_correct']
                ai_correct = (fn + fp) == 0

                metadata = {
                        'ts': ts,
                        'groups': groups,
                        'siwim': siwim_correct,
                        'ai': ai_correct,
                        'missed': fn,
                        'ghost': fp,
                        'image': vehicle_info[ts]['photo']
                    }

                filename = f'{self._dir_plots}{vehicle_info[ts]['id']}_{'T' if siwim_correct else 'F'}{'T' if ai_correct else 'F'}.png'
                visualizations.plot_testing_sample(filename, signal, pulses, filtered_prediction, class_threshold, metadata)
                
            cnt += 1

            self.axle_confusion['tp'] += tp
            self.axle_confusion['fn'] += fn
            self.axle_confusion['fp'] += fp

            tp = self.axle_confusion['tp']
            fn = self.axle_confusion['fn']
            fp = self.axle_confusion['fp']

            acc = tp / (tp + fn + fp)
            sum_mae += mae

            if cnt % 10 == 0:
                bar.suffix = f'Complete: {bar.percent:3.0f}/100 | AC axles/samples: {(100 * acc):3.2f}/{(100 * cnt_correct / cnt):3.2f}'

            thrs_t_1, thrs_f_1 = evaluation.sample_thresholds(pulses, prediction, kernel_size)

            thrs_t = thrs_t + thrs_t_1
            thrs_f = thrs_f + thrs_f_1

            bar.next()
        
        bar.finish()
        file.close()

        tp = self.axle_confusion['tp']
        fn = self.axle_confusion['fn']
        fp = self.axle_confusion['fp']

        print('Axle results:')
        print(f'TP: {tp}, FN: {fn}, FP: {fp}')
        print(f'Accuracy: {tp / (tp + fn + fp)}')
        print(f'Mae: {sum_mae / cnt}')
        print('Sample results:')
        print(f'Accuracy: {cnt_correct / cnt}')

        ret_val = (tp, fn, fp, cnt_correct, cnt)
        return ret_val

        file = open(f'{self._dir_results}metrics.txt', 'w')
        file.write('Axle results:\n')
        file.write(f'- TP: {tp}, FN: {fn}, FP: {fp}\n')
        file.write(f'- Accuracy: {tp / (tp + fn + fp)}\n')
        file.write(f'- Mae: {sum_mae / cnt}\n\n')
        file.write('Sample results:\n')
        file.write(f'- Accuracy: {cnt_correct / cnt}\n')
        for groups in self.groups_confusion:
            pos = self.groups_confusion[groups]['positive']
            neg = self.groups_confusion[groups]['negative']
            file.write(f'{groups}: positive = {pos}, negative = {neg}, accuracy = {100 * pos / (pos + neg):.2f}\n')
        file.close()

        plt.hist(thrs_t, bins=100, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'True confidence values ({len(thrs_t)})')
        plt.savefig(f'{self._dir_evaluation}hist_confidence_true.png')

        plt.clf()

        plt.hist(thrs_f, bins=100, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'False confidence values ({len(thrs_f)})')
        plt.savefig(f'{self._dir_evaluation}hist_confidence_false.png')

        plt.clf()

        plt.hist(thrs_t + thrs_f, bins=100, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title(f'Confidence values ({len(thrs_t) + len(thrs_f)})')
        plt.savefig(f'{self._dir_evaluation}hist_confidence.png')

        return ret_val
