import os
import numpy as np

import dataset
from models.tcn import TCN
from models.cnn import CNN

dir_braid = '/mnt/workspace/braid/workdir/'
signal_length = 1300

force_training = True

#####################   Data preparation   ########################################

data = dataset.get_data(
    dir_braid=dir_braid,
    input_signal='11admp',
    output_type='pulses',
    normalized_signals=True,
    include_correct=True,
    include_fixed=True,
    signal_length=signal_length,
    shuffle=False
)

quit()

data_train, data_test = dataset.split_data(data, testing_ratio=0.3)

(meta_train, X_train, Y_train) = data_train
(meta_test, X_test, Y_test) = data_test

########################################   Evaluate Models   ########################################

# CNN
if False:
    model = CNN(dir_braid, signal_length, filters=64)
    model.print()

    if force_training or not model.load():
        model.train(X_train, Y_train)
        model.save()

    model.evaluate(X_test, Y_test, meta_test, class_threshold=0.1, kernel_size=3, plots=False)

# Hyper-parameter fine-tuning for CNN
if False:
    model = CNN(dir_braid, signal_length, filters=32)
    model.print()

    model.train(X_train, Y_train)

    results = []
    for threshold in np.arange(0.01, 0.99, 0.01):
        print(f'Threshold: {threshold}')
        measurements = model.evaluate(X_test, Y_test, meta_test, class_threshold=threshold, kernel_size=3, plots=False)
        results.append((threshold, measurements))
    
    f = open('cnn-threshold-results.txt', 'w')
    for (threshold, measurements) in results:
        (tp, fn, fp, cnt_correct, cnt) = measurements
        f.write(f'{threshold},{tp},{fn},{fp},{cnt_correct},{cnt}\n')
    f.close()

# TCN
if False:
    model = TCN(dir_braid, signal_length, tcn_filters=64, tcn_dilations=[1, 2, 3, 4, 5, 6, 7], dropout=0.1)
    model.print()

    if force_training or not model.load():
        model.train(X_train, Y_train)
        model.save()

    model.evaluate(X_test, Y_test, meta_test, class_threshold=0.2, kernel_size=3, plots=False, vehicle_info=vehicle_info)

# TCN threshold optimization
if True:
    model = TCN(dir_braid, signal_length, tcn_filters=64, tcn_dilations=[1, 2, 3, 4, 5, 6, 7], dropout=0.1)
    model.print()

    model.train(X_train, Y_train)

    results = []
    for threshold in np.arange(0.1, 0.5, 0.01):
        print(f'Threshold: {threshold}')
        measurements = model.evaluate(X_test, Y_test, meta_test, class_threshold=threshold, kernel_size=3, plots=False)
        results.append((threshold, measurements))
    
    f = open('tcn-threshold-results.txt', 'w')
    for (threshold, measurements) in results:
        (tp, fn, fp, cnt_correct, cnt) = measurements
        f.write(f'{threshold},{tp},{fn},{fp},{cnt_correct},{cnt}\n')
    f.close()
