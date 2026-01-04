import os
import numpy as np

import dataset
from models.tcn import TCN

dir_braid = '/mnt/workspace/braid/workdir/'
signal_length = 1300

force_training = True

#####################   Data preparation   ########################################

data = dataset.get_data(
    dir_braid=dir_braid,
    input_signal='11admp',
    normalized_signals=True,
    include_correct=True,
    include_fixed=False,
    signal_length=signal_length,
    shuffle=True
)

(meta_train, X_train, Y_train) = data_train
(meta_test, X_test, Y_test) = data_test

########################################   Evaluate Model   ########################################

fold_n = 10
results = []

for fold_k in range(fold_n):
    data_train, data_test = split_fold(data, fold_k, fold_n)

    (meta_train, X_train, Y_train) = data_train
    (meta_test, X_test, Y_test) = data_test

    model = TCN(dir_braid, signal_length, tcn_filters=64, tcn_dilations=[1, 2, 3, 4, 5, 6, 7], dropout=0.1)
    
    model.train(X_train, Y_train)
    measurements = model.evaluate(X_test, Y_test, meta_test, class_threshold=0.2, kernel_size=3, plots=False, vehicle_info=vehicle_info)
    results.append(measurements)

    f = open('tcn-cross-validation-results.txt', 'w')
    for (i, measurements) in enumerate(results):
        (tp, fn, fp, cnt_correct, cnt) = measurements
        f.write(f'{i},{tp},{fn},{fp},{cnt_correct},{cnt}\n')
    f.close()
