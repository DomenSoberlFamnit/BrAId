import os
import numpy as np

import dataset
from models.tcn import TCN

dir_braid = '/mnt/workspace/braid/workdir/'
signal_length = 1300

force_training = True

#####################   Data preparation   ########################################

# Load or generate the vehicle index (a leftover from the camera project).
vehicle_info = dataset.load_vehicle_info(dir_braid, f'{dir_braid}camera/')
if vehicle_info is None:
    print('Cannot load vehicle info.')
    quit()

data = dataset.get_data(
    dir_braid=dir_braid,
    input_signal='11admp',
    output_type='pulses',
    normalized_signals=True,
    include_correct=True,
    include_fixed=True,
    signal_length=signal_length,
    shuffle=True
)

########################################   Evaluate Model   ########################################

fold_n = 10
results = []

for fold_k in range(fold_n):
    data_train, data_test = dataset.split_fold(data, fold_k, fold_n)

    (meta_train, X_train, Y_train) = data_train
    (meta_test, X_test, Y_test) = data_test

    model = TCN(dir_braid, signal_length, tcn_filters=64, tcn_dilations=[1, 2, 3, 4, 5, 6, 7], dropout=0.1)
    
    model.train(X_train, Y_train)
    measurements = model.evaluate(X_test, Y_test, meta_test, class_threshold=0.2, kernel_size=9, vehicle_info=vehicle_info)
    results.append(measurements)

    f = open('tcn-cross-validation-results.txt', 'w')
    for (i, measurements) in enumerate(results):
        (_, tp, fn, fp, cnt_correct, cnt) = measurements
        f.write(f'{i},{tp},{fn},{fp},{cnt_correct},{cnt}\n')
    f.close()
