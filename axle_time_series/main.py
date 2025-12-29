import os
import numpy as np

import dataset
import baseline
from models.mlp_raw import MlpRaw
from models.mlp_frames_one import MlpFramesOne
from models.tcn import TCN
from models.cnn import CNN

from progress.bar import Bar

dir_braid = '/mnt/workspace/braid/workdir/'
input_signal = '11admp'

normalized_signals = True
signal_length = 1300 if normalized_signals else 2048 

plot_data = False
force_training = True

#####################= self._model.evaluate(X, Y, verbose=1)###################   Data preparation   ########################################

# Check if the database can be opened.
if not dataset.signal_data_found(dir_braid, normalized_signals):
    quit()

# Analyze the dataset.
base_ca = baseline.accuracy(dir_braid, normalized_signals)
print(f'Baseline accuracy: {100 * base_ca:.2f}')

# Try to load the training set and the testing set.
training_data = dataset.load_training_samples(dir_braid)
testing_data = dataset.load_testing_samples(dir_braid)

# If the split data cannot be loaded, generate it.
if training_data is None or testing_data is None:
    samples = dataset.load_samples(dir_braid)

    # If samples cannot be loaded, generate them.
    if samples is None:
        meta, X, Y = dataset.generate_training_samples(dir_braid, signal_length, input_signal, normalized_signals)

        if plot_data:
            print('Plotting the training data.')
            if not os.path.exists(f'{dir_braid}plots/raw/'):
                os.makedirs(f'{dir_braid}plots/raw/')

            bar = Bar('Plotting', max=len(meta))
            cnt = 0
            for x, y, m in zip(X, Y, meta):
                ts, groups = str(m[0]), str(m[1])
                dataset.plot_sample(f'{dir_braid}plots/raw/{ts}.png', x, y, ts, groups)
                cnt += 1
                bar.next()
            bar.finish()

    else:
        meta, X, Y = samples
    
    # Now we have the samples, split them.
    meta_train, X_train, Y_train, meta_test, X_test, Y_test = dataset.split_samples(meta, X, Y, testing_size=0.3)

    print(f'Saving {dir_braid}meta_training.npy')
    np.save(f'{dir_braid}meta_training.npy', meta_train)

    print(f'Saving {dir_braid}meta_testing.npy')
    np.save(f'{dir_braid}meta_testing.npy', meta_test)

    print(f'Saving {dir_braid}signals_training_x.npy')
    np.save(f'{dir_braid}signals_training_x.npy', X_train)
    
    print(f'Saving {dir_braid}signals_training_y.npy')
    np.save(f'{dir_braid}signals_training_y.npy', Y_train)

    print(f'Saving {dir_braid}signals_testing_x.npy')
    np.save(f'{dir_braid}signals_testing_x.npy', X_test)
    
    print(f'Saving {dir_braid}signals_testing_y.npy')
    np.save(f'{dir_braid}signals_testing_y.npy', Y_test)

# Training and testing samples have been loaded.
else:
    meta_train, X_train, Y_train = training_data
    meta_test, X_test, Y_test = testing_data

print(f'The number of training samples: {len(meta_train)}')
print(f'The number of testing samples: {len(meta_test)}')

########################################   Evaluate MLP   ########################################

# MLP Raw
if False:
    model = MlpRaw(dir_braid, signal_length, [1024, 512, 1024])
    #model = MlpRaw(dir_braid, signal_length, [1024, 512, 256, 64, 256, 512, 1024])
    #model = MlpRaw(dir_braid, signal_length, [1024, 512, 256, 512, 1024])
    model.print()

    if force_training or not model.load():
        model.train(X_train, Y_train)
        model.save()

    model.evaluate(X_test, Y_test, meta_test, class_threshold=0.1, kernel_size=11, plots=False)

# MLP Frames
if False:
    model = MlpFramesOne(dir_braid, 64, [8192, 4096])
    model.print()

    if force_training or not model.load():
        model.train(X_train, Y_train)
        model.save()

    model.evaluate(X_test, Y_test, meta_test, class_threshold=0.1, kernel_size=9, plots=False)

# CNN
if False:
    model = CNN(dir_braid, signal_length, filters=64)
    model.print()

    if force_training or not model.load():
        model.train(X_train, Y_train)
        model.save()

    model.evaluate(X_test, Y_test, meta_test, class_threshold=0.1, kernel_size=3, plots=False)

# Hyper-parameter fine-tuning for CNN
if True:
    model = CNN(dir_braid, signal_length, filters=32)
    model.print()

    model.train(X_train, Y_train)

    results = []
    for threshold in np.arange(0.01, 0.99, 0.01):
        print(f'Threshold: {threshold}')
        measurements = model.evaluate(X_test, Y_test, meta_test, class_threshold=threshold, kernel_size=3, plots=False)
        results.append((threshold, measurements))
    
    f = open('threshold-results.txt', 'w')
    for (threshold, measurements) in results:
        (tp, fn, fp, cnt_correct, cnt) = measurements
        f.write(f'{threshold},{tp},{fn},{fp},{cnt_correct},{cnt}\n')
    f.close()

# TCN
if False:
    model = TCN(dir_braid, signal_length, cnn_filters=32, tcn_filters=64, tcn_dilations=[1, 2, 4, 8])
    model.print()

    if force_training or not model.load():
        model.train(X_train, Y_train)
        model.save()

    model.evaluate(X_test, Y_test, meta_test, class_threshold=0.1, kernel_size=9, plots=False)