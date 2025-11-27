import os
import numpy as np

import dataset
import baseline
import models

dir_braid = '/mnt/workspace/braid/workdir/'
input_signal = '11admp'

normalized_signals = True
signal_length = 1300 if normalized_signals else 2048 

########################################   Data preparation   ########################################

# Check if the database can be opened.
if not dataset.signal_data_found(dir_braid, normalized_signals):
    quit()

# Analyze the dataset.
base_ca = baseline.accuracy(dir_braid, normalized_signals)
print(f'Baseline accuracy: {100 * base_ca:.2f}')

# Try to load the training set and the validation set.
training_set = dataset.load_training_samples(dir_braid)
validation_set = dataset.load_validation_samples(dir_braid)

# If the split data cannot be loaded, generate it.
if training_set is None or validation_set is None:
    samples = dataset.load_samples(dir_braid)

    # If samples cannot be loaded, generate them.
    if samples is None:
        X, Y, timestamps, axle_groups = dataset.generate_training_samples(dir_braid, signal_length, input_signal, normalized_signals)

        print('Plotting the training data.')
        if not os.path.exists(f'{dir_braid}plots/raw/'):
            os.makedirs(f'{dir_braid}plots/raw/')
        
        cnt = 0
        for x, y, ts, groups in zip(X, Y, timestamps, axle_groups):
            dataset.plot_sample(f'{dir_braid}plots/raw/{ts}.png', x, y, ts, groups)
            cnt += 1
            if cnt % 1000 == 0:
                print(f'Finished {cnt}/{len(timestamps)}.')
        print(f'Finished {len(timestamps)}.')

    else:
        X, Y = samples
    
    # Now we have the samples, split them.
    X_train, Y_train, X_validate, Y_validate = dataset.split_samples(X, Y)

    print(f'Saving {dir_braid}signals_training_x.npy')
    np.save(f'{dir_braid}signals_training_x.npy', X_train)
    
    print(f'Saving {dir_braid}signals_training_y.npy')
    np.save(f'{dir_braid}signals_training_y.npy', Y_train)

    print(f'Saving {dir_braid}signals_validation_x.npy')
    np.save(f'{dir_braid}signals_validation_x.npy', X_validate)
    
    print(f'Saving {dir_braid}signals_validation_y.npy')
    np.save(f'{dir_braid}signals_validation_y.npy', Y_validate)

# Training and validation samples have been loaded.
else:
    X_train, Y_train = training_set
    X_validate, Y_validate = validation_set

print(f'The number of training samples: {len(X_train)}')
print(f'The number of validation samples: {len(X_validate)}')

########################################   Evaluate MLP   ########################################

# MLP Raw
if False:
    model = models.MlpRaw(dir_braid, signal_length, [8192, 4096])
    model.print()

    if not model.load():
        model.train(X_train, Y_train)
        model.save()

    model.evaluate(X_validate, Y_validate, class_threshold=0.1, kernel_size=7, plots=True)

# MLP Frame
if True:
    model = models.MlpFrames(dir_braid, 64, [8192, 4096])
    model.print()

    if not model.load():
        model.train(X_train, Y_train)
        model.save()

    model.evaluate(X_validate, Y_validate, class_threshold=0.1, kernel_size=7, plots=True)

