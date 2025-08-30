import os
import dataset
import training
import evaluation

dir_braid = '/home/hicup/disk/braid/'
input_signal = '11admp'

# Check if the database can be opened.
if os.path.exists(f'{dir_braid}signals/nn_signals.hdf5'):
    print('Found file nn_signals.hdf5')
else:
    print('File nn_signals.hdf5 not found!')
    quit()

if os.path.exists(f'{dir_braid}signals/nn_pulses.json'):
    print('Found file nn_pulses.json')
else:
    print('File nn_pulses.json not found!')
    quit()

# Check if the training set exists.
generate_training_set = False
if os.path.exists(f'{dir_braid}/signals_training_x.npy'):
    print('Found signals_training_x.npy')
else:
    print('File signals_training_x.npy has not been found.')
    generate_training_set = True

if os.path.exists(f'{dir_braid}/signals_training_y.npy'):
    print('Found signals_training_y.npy')
else:
    print('File signals_training_y.npy has not been found.')
    generate_training_set = True

if generate_training_set:
    X, Y, timestamps, axle_groups = dataset.generate_training_samples(dir_braid, 2048, input_signal)

    print('Plotting the training data.')
    if not os.path.exists(f'{dir_braid}signals/plots/'):
        os.mkdir(f'{dir_braid}signals/plots/')
    
    cnt = 0
    for x, y, ts, groups in zip(X, Y, timestamps, axle_groups):
        dataset.plot_sample(f'{dir_braid}signals/plots/{ts}.png', x, y, ts, groups)
        cnt += 1
        if cnt % 1000 == 0:
            print(f'Finished {cnt}/{len(timestamps)}.')
    print(f'Finished {len(timestamps)}.')

else:
    X, Y = dataset.load_training_samples(dir_braid)

# Split the dataset
X_train, Y_train, X_validate, Y_validate = dataset.split_samples(X, Y)
print(f'The number of training samples: {len(X_train)}')
print(f'The number of validation samples: {len(X_validate)}')

# Train the model
# model = training.train_mlp(2048, [8192, 4096], X_train, Y_train)
model = training.train_cnn(2048, X_train, Y_train)

# Evaluate the model
evaluation.evaluate(model, X_validate, Y_validate)