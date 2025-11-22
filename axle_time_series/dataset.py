import os
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def signal_data_found(dir_braid):
    if os.path.exists(f'{dir_braid}data/nn_signals.hdf5'):
        print('Found file nn_signals.hdf5')
    else:
        print('File nn_signals.hdf5 not found!')
        return False

    if os.path.exists(f'{dir_braid}data/nn_pulses.json'):
        print('Found file nn_pulses.json')
    else:
        print('File nn_pulses.json not found!')
        return False
    
    return True

def generate_training_samples(dir_braid, signal_size, signal_name):
    print(f'Generating the training dataset for signal {signal_name}.')

    X = []
    Y = []

    lengths = []
    locations = []
    axle_groups = []
    timestamps = []

    with h5py.File(f'{dir_braid}data/nn_signals.hdf5', 'r') as signals:
        with open(f'{dir_braid}data/nn_pulses.json', 'r') as file:
            pulses = json.load(file)

        cnt = 0

        for pulse in pulses:
            photo_match = pulse['photo_match']

            if not photo_match:
                continue

            cnt += 1

            # Get metadata.
            ts = pulse['ts_str']
            timestamps.append(ts)
            axle_groups.append(pulse['vehicle']['final']['axle_groups'])

            # Construct matrix X.
            raw_signal = np.array(signals[ts][signal_name])
            raw_size = len(raw_signal)

            lengths.append(raw_size)

            signal = np.zeros(signal_size)
            if raw_size <= signal_size:
                signal[0:raw_size] = raw_signal
            else:
                signal = raw_signal[0:signal_size]
            
            X.append(signal)

            # Construct matrix Y.
            raw_pulses = pulse['vehicle']['detected']['axle_pulses']

            pulses = np.zeros(signal_size)
            for location in raw_pulses:
                pulses[location] = 1
                locations.append(location)
        
            Y.append(pulses)
        
        X = np.array(X)
        Y = np.array(Y)

        print(f'Found {cnt} training samples.')
        print(f'Signal lengths are between {np.min(lengths)} and {np.max(lengths)}.')
        print(f'Pulses are located between {np.min(locations)} and {np.max(locations)}.')

        print(f'Saving {dir_braid}signals_x.npy')
        np.save(f'{dir_braid}signals_x.npy', X)
        
        print(f'Saving {dir_braid}signals_y.npy')
        np.save(f'{dir_braid}signals_y.npy', Y)

        return X, Y, timestamps, axle_groups

def load_samples(dir_braid):
    # Check if the samples set exists.
    training_set_found = True
    if os.path.exists(f'{dir_braid}/signals_x.npy'):
        print('Found signals_x.npy')
    else:
        print('File signals_x.npy has not been found.')
        training_set_found = False

    if os.path.exists(f'{dir_braid}/signals_y.npy'):
        print('Found signals_y.npy')
    else:
        print('File signals_y.npy has not been found.')
        training_set_found = False

    if training_set_found:
        print('Loading samples.')
        X = np.load(f'{dir_braid}signals_x.npy')
        Y = np.load(f'{dir_braid}signals_y.npy')
        return X, Y
    
    return None

def load_training_samples(dir_braid):
    # Check if the training set exists.
    training_set_found = True
    if os.path.exists(f'{dir_braid}/signals_training_x.npy'):
        print('Found signals_training_x.npy')
    else:
        print('File signals_training_x.npy has not been found.')
        training_set_found = False

    if os.path.exists(f'{dir_braid}/signals_training_y.npy'):
        print('Found signals_training_y.npy')
    else:
        print('File signals_training_y.npy has not been found.')
        training_set_found = False

    if training_set_found:
        print('Loading training samples.')
        X = np.load(f'{dir_braid}signals_training_x.npy')
        Y = np.load(f'{dir_braid}signals_training_y.npy')
        return X, Y
    
    return None

def load_validation_samples(dir_braid):
    # Check if the training set exists.
    validation_set_found = True
    if os.path.exists(f'{dir_braid}/signals_validation_x.npy'):
        print('Found signals_validation_x.npy')
    else:
        print('File signals_validation_x.npy has not been found.')
        validation_set_found = False

    if os.path.exists(f'{dir_braid}/signals_validation_y.npy'):
        print('Found signals_validation_y.npy')
    else:
        print('File signals_validation_y.npy has not been found.')
        validation_set_found = False

    if validation_set_found:
        print('Loading validation samples.')
        X = np.load(f'{dir_braid}signals_validation_x.npy')
        Y = np.load(f'{dir_braid}signals_validation_y.npy')
        return X, Y
    
    return None

def split_samples(X, Y, validation_size=0.2):
    indices = np.arange(0, len(X))
    indices = np.random.permutation(indices)
    split_idx = round(len(X) * validation_size)
    validate_idx = indices[0:split_idx]
    train_idx = indices[split_idx:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_validate = X[validate_idx]
    Y_validate = Y[validate_idx]

    return X_train, Y_train, X_validate, Y_validate

def plot_sample(filename, signal, pulses, timestamp, axle_groups):
    bars = []
    cnt = 0
    for pulse in pulses:
        if pulse > 0:
            bars.append(cnt)
        cnt += 1

    maxy = np.max(signal)

    fig, ax = plt.subplots()

    ax.plot(signal)
    ax.vlines(x=[x for x in bars], ymin=-maxy/10, ymax=0, color='r')
    ax.axhline(color='k', linewidth=0.5, linestyle='--')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(512))
    ax.text(0, 1.1 * maxy, f'Axle groups: {axle_groups}    timestamp: {timestamp}', fontsize=10, color="black")

    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    X, Y, timestamps, axle_groups = generate_training_samples('.', 2048, '11admp')
    plot_sample('sample.png', X[0], Y[0], timestamps[0], axle_groups[0])