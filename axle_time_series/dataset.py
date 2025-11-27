import os
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def signal_data_found(dir_braid, normalized_signals=True):
    if normalized_signals:
        fname_signals = 'nn_normalised_signals.hdf5'
        fname_pulses = 'nn_normalised_pulses.json'
    else:
        fname_signals = 'nn_signals.hdf5'
        fname_pulses = 'nn_pulses.json'

    if os.path.exists(f'{dir_braid}data/{fname_signals}'):
        print(f'Found file {fname_signals}')
    else:
        print(f'File {fname_signals} not found!')
        return False

    if os.path.exists(f'{dir_braid}data/{fname_pulses}'):
        print(f'Found file {fname_pulses}')
    else:
        print(f'File {fname_pulses} not found!')
        return False
    
    return True

def generate_training_samples(dir_braid, signal_length, signal_name, normalized_signals=True):
    if normalized_signals:
        fname_signals = 'nn_normalised_signals.hdf5'
        fname_pulses = 'nn_normalised_pulses.json'
    else:
        fname_signals = 'nn_signals.hdf5'
        fname_pulses = 'nn_pulses.json'
    
    print(f'Generating the training dataset for signal {signal_name}.')

    X = []
    Y = []

    lengths = []
    locations = []
    axle_groups = []
    timestamps = []

    with h5py.File(f'{dir_braid}data/{fname_signals}', 'r') as signals:
        with open(f'{dir_braid}data/{fname_pulses}', 'r') as file:
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

            offset = int((signal_length - raw_size) / 2)
            offset = offset if offset >= 0 else 0

            signal = np.zeros(signal_length)
            if raw_size <= signal_length:
                signal[offset:(offset + raw_size)] = raw_signal
            else:
                signal = raw_signal[0:signal_length]
            
            X.append(signal)

            # Construct matrix Y.
            raw_pulses = pulse['vehicle']['detected']['axle_pulses']

            pulses = np.zeros(signal_length)
            for location in raw_pulses:
                pulses[location+offset] = 1
                locations.append(location+offset)
        
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

def frames_from_signal(signal, pulses, frame_size, include_empty=False):
    # Frame size is expected to be an even number.
    epsilon = int(frame_size / 2)

    # Store all the generated frames.
    frames = []

    # Frames over the left edge [0] .. [frame size - 1] inclusive.
    for i in range(epsilon):
        frame = np.zeros(frame_size)
        frame[(epsilon - i):frame_size] = signal[0:(epsilon + i)]

        empty = np.max(frame) == 0 and np.min(frame) == 0
        if not empty or include_empty:
            frames.append((frame, pulses[i]))
    
    # Frames between edges.
    for i in range(epsilon, len(signal) - epsilon + 1):
        frame = signal[(i - epsilon):(i - epsilon + frame_size)]
        
        empty = np.max(frame) == 0 and np.min(frame) == 0
        if not empty or include_empty:
            frames.append((frame, pulses[i]))

    # Frames over the right edge.
    for i in range(len(signal) - epsilon + 1, len(signal)):
        frame = np.zeros(frame_size)
        frame[0:(len(signal) - i + epsilon + 1)] = signal[(i - epsilon - 1):len(signal)]
        
        empty = np.max(frame) == 0 and np.min(frame) == 0
        if not empty or include_empty:
            frames.append((frame, pulses[i]))
    
    return frames

def signal_from_frames(frames):
    (frame, _) = frames[0]
    
    signal_length = len(frames)
    frame_size = len(frame)
    signal_focus = int(frame_size / 2)

    signal = []
    pulses = []

    for (x, y) in frames:
        signal.append(x[signal_focus])
        pulses.append(y)
    
    return np.array(signal), np.array(pulses)