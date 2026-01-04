import os
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from progress.bar import Bar

import baseline

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

def generate_training_samples(dir_braid, signal_length, signal_name, normalized_signals=True, include_correct=True, include_fixed=True):
    if normalized_signals:
        fname_signals = 'nn_normalised_signals.hdf5'
        fname_pulses = 'nn_normalised_pulses.json'
    else:
        fname_signals = 'nn_signals.hdf5'
        fname_pulses = 'nn_pulses.json'
    
    X = []
    Y = []

    lengths = []
    locations = []
    timestamps = []
    meta = []

    min_dist = 1000000

    with open(f'{dir_braid}data/{fname_pulses}', 'r') as file:
        pulses_file = json.load(file)
    file.close()

    with h5py.File(f'{dir_braid}data/{fname_signals}', 'r') as signals:
        cnt = 0

        bar = Bar('Generating training instances', max=len(pulses_file))
        for pulse in pulses_file:
            photo_match = pulse['photo_match']

            if not photo_match:
                bar.next()
                continue

            cnt += 1

            sample_correct = pulse['vehicle']['detected']['axle_pulses'] == pulse['vehicle']['final']['axle_pulses']

            # Do we skip correct samples?
            if not include_correct and sample_correct:
                bar.next()
                continue

            # Do we skip fixed (incorrect) samples?
            if not include_fixed and not sample_correct:
                bar.next()
                continue

            # Get metadata.
            ts = pulse['ts_str']
            timestamps.append(ts)

            # Store metadata.
            meta.append([pulse['ts'], pulse['vehicle']['detected']['axle_groups'], pulse['vehicle']['final']['axle_groups']])

            # Construct matrix X.
            raw_signal = np.array(signals[ts][signal_name])
            
            # Length of the signal.
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
            raw_pulses = pulse['vehicle']['final']['axle_pulses']

            # Min distances between pulses.
            for i in range(len(raw_pulses) - 1):
                dist = raw_pulses[i + 1] - raw_pulses[i]
                if dist < min_dist:
                    min_dist = dist

            pulses = np.zeros(signal_length)
            for location in raw_pulses:
                pulses[location+offset] = 1
                locations.append(location+offset)

            Y.append(pulses)

            bar.next()
        bar.finish()

    signals.close()

    meta = np.array(meta)
    X = np.array(X)
    Y = np.array(Y)

    print(f'Found {cnt} training samples.')
    print(f'Signal lengths are between {np.min(lengths)} and {np.max(lengths)}.')
    print(f'Pulses are located between {np.min(locations)} and {np.max(locations)}.')
    print(f'Minimal distance between pulses is {min_dist}.')

    print(f'Saving {dir_braid}meta.npy')
    np.save(f'{dir_braid}meta.npy', X)

    print(f'Saving {dir_braid}signals_x.npy')
    np.save(f'{dir_braid}signals_x.npy', X)
    
    print(f'Saving {dir_braid}signals_y.npy')
    np.save(f'{dir_braid}signals_y.npy', Y)

    return meta, X, Y

def load_samples(dir_braid):
    # Check if the samples set exists.
    data_found = True
    if os.path.exists(f'{dir_braid}/meta.npy'):
        print('Found meta.npy')
    else:
        print('File meta.npy has not been found.')
        data_found = False

    if os.path.exists(f'{dir_braid}/signals_x.npy'):
        print('Found signals_x.npy')
    else:
        print('File signals_x.npy has not been found.')
        data_found = False

    if os.path.exists(f'{dir_braid}/signals_y.npy'):
        print('Found signals_y.npy')
    else:
        print('File signals_y.npy has not been found.')
        data_found = False

    if data_found:
        print('Loading samples.')
        meta = np.load(f'{dir_braid}meta.npy')
        X = np.load(f'{dir_braid}signals_x.npy')
        Y = np.load(f'{dir_braid}signals_y.npy')
        return meta, X, Y
    
    return None

def load_training_samples(dir_braid):
    # Check if the training set exists.
    data_found = True
    if os.path.exists(f'{dir_braid}/meta_training.npy'):
        print('Found meta_training.npy')
    else:
        print('File meta_training.npy has not been found.')
        data_found = False
    
    if os.path.exists(f'{dir_braid}/signals_training_x.npy'):
        print('Found signals_training_x.npy')
    else:
        print('File signals_training_x.npy has not been found.')
        data_found = False

    if os.path.exists(f'{dir_braid}/signals_training_y.npy'):
        print('Found signals_training_y.npy')
    else:
        print('File signals_training_y.npy has not been found.')
        data_found = False

    if data_found:
        print('Loading training samples.')
        meta = np.load(f'{dir_braid}meta_training.npy')
        X = np.load(f'{dir_braid}signals_training_x.npy')
        Y = np.load(f'{dir_braid}signals_training_y.npy')
        return meta, X, Y
    
    return None

def load_testing_samples(dir_braid):
    # Check if the training set exists.
    data_found = True
    if os.path.exists(f'{dir_braid}/meta_testing.npy'):
        print('Found meta_testing.npy')
    else:
        print('File meta_testing.npy has not been found.')
        data_found = False
    
    if os.path.exists(f'{dir_braid}/signals_testing_x.npy'):
        print('Found signals_testing_x.npy')
    else:
        print('File signals_testing_x.npy has not been found.')
        data_found = False

    if os.path.exists(f'{dir_braid}/signals_testing_y.npy'):
        print('Found signals_testing_y.npy')
    else:
        print('File signals_testing_y.npy has not been found.')
        data_found = False

    if data_found:
        print('Loading test samples.')
        meta = np.load(f'{dir_braid}meta_testing.npy')
        X = np.load(f'{dir_braid}signals_testing_x.npy')
        Y = np.load(f'{dir_braid}signals_testing_y.npy')
        return meta, X, Y
    
    return None

def split_samples(meta, X, Y, testing_size):
    indices = np.arange(0, len(X))
    #indices = np.random.permutation(indices)
    
    split_idx = round(len(X) * testing_size)
    testing_idx = indices[0:split_idx]
    train_idx = indices[split_idx:]

    meta_train = meta[train_idx]
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    
    meta_test = meta[testing_idx]
    X_test = X[testing_idx]
    Y_test = Y[testing_idx]

    return meta_train, X_train, Y_train, meta_test, X_test, Y_test

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

def load_vehicle_info(dir_braid, dir_vehicle_index):
    fpath_info = f'{dir_braid}vehicle_info.json'
    fpath_index = f'{dir_vehicle_index}vehicle_index.json'

    if os.path.exists(fpath_info):
        with open(fpath_info, 'r') as file:
            vehicle_info = json.load(file)
        file.close()
        return vehicle_info
    
    if not os.path.exists(fpath_index):
        return None

    with open(fpath_index, 'r') as file:
        vehicle_index = json.load(file)
    file.close()

    bar = Bar('Generating vehicle info', max=len(vehicle_index))
    vehicle_info = {}
    for vehicle in vehicle_index:
        vehicle_info[str(vehicle['ts_vehicle'])] = {
            'id': vehicle['id'],
            'photo': f'{dir_braid}camera/photos/{vehicle['file']}'
        }

        bar.next()
    bar.finish()

    with open(fpath_info, "w") as file:
        json.dump(vehicle_info, file)
    file.close()

    return vehicle_info

def print_sample(dir_braid, data, id, vehicle_info):
    ts_vehicle = None
    for ts in vehicle_info:
        if vehicle_info[ts]['id'] == id:
            ts_vehicle = float(ts)
            break
    
    if ts_vehicle is None:
        print("ID NOT FOUND!")
        return
    
    print(ts_vehicle, type(ts_vehicle))
    
    fname_signals = 'nn_normalised_signals.hdf5'
    fname_pulses = 'nn_normalised_pulses.json'
    with h5py.File(f'{dir_braid}data/{fname_signals}', 'r') as signals:
        with open(f'{dir_braid}data/{fname_pulses}', 'r') as file:
            pulses = json.load(file)

        for pulse in pulses:
            photo_match = pulse['photo_match']

            if not photo_match:
                continue

            if pulse['ts'] == ts_vehicle:
                print(pulse)
                break

    meta, X, Y = data
    for (y, m) in zip(Y, meta):
        if m[0] == ts_vehicle:
            print(m, y)
            break

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

    signal = np.zeros(signal_length)
    pulses = np.zeros(signal_length)

    for i, (x, y) in enumerate(frames):
        signal[i] = (x[signal_focus])
        pulses[i] = y[0]
    
    return signal, pulses

def prepare(
    dir_braid,
    input_signal,
    normalized_signals,
    include_correct,
    include_fixed,
    signal_length,
    plot_data,
    testing_ratio,
    force_data_generation
):
    # Load or generate the vehicle index (a leftover from the camera project).
    vehicle_info = load_vehicle_info(dir_braid, f'{dir_braid}camera/')
    if vehicle_info is None:
        print('Cannot load vehicle info.')
        quit()

    # Analyze the dataset.
    base_ca = baseline.accuracy(dir_braid, normalized_signals)
    print(f'Baseline accuracy: {100 * base_ca:.2f}')

    # Try to load the training set and the testing set.
    if not force_data_generation:
        training_data = load_training_samples(dir_braid)
        testing_data = load_testing_samples(dir_braid)
    else:
        training_data = None
        testing_data = None

    # If the split data cannot be loaded, generate it.
    if training_data is None or testing_data is None:
        if not force_data_generation:
            samples = load_samples(dir_braid)
        else:
            samples = None

        # If samples cannot be loaded, generate them.
        if samples is None:
            meta, X, Y = generate_training_samples(dir_braid, signal_length, input_signal, normalized_signals, include_correct, include_fixed)

            if plot_data:
                print('Plotting the training data.')
                if not os.path.exists(f'{dir_braid}plots/raw/'):
                    os.makedirs(f'{dir_braid}plots/raw/')

                bar = Bar('Plotting', max=len(meta))
                cnt = 0
                for x, y, m in zip(X, Y, meta):
                    ts, groups = str(m[0]), str(m[1])
                    plot_sample(f'{dir_braid}plots/raw/{ts}.png', x, y, ts, groups)
                    cnt += 1
                    bar.next()
                bar.finish()

        else:
            meta, X, Y = samples
        
        # Now we have the samples, split them.
        meta_train, X_train, Y_train, meta_test, X_test, Y_test = split_samples(meta, X, Y, testing_ratio=0.3)

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

    return meta_train, X_train, Y_train, meta_test, X_test, Y_test

def get_data(
    dir_braid,
    input_signal,
    normalized_signals,
    include_fixed,
    include_correct,
    signal_length,
    shuffle
):   
    meta, X, Y = generate_training_samples(dir_braid, signal_length, input_signal, normalized_signals, include_correct, include_fixed)

    print(f'The number of samples: {len(meta)}')

    if shuffle:
        print('Shuffling data.')
        
        indices = np.arange(0, len(meta))
        indices = np.random.permutation(indices)

        meta = meta[indices]
        X = X[indices]
        Y = Y[indices]

    return (meta, X, Y)

def split_data(data, testing_ratio):
    (meta, X, Y) = data
    
    meta_train, X_train, Y_train, meta_test, X_test, Y_test = split_samples(meta, X, Y, testing_ratio)

    print(f'Random split with {testing_ratio} testing ratio.')
    print(f'The number of training samples: {len(meta_train)}')
    print(f'The number of testing samples: {len(meta_test)}')

    return (meta_train, X_train, Y_train), (meta_test, X_test, Y_test)

def split_fold(data, fold_k, fold_n):
    (meta, X, Y) = data

    assert fold_k < fold_n

    indices = np.arange(0, len(meta))
    fold_size = len(meta) // fold_n
    
    start_idx = fold_k * fold_size
    end_idx = start_idx + fold_size

    test_idx = indices[start_idx:end_idx]
    
    if fold_k == 0:
        train_idx = indices[end_idx:]
    elif fold_k == fold_n - 1:
        train_idx = indices[0:start_idx]
    else:
        train_idx = np.concat((indices[0:start_idx], indices[end_end:]))

    meta_train = meta[train_idx]
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    
    meta_test = meta[test_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    print(f'Split fold {fold_k + 1}/{fold_n}.')
    print(f'The number of training samples: {len(meta_train)}')
    print(f'The number of testing samples: {len(meta_test)}')

    return (meta_train, X_train, Y_train), (meta_test, X_test, Y_test)