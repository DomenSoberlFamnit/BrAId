import os
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from progress.bar import Bar
from datetime import datetime

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

def signal_trim_back(raw_signal, raw_pulses):
    i = raw_pulses[-1]
    
    while i < len(raw_signal) and raw_signal[i] >= 0:
        i += 1
    
    while i < len(raw_signal) and raw_signal[i] < 0:
        i += 1
    
    amount = 0
    while i < len(raw_signal):
        amount += abs(raw_signal[i])
        raw_signal[i] = 0
        i += 1

    return amount

def generate_samples(
    dir_braid,
    signal_length,
    signal_name,
    output_type="pulses",
    normalized_signals=True,
    include_correct=True,
    include_fixed=True,
    gen_daily_dist=False,
    from_csv=None,
    include_days=None,
    exclude_days=None
):
    if normalized_signals:
        fname_signals = 'nn_normalised_signals.hdf5'
        fname_pulses = 'nn_normalised_pulses.json'
    else:
        fname_signals = 'nn_signals.hdf5'
        fname_pulses = 'nn_pulses.json'
    X = []
    Y = []

    Y_dist = []
    Y_size = []


    lengths = []
    locations = []
    timestamps = []
    meta = []
    predictions = []

    min_dist = 1000000
    max_dist = 0
    max_axles = 0

    allowed_ts = None
    allowed_predictions = {}
    if from_csv is not None and os.path.exists(from_csv):
        allowed_ts = []
        file = open(from_csv, 'r')
        header = True
        for line in file:
            if header:
                header = False
                continue
            rows = line.strip().split(",")
            
            ts = float(rows[0])
            allowed_ts.append(ts)

            prediction = []
            f = open(rows[10], 'r')
            for value in f.readline().strip().split(','):
                prediction.append(float(value))
            f.close()

            allowed_predictions[ts] = np.array(prediction)
            

    daily_dist = {}

    with open(f'{dir_braid}data/{fname_pulses}', 'r') as file:
        pulses_file = json.load(file)
    file.close()

    cnt_corrected = 0
    with h5py.File(f'{dir_braid}data/{fname_signals}', 'r') as signals:
        cnt = 0

        bar = Bar('Generating instances', max=len(pulses_file))
        for pulse in pulses_file:
            photo_match = pulse['photo_match']

            if not photo_match:
                bar.next()
                continue

            sample_correct = pulse['vehicle']['detected']['axle_pulses'] == pulse['vehicle']['weighed']['axle_pulses'] == pulse['vehicle']['final']['axle_pulses']

            # Do we skip correct samples?
            if not include_correct and sample_correct:
                bar.next()
                continue

            # Do we skip fixed (incorrect) samples?
            if not include_fixed and not sample_correct:
                bar.next()
                continue

            # If we have a list of allowed TS, skip if not in the list.
            if allowed_ts is not None:
                if pulse['ts'] not in allowed_ts:
                    bar.next()
                    continue

            # Time stamp
            date = datetime.fromtimestamp(pulse['ts'])
            date_str = f'{date.year}-{date.month}-{date.day}'

            if exclude_days is not None:
                if date_str in exclude_days:
                    bar.next()
                    continue

            if include_days is not None:
                if date_str not in include_days:
                    bar.next()
                    continue

            # Daily distibution
            if date_str not in daily_dist:
                daily_dist[date_str] = {
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'correct': 0,
                    'fixed': 0
                }

            if sample_correct:
                daily_dist[date_str]['correct'] += 1
            else:
                daily_dist[date_str]['fixed'] += 1

            # This sample is included.
            cnt += 1

            # Get metadata.
            ts = pulse['ts_str']
            timestamps.append(ts)

            # Store metadata.
            meta.append([pulse['ts'], pulse['vehicle']['detected']['axle_groups'], pulse['vehicle']['weighed']['axle_groups'], pulse['vehicle']['final']['axle_groups']])

            # Get raw signal data.
            raw_signal = np.array(signals[ts][signal_name])
            raw_pulses = pulse['vehicle']['final']['axle_pulses']

            # Filter the signal.
            amount = signal_trim_back(raw_signal, raw_pulses)
            if amount > 0:
                cnt_corrected += 1
            
            # Length of the signal.
            raw_size = len(raw_signal)
            lengths.append(raw_size)

            offset = int((signal_length - raw_size) / 2)
            offset = offset if offset >= 0 else 0

            #offset = 0

            signal = np.zeros(signal_length)
            if raw_size <= signal_length:
                signal[offset:(offset + raw_size)] = raw_signal
            else:
                signal = raw_signal[0:signal_length]
            
            X.append(signal)

            # Min distances between pulses.
            for i in range(len(raw_pulses) - 1):
                dist = raw_pulses[i + 1] - raw_pulses[i]
                if dist < min_dist:
                    min_dist = dist
            
            # Max distances between pulses.
            for i in range(len(raw_pulses) - 1):
                dist = raw_pulses[i + 1] - raw_pulses[i]
                if dist > max_dist:
                    max_dist = dist

            # Max axles (pulses)
            if len(raw_pulses) > max_axles:
                max_axles = len(raw_pulses)

            if output_type == "pulses":
                pulses = np.zeros(signal_length)
                
                for location in raw_pulses:
                    pulses[location+offset] = 1
                    locations.append(location + offset)

                Y.append(pulses)
            
            elif output_type == "distances":
                vector = np.zeros(10) # Max 11 axles, 10 distances
                for i in range(len(raw_pulses) - 1):
                    vector[i] = raw_pulses[i + 1] - raw_pulses[i]
                Y_dist.append(vector)

                Y_size.append(len(raw_pulses) - 2) # Distances: 1 (class 0), 2 (class 1), ..., 10 (class 9)

                for location in raw_pulses:
                    locations.append(location + offset)
            
            else:
                print("Unknown data output type.")
                quit()

            if len(allowed_predictions) > 0:
                predictions.append(allowed_predictions[pulse['ts']])

            bar.next()
        bar.finish()

    signals.close()

    meta = np.array(meta)
    X = np.array(X)
    
    if output_type == "pulses":
        Y = np.array(Y)
    elif output_type == "distances":
        Y = (np.array(Y_dist), np.array(Y_size))
    else:
        Y = np.array([])

    predictions = np.array(predictions)

    print(f'Found {cnt} samples.')
    print(f'Signal lengths are between {np.min(lengths)} and {np.max(lengths)}.')
    print(f'Pulses are located between {np.min(locations)} and {np.max(locations)}.')
    print(f'Maximum number of axles is {max_axles}.')
    print(f'Minimal distance between pulses is {min_dist}.')
    print(f'Maximal distance between pulses is {max_dist}.')
    print(f'Corrected {cnt_corrected} signals.')

    print(f'Saving {dir_braid}meta.npy')
    np.save(f'{dir_braid}meta.npy', meta)

    print(f'Saving {dir_braid}signals_x.npy')
    np.save(f'{dir_braid}signals_x.npy', X)
    
    if output_type == "pulses":
        print(f'Saving {dir_braid}signals_y.npy')
        np.save(f'{dir_braid}signals_y.npy', Y)
    elif output_type == "distances":
        print(f'Saving {dir_braid}signals_y_dist.npy')
        np.save(f'{dir_braid}signals_y_dist.npy', Y_dist)
        print(f'Saving {dir_braid}signals_y_size.npy')
        np.save(f'{dir_braid}signals_y_size.npy', Y_size)

    if len(allowed_predictions) > 0:
        print(f'Saving {dir_braid}signals_p.npy')
        np.save(f'{dir_braid}signals_p.npy', predictions)

    if gen_daily_dist:
        print(f'Saving {dir_braid}daily_distribution.csv')
        f = open(f'{dir_braid}daily_distribution.csv', 'w')
        for ts in daily_dist:
            record = daily_dist[ts]
            f.write(f'{record['year']},{record['month']},{record['day']},{record['correct']},{record['fixed']}\n')
        f.close()

    return meta, X, Y

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

def get_data(
    dir_braid,
    input_signal,
    output_type,
    normalized_signals,
    include_fixed,
    include_correct,
    signal_length,
    shuffle,
    gen_daily_dist=False,
    from_csv=None,
    include_days=None,
    exclude_days=None
):   
    meta, X, Y = generate_samples(
        dir_braid,
        signal_length,
        input_signal,
        output_type,
        normalized_signals,
        include_correct,
        include_fixed,
        gen_daily_dist,
        from_csv,
        include_days,
        exclude_days
    )

    print(f'The number of samples: {len(meta)}')

    if shuffle:
        print('Shuffling data.')
        
        indices = np.arange(0, len(meta))
        indices = np.random.permutation(indices)

        meta = meta[indices]
        X = X[indices]
        Y = Y[indices]

    return (meta, X, Y)

def export_data(
    dir_braid,
    input_signal,
    normalized_signals,
    include_fixed,
    include_correct,
    signal_length,
    vehicle_info
):
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

    data = []
    with h5py.File(f'{dir_braid}data/{fname_signals}', 'r') as signals:
        cnt = 0

        bar = Bar('Exporting training instances', max=len(pulses_file))
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
            ts_float = pulse['ts']

            # Construct matrix X.
            raw_signal = np.array(signals[ts][input_signal])
            
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

            # Axle distances
            axle_distances = pulse['vehicle']['final']['axle_distance']

            instance = {
                'id': vehicle_info[str(ts_float)]['id'],
                'signal': signal.tolist(),
                'pulses': pulses.tolist(),
                'axle_distances': axle_distances
            }

            data.append(instance)

            bar.next()
        bar.finish()

    signals.close()

    with open("axle_data.json", "w") as f:
        json.dump(data, f)

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
        train_idx = np.concat((indices[0:start_idx], indices[end_idx:]))

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
