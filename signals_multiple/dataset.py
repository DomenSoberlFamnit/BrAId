import os
import json
import h5py
import numpy as np
from progress.bar import Bar
from datetime import datetime

def generate(
    dir_braid,
    include_non_fixed=True,
    include_fixed=True,
    exclude_days=None
):
    max_signal_size = 1500

    fname_signals = f'{dir_braid}data/nn_normalised_all_signals.hdf5'
    fname_pulses = f'{dir_braid}data/nn_normalised_all_pulses.json'
    fname_ground_truth = f'{dir_braid}data/ground_truth.json'
    fname_vehicle_index = f'{dir_braid}data/vehicle_index.json'

    print('Loading data.')

    with open(fname_pulses, 'r') as file:
        pulses = json.load(file)
    file.close()

    with open(fname_ground_truth, 'r') as file:
        ground_truth = json.load(file)
    file.close()

    with open(fname_vehicle_index, 'r') as file:
        vehicle_index = json.load(file)
    file.close()

    ts_id = {}
    for vehicle in vehicle_index:
        ts_id[str(vehicle['ts_vehicle'])] = vehicle['id']

    signals = h5py.File(fname_signals, 'r')

    cnt = 0
    meta = []
    X_single = []
    X_multiple = []
    Y = []

    min_dist = 1000000
    max_dist = 0
    max_axles = 0
    lengths = []
    locations = []

    bar = Bar('Generating instances', max=len(pulses))
    for pulse in pulses:
        bar.next()

        # Does the sample pass all the include conditions?
        if not pulse['photo_match']:
            continue
        
        sample_fixed = not (pulse['vehicle']['detected']['axle_pulses'] == pulse['vehicle']['weighed']['axle_pulses'] == pulse['vehicle']['final']['axle_pulses'])

        if not sample_fixed and not include_non_fixed:
            continue

        if sample_fixed and not include_fixed:
            continue
        
        date = datetime.fromtimestamp(pulse['ts'])
        date_str = f'{date.year}-{date.month}-{date.day}'

        if exclude_days is not None:
            if date_str in exclude_days:
                continue
        
        # This sample is included.
        cnt += 1

        id = ts_id[str(pulse['ts'])]
        ts = pulse['ts_str']

        # Store metadata.
        meta.append({
            'id': id,
            'timestamp': ts,
            'axle_groups': {
                'detected': pulse['vehicle']['detected']['axle_groups'],
                'weighed': pulse['vehicle']['weighed']['axle_groups'],
                'final': pulse['vehicle']['final']['axle_groups'],
            }
        })

        # Collect signal 11admp. Use it in single setting.
        raw_signal = np.array(signals[ts]['11admp'])
        output = np.zeros(max_signal_size)
        output[0:len(raw_signal)] = raw_signal
        X_single.append(output)
        lengths.append(len(raw_signal))

        output = np.zeros((14, max_signal_size))

        # Collect s111 and s112. Use them in multiple setting.
        raw_signal = np.array(signals[ts]['s111'])
        output[0, 0:len(raw_signal)] = raw_signal
        X_multiple.append(output)
        lengths.append(len(raw_signal))

        raw_signal = np.array(signals[ts]['s112 a11'])
        output[1, 0:len(raw_signal)] = raw_signal
        X_multiple.append(output)
        lengths.append(len(raw_signal))

        # Collect signals w01 - w12.
        for i in range(12):
            raw_signal = np.array(signals[ts][f'w{(i + 1):02d}'])
            output[i + 2, 0:len(raw_signal)] = raw_signal
            lengths.append(len(raw_signal))
        X_multiple.append(output)

        # Collect pulses
        raw_pulses = pulse['vehicle']['final']['axle_pulses']
        
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

        pulses = np.zeros(max_signal_size)
        for location in raw_pulses:
            pulses[location] = 1
            locations.append(location)
        
        Y.append(pulses)

    bar.finish()
    signals.close()

    X_single = np.array(X_single)
    X_multiple = np.array(X_multiple)
    Y = np.array(Y)

    print('Saving data.')
    file = open(f'{dir_braid}meta.json', 'w')
    json.dump(meta, file)
    file.close()

    np.save(f'{dir_braid}x_single.npy', X_single)
    np.save(f'{dir_braid}x_multiple.npy', X_multiple)
    np.save(f'{dir_braid}y.npy', Y)

    print(f'Found {cnt} samples.')
    print(f'Signal lengths are between {np.min(lengths)} and {np.max(lengths)}.')
    print(f'Pulses are located between {np.min(locations)} and {np.max(locations)}.')
    print(f'Maximum number of axles is {max_axles}.')
    print(f'Minimal distance between pulses is {min_dist}.')
    print(f'Maximal distance between pulses is {max_dist}.')

    return (meta, (X_single, X_multiple), Y)

def load(
    dir_braid
):
    print("Loading data.")

    with open(f'{dir_braid}meta.json', 'r') as file:
        meta = json.load(file)
    file.close()    

    X_single = np.load(f'{dir_braid}x_single.npy')
    X_multiple = np.load(f'{dir_braid}x_multiple.npy')
    Y = np.load(f'{dir_braid}y.npy')

    return (meta, (X_single, X_multiple), Y)

def load_or_generate(
    dir_braid,
    include_non_fixed=True,
    include_fixed=True,
    exclude_days=None,
    force_dataset_generation=False
):
    if not force_dataset_generation and os.path.exists(f'{dir_braid}meta.json'):
        return load(dir_braid)
    else:
        return generate(dir_braid, include_non_fixed, include_fixed, exclude_days)

def shuffle(data):
    print("Shuffling data.")

    (meta, (X_single, X_multiple), Y) = data

    indices = np.arange(0, len(Y))
    indices = np.random.permutation(indices)

    meta_new = []
    for i in indices:
        meta_new.append(meta[i])

    X_single_new = X_single[indices]
    X_multiple_new = X_multiple[indices]
    Y_new = Y[indices]

    return (meta_new, (X_single_new, X_multiple_new), Y_new)

def fold(data, fold_k, fold_n):
    print(f'Creating fold {fold_k + 1}/{fold_n}.')

    (meta, (X_single, X_multiple), Y) = data

    assert fold_k < fold_n

    indices = np.arange(0, len(Y))
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
    
    meta_train = []
    for i in train_idx:
        meta_train.append(meta[i])

    X_single_train = X_single[train_idx]
    X_multiple_train = X_multiple[train_idx]
    Y_train = Y[train_idx]
    
    data_train = (meta_train, (X_single_train, X_multiple_train), Y_train)

    meta_test = []
    for i in test_idx:
        meta_test.append(meta[i])

    X_single_test = X_single[test_idx]
    X_multiple_test = X_multiple[test_idx]
    Y_test = Y[test_idx]

    data_test = (meta_test, (X_single_test, X_multiple_test), Y_test)

    return data_train, data_test