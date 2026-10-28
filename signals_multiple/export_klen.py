import os
import json
import h5py
import numpy as np
from progress.bar import Bar
from datetime import datetime

def generate(dir_braid):
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

    gvw = {}
    speed = {}
    lane = {}
    bar = Bar('Reading index values', max=len(vehicle_index))
    for vehicle in vehicle_index:
        gvw[vehicle['id']] = vehicle['gvw']
        speed[vehicle['id']] = vehicle['speed']
        lane[vehicle['id']] = vehicle['lane']
        bar.next()
    bar.finish()

    
    bar = Bar('Generating metadata', max=len(pulses))
    for pulse in pulses:
        bar.next()

        # Does the sample pass all the include conditions?
        if not pulse['photo_match']:
            continue
        
        date = datetime.fromtimestamp(pulse['ts'])
        date_str = f'{date.year}-{date.month}-{date.day}'

        # This sample is included.
        cnt += 1

        id = ts_id[str(pulse['ts'])]
        ts = pulse['ts_str']

        # Store metadata.
        meta.append({
            'id': id,
            'date': {'year': date.year, 'month': date.month, 'day': date.day},
            'time': {'hour': date.hour, 'minute': date.minute, 'second': date.second},
            'lane': lane[id],
            'speed': speed[id],
            'GVW': gvw[id],
            'axle_positions': pulse['vehicle']['final']['axle_pulses'],
            'axle_weights': pulse['vehicle']['final']['axle_weight'],
            'axle_groups': pulse['vehicle']['final']['axle_groups']
        })

    bar.finish()
    signals.close()

    print('Saving data.')
    file = open(f'sentvid_meta.json', 'w')
    json.dump(meta, file)
    file.close()

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

generate('/mnt/workspace/braid/workdir/')
