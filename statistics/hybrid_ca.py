import sys
import os
import h5py
import json
import matplotlib.pyplot as plt
import pandas as pd

dir_braid = '/home/hicup/disk/braid/'

use_architectures = ['VGG19']
use_classes = ['1111', '1112', '111', '22', '1222', '113', '123', '122', '1211', '11', '1212', '112', '12']
error_flags = ['yolo_error', 'photo_truncated', 'vehicle_joined', 'vehicle_split', 'cannot_label', 'inconsistent_data', 'off_lane', 'wrong_lane', 'multiple_vehicles']

def prop_has_errors(prop):
    if 'errors' not in prop:
        return False
    
    errors = prop['errors']

    for flag in error_flags:
        if flag in errors and errors[flag] != 0:
            return True

    return False

def remove_raised_axles(truth_camera, raised_axles):
    raised_groups = list(truth_camera)   # String to list of characters.
    for axle in raised_axles.split(','): # For each group with a raised axle.
        idx = int(axle) - 1              # The index of the group with a raised axle.
        raised_groups[idx] = str(int(raised_groups[idx]) - 1) # Remove the raised axle.
    return ''.join(raised_groups)

def get_testing_results(number=None):
    (a, b) = (1, 11) if number == None else (number, number + 1)
    data = []
    for i in range(a, b):
        path = f'{dir_braid}results{i}/'
        for architecture in os.listdir(path):
            if architecture not in use_architectures:
                continue
            cls_data = {}
            filepath = os.path.join(path, architecture)
            if os.path.isdir(filepath):
                for (_, _, files) in os.walk(filepath):
                    for file in files:
                        if file.endswith('.png'):
                            parts = file.split('.')[0].split('_')
                            if len(parts) == 4:
                                id = parts[0]
                                truth = parts[1]
                                prediction = parts[2]
                                cls_data[id] = (truth, prediction)
            data.append((i, architecture, cls_data))
    return data

def process_classification_data(i, architecture, cls_data, output):
    # Build the index of siwim axle groups.
    hdf = pd.read_hdf('../metadata/grp_and_fixed.hdf5')
    siwim_groups = {}
    for index, row in hdf.iterrows():
        id = row['id']
        if id != 'nan' and row['rp01_grp'] != 'nan' and row['rp02_grp'] != 'nan' and row['rp03_grp'] != 'nan':
            siwim_groups[str(id)] = {'rp1': row['rp01_grp'], 'rp2': row['rp02_grp'], 'rp3': row['rp03_grp']}

    # Iterate through hand-checked instances.
    cnt_missing = 0
    with h5py.File('../metadata/metadata.hdf5', 'r') as file:
        for groups in file.keys():
            data = file[groups]
            for id in data:
                if id not in cls_data:
                    continue

                prop = json.loads(file[f'{groups}/{id}'].asstr()[()])

                (nn_truth, nn_prediction) = cls_data[id]

                # The label is the camera ground truth.
                truth_camera = prop['axle_groups'] if 'axle_groups' in prop else groups

                # The road ground truth is different if there are raised axles.
                truth_road = truth_camera
                raised_axles = prop['raised_axles'].strip() if 'raised_axles' in prop else ''
                if len(raised_axles) > 0:
                    truth_road = remove_raised_axles(truth_camera, raised_axles)

                if id in siwim_groups:
                    rp = siwim_groups[id]
                    output.write(f'{i},{architecture},{id},{rp['rp1']},{rp['rp2']},{rp['rp3']},{truth_camera},{truth_road},{raised_axles},{nn_truth},{nn_prediction}\n')
                else:
                    cnt_missing += 1

    print(f'Missing instances: {cnt_missing}')

def main():
    for (i, architecture, cls_data) in get_testing_results(number=None if len(sys.argv) < 2 else int(sys.argv[1])):
        print(f'Processing architecture {architecture} number {i}.')
        output = open(f'hybrid_ca_{i}.csv', 'w')
        output.write('EXPERIMENT,ARCHITECTURE,ID,RP1,RP2,RP3,CAMERA,ROAD,RAISED,NN_TRUTH,NN_PREDICTION\n')
        process_classification_data(i, architecture, cls_data, output)
        output.close()

if __name__ == "__main__":
    main()
