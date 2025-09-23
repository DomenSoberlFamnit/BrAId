import sys
import os
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir_braid = '/home/hicup/disk/braid/'
dir_data = f'{dir_braid}data/'

use_architectures = ['VGG19', 'VGG16', 'ResNet101V2', 'DenseNet121', 'MobileNetV3Small']
use_groups = ['1111', '1112', '111', '22', '1222', '113', '123', '122', '1211', '11', '1212', '112', '12']

def remove_raised_axles(truth_camera, raised_axles):
    raised_groups = list(truth_camera)   # String to list of characters.
    for axle in raised_axles.split(','): # For each group with a raised axle.
        idx = int(axle) - 1              # The index of the group with a raised axle.
        raised_groups[idx] = str(int(raised_groups[idx]) - 1) # Remove the raised axle.
    return ''.join(raised_groups)

def process_architecture(architecture, validation_set, siwim_groups, meta_index, include_raised):
    # Get ground truth and NN predictions
    nn_results = {}

    path = f'{dir_braid}results/{architecture}/photos_test/'
    for (_, _, files) in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                parts = file.split('.')[0].split('_')
                id = parts[0]
                truth = parts[1]
                prediction = parts[2]
                
                nn_results[id] = {'truth':truth, 'prediction':prediction}

    for id in validation_set:
        if id not in nn_results:
            truth = validation_set[id]
            nn_results[id] = {'truth':truth, 'prediction':truth}

    if include_raised:
        f = open(f'siwim_validation_{architecture}.csv', 'w')
    else:
        f = open(f'siwim_validation_{architecture}_no_raised.csv', 'w')
    f.write('ID,RP1,RP2,RP3,CAMERA,ROAD,RAISED,IS_RAISED,RP1_CORRECT,RP2_CORRECT,RP3_CORRECT')
    f.write('NN_TRUTH,NN_PREDICTION,NN_CA,AGREE,TP,TN,FP,FN\n')

    tp_cnt = 0
    tn_cnt = 0
    fp_cnt = 0
    fn_cnt = 0

    for id in validation_set:
        nn_truth = nn_results[id]['truth']
        nn_prediction = nn_results[id]['prediction']

        truth_camera = nn_truth
        truth_road = truth_camera

        (meta_groups, meta_prop) = meta_index[id]

        is_raised = 0
        # If manually checked.
        raised_axles = meta_prop['raised_axles'].strip() if 'raised_axles' in meta_prop else ''
        if len(raised_axles) > 0:
            is_raised = 1
            truth_road = remove_raised_axles(truth_camera, raised_axles)

        if id in siwim_groups:
            rp1 = siwim_groups[id]['rp1']
            rp2 = siwim_groups[id]['rp2']
            rp3 = siwim_groups[id]['rp3']
        else:
            rp1 = truth_road
            rp2 = truth_road
            rp3 = truth_road

        rp1corr = 1 if rp1 == truth_road else 0
        rp2corr = 1 if rp2 == truth_road else 0
        rp3corr = 1 if rp3 == truth_road else 0

        nn_ca = 1 if nn_truth == nn_prediction else 0
        agree = 1 if rp2 == nn_prediction else 0

        tp = 1 if agree == 1 and rp2corr == 1 else 0
        tn = 1 if agree == 0 and rp2corr == 0 else 0
        fp = 1 if agree == 1 and rp2corr == 0 else 0
        fn = 1 if agree == 0 and rp2corr == 1 else 0

        if not include_raised and is_raised and fn == 1:
            fn = 0
            tn = 1

        tp_cnt += tp
        tn_cnt += tn
        fp_cnt += fp
        fn_cnt += fn

        raised_axles = raised_axles.replace(',', ' ')

        f.write(f"{id},{rp1},{rp2},{rp3},{truth_camera},{truth_road},{raised_axles},{is_raised},{rp1corr},{rp2corr},{rp3corr}")
        f.write(f"{nn_truth},{nn_prediction},{nn_ca},{agree},{tp},{tn},{fp},{fn}\n")

    f.close()

    precision = tp_cnt / (tp_cnt + fp_cnt)
    recall = tp_cnt / (tp_cnt + fn_cnt)

    print(f'TP: {tp_cnt}, TN: {tn_cnt}, FP: {fp_cnt}, FN: {fn_cnt}, precision: {precision}, recall: {recall}')

def main():
    print("Loading testing_id.npy")
    testing_id = np.load(f'{dir_data}testing_id.npy')

    print("Loading testing_y.npy")
    testing_y = np.load(f'{dir_data}testing_y.npy')

    print("Collecting the validation set.")
    validation_set = {}

    for (id, y) in zip (testing_id, testing_y):
        truth = use_groups[np.argmax(y)]
        validation_set[str(id)] = truth

    print(f'We have {len(validation_set)} validation instances.')

    print("Loading SIWIM predictions.")
    hdf = pd.read_hdf('../metadata/grp_and_fixed.hdf5')
    siwim_groups = {}
    for index, row in hdf.iterrows():
        id = row['id']
        if id != 'nan' and row['rp01_grp'] != 'nan' and row['rp02_grp'] != 'nan' and row['rp03_grp'] != 'nan':
            siwim_groups[str(id)] = {'rp1': row['rp01_grp'], 'rp2': row['rp02_grp'], 'rp3': row['rp03_grp']}
    
    print("Loading metadata.")
    meta_index = {}
    with h5py.File('../metadata/metadata.hdf5', 'r') as file:
        for groups in file.keys():
            data = file[groups]
            for id in data:
                prop = json.loads(file[f'{groups}/{id}'].asstr()[()])
                meta_index[id] = (groups, prop)

    for architecture in use_architectures:
        print(f"Processing {architecture}.")
        process_architecture(architecture, validation_set, siwim_groups, meta_index, True)
        process_architecture(architecture, validation_set, siwim_groups, meta_index, False)

if __name__ == "__main__":
    main()
