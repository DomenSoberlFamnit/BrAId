import h5py
import json
import matplotlib.pyplot as plt
import pandas as pd

dir_braid = '/home/hicup/disk/braid/'

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

def main():
    # Build the index of siwim axle groups.
    hdf = pd.read_hdf('../metadata/grp_and_fixed.hdf5')
    siwim_groups = {}
    for index, row in hdf.iterrows():
        id = row['id']
        if id != 'nan' and row['rp01_grp'] != 'nan' and row['rp02_grp'] != 'nan' and row['rp03_grp'] != 'nan':
            siwim_groups[str(id)] = {'rp1': row['rp01_grp'], 'rp2': row['rp02_grp'], 'rp3': row['rp03_grp']}

    output = open('siwim_ca.csv', 'w')
    output.write('ID,RP1,RP2,RP3,CAMERA,ROAD,RAISED\n')

    # Iterate through hand-checked instances.
    cnt_missing = 0
    with h5py.File('../metadata/metadata.hdf5', 'r') as file:
        for groups in file.keys():
            data = file[groups]
            for id in data:
                prop = json.loads(file[f'{groups}/{id}'].asstr()[()])

                # A photo is useful if someone saw it and flagged no errors.
                photo_useful = False
                if prop['seen_by'] == None: # No one saw it, not useful.
                    continue

                if prop['changed_by'] == None:
                    photo_useful = True     # No changes made, the label is OK.
                elif not prop_has_errors(prop):
                    photo_useful = True     # Changes made, but no errors.
                
                if not photo_useful:
                    continue

                # The label is the camera ground truth.
                truth_camera = prop['axle_groups'] if 'axle_groups' in prop else groups

                # The road ground truth is different if there are raised axles.
                truth_road = truth_camera
                raised_axles = prop['raised_axles'].strip() if 'raised_axles' in prop else ''

                if len(raised_axles) > 0:
                    truth_road = remove_raised_axles(truth_camera, raised_axles)

                if id in siwim_groups:
                    rp = siwim_groups[id]
                    raised_axles = raised_axles.replace(',', ' ')
                    output.write(f'{id},{rp['rp1']},{rp['rp2']},{rp['rp3']},{truth_camera},{truth_road},{raised_axles}\n')
                else:
                    cnt_missing += 1

    print(f'Missing instances: {cnt_missing}')
    output.close()

if __name__ == "__main__":
    main()
