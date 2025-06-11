import os
import json
import collect_training_data

# Directories
dir_siwim = '/home/hicup/disk/siwim/siwim/'
dir_braid = '/home/hicup/disk/braid/'

# Check prerequisite files. They are generated with the axle_recognition_script or sent by ZAG.
if not os.path.exists(f'{dir_braid}vehicle_index.json'):
    print("File vehicle_index.json not found!")
    quit()

file = open(f'{dir_braid}vehicle_index.json')
vehicle_index = json.load(file)
file.close()

print(f'Vehicles: {len(vehicle_index)}')

if not os.path.exists(f'../metadata/metadata.hdf5'):
    print("File metadata.hdf5 not found!")
    quit()

collect_training_data.run(dir_siwim, dir_braid, vehicle_index)
