import os
import numpy as np
import h5py
import json
from datetime import datetime
import shutil

dir_braid = '/home/hicup/disk/braid/'
dir_yolo_photos = f'{dir_braid}yolo_photos/'

year = 2014
month = 3
day = 10

seen = []

print("Loading the metadata.")
with h5py.File('../metadata/metadata.hdf5', 'r') as meta:
    for groups in meta.keys():
        data = meta[groups]
        for id in data:
            prop = json.loads(data[id].asstr()[()])
            
            if prop['seen_by'] is not None:
                seen.append(int(id))

print("Loading recognized_vehicles.json")
f = open(f'{dir_braid}recognized_vehicles.json')
rv = json.load(f)
f.close()

print("Loading data_id.npy")
training_set = np.load(f'{dir_braid}data/data_id.npy')

print("Collecting instances.")

distribution = {}
validation_set = {}
unchecked_set = {}

cnt = 0
calendar = {}
for vehicle in rv:
    id = vehicle['photo_id']
    ts = vehicle['vehicle_timestamp']
    groups = vehicle['axle_groups']

    dt = datetime.fromtimestamp(ts)

    if dt.year == year and dt.month == month and dt.day == day:
        pass
    else:
        continue

    date = f'{dt.day}-{dt.month}-{dt.year}'

    if id in seen:
        if groups not in distribution:
            distribution[groups] = 1
        else:
            distribution[groups] += 1

        if np.isin(id, training_set):
            validation_set[id] = groups
        else:
            validation_set[id] = 'invalid'
    
    else:
        validation_set[id] = ''
        unchecked_set[id] = ''
    
for groups in distribution:
    print(f'{groups} {distribution[groups]}')

with open(f'{dir_braid}validation_set.json', 'w') as file:
    json.dump(validation_set, file)

f = open(f'{dir_braid}unchecked_set.txt', "w")
for id in unchecked_set:
    f.write(f'{id} ?\n')
f.close()


print("Copying photos.")

dir_photos = f'{dir_braid}unchecked_photos/'

if not os.path.exists(dir_photos):
    os.mkdir(dir_photos)

for id in unchecked_set:
    src = f'{dir_yolo_photos}{int(id / 1000)}/{id}.png'
    dst = f'{dir_photos}{id}.png'
    shutil.copyfile(src, dst)