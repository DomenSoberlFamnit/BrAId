import os
import numpy as np
import h5py
import json
from datetime import datetime
import shutil

dir_braid = '/home/hicup/disk/braid/'
dir_data = f'{dir_braid}data/'
dir_yolo_photos = f'{dir_braid}yolo_photos/'

year = 2014
month = 3
day = 10

def time_seen(ts, times):
    for t in times:
        dt = abs(t - ts)
        if dt <= 1.0:
            return True
    
    return False

print("Loading the metadata.")
meta_id = []
with h5py.File('../metadata/metadata.hdf5', 'r') as meta:
    for groups in meta.keys():
        data = meta[groups]
        for id in data:
            prop = json.loads(data[id].asstr()[()])
            meta_id.append(str(id))

print("Loading validation set.")
with open(f'{dir_braid}validation_set.json', 'r') as f:
    validation_set = json.load(f)
    f.close()

print("Loading recognized_vehicles.json")
f = open(f'{dir_braid}recognized_vehicles.json')
rv = json.load(f)
f.close()

print("Loading testing_id.npy")
testing_id = np.load(f'{dir_data}testing_id.npy')

print("Collecting the validation set.")
validation_set = []

for id in testing_id:
    validation_set.append(str(id))

print(f'We have {len(validation_set)} validation instances.')

print("Collecting instances.")

distribution = {}
distribution_labelled = {}

times = []

cnt = 0
cnt_double = 0
cnt_nometa = 0
calendar = {}
for vehicle in rv:
    id = str(vehicle['photo_id'])
    ts = vehicle['vehicle_timestamp']
    groups = vehicle['axle_groups']

    dt = datetime.fromtimestamp(ts)

    if dt.year == year and dt.month == month and dt.day == day:
        pass
    else:
        continue
    
    if id not in meta_id:
        cnt_nometa += 1
        continue

    if time_seen(ts, times):
        cnt_double += 1
        times.append(ts)
        continue
    
    times.append(ts)

    if dt.hour not in distribution:
        distribution[dt.hour] = 0
    
    distribution[dt.hour] += 1

    if id in validation_set:
        if dt.hour not in distribution_labelled:
            distribution_labelled[dt.hour] = 0
    
        distribution_labelled[dt.hour] += 1

print("No meta:", cnt_nometa)
print("Doubled:", cnt_double)
print(distribution)
print(distribution_labelled)

