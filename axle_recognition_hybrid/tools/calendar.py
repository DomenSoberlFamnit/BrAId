import numpy as np
import h5py
import json
from datetime import datetime

dir_braid = '/home/hicup/disk/braid/'

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

cnt = 0
calendar = {}
for vehicle in rv:
    id = vehicle['photo_id']
    ts = vehicle['vehicle_timestamp']

    dt = datetime.fromtimestamp(ts)
    date = f'{dt.day}-{dt.month}-{dt.year}'

    if date not in calendar:
        calendar[date] = {'seen':0, 'labeled':0, 'all':0}

    if id in seen:
        calendar[date]['seen'] += 1
        if np.isin(id, training_set):
            calendar[date]['labeled'] += 1
    
    calendar[date]['all'] += 1
    
    cnt += 1
    if cnt % 10000 == 0:
        print(cnt)

f = open("calendar.txt", "w")
for date in calendar:
    seen = calendar[date]['seen']
    labeled = calendar[date]['labeled']
    cnt = calendar[date]['all']

    f.write(f'{date},{seen},{labeled},{cnt}\n')

f.close()