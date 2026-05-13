import os
import json
import h5py
import numpy as np
import shutil

dir_braid = '/mnt/workspace/braid/workdir/'
dir_photos = f'{dir_braid}camera/photos/'

print('Loading vehicle info.')
with open(f'{dir_braid}camera/recognized_vehicles.json', 'r') as file:
    vehicle_info = json.load(file)
file.close()

photo_ids = {}
for record in vehicle_info:
    photo_ids[str(record['vehicle_timestamp'])] = record['photo_id']

print('Loading samples.')
meta = np.load(f'{dir_braid}meta.npy')
X = np.load(f'{dir_braid}signals_x.npy')
Y = np.load(f'{dir_braid}signals_y.npy')
P = np.load(f'{dir_braid}signals_p.npy')

csv = open('samples.csv', 'w')

print('Exporting.')
for signal, pulse, prediction, m in zip(X, Y, P, meta):
    photo_id = photo_ids[m[0]]

    groups_detected = m[1]
    groups_weighed = m[2]
    groups_final = m[3]

    csv.write(f'{photo_id},{groups_detected},{groups_weighed},{groups_final}')
    
    for value in signal:
        csv.write(f',{value}')
    
    for value in pulse:
        csv.write(f',{value}')
    
    for value in prediction:
        csv.write(f',{value}')
    
    csv.write('\n')

    photo_src = f'{dir_photos}{photo_id // 1000}/{photo_id}.png'
    photo_dst = f'{photo_id}.png'
    shutil.copyfile(photo_src, photo_dst)

csv.close()

