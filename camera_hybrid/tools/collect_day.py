import os
import numpy as np
import h5py
import json
from datetime import datetime
import shutil

dir_braid = '/home/hicup/disk/braid/'
dir_photos = f'{dir_braid}photos/'
dir_yolo_photos = f'{dir_braid}yolo_photos/'

used_groups = ['1111', '1112', '111', '22', '1222', '113', '123', '122', '1211', '11', '1212', '112', '12']

year = 2014
month = 3
day = 10

seen = []
props = {}

print("Loading the metadata.")
with h5py.File('../metadata/metadata.hdf5', 'r') as meta:
    for groups in meta.keys():
        data = meta[groups]
        for id in data:
            prop = json.loads(data[id].asstr()[()])
            
            if prop['seen_by'] is not None:
                seen.append(int(id))
                props[int(id)] = prop

print("Loading recognized_vehicles.json")
f = open(f'{dir_braid}recognized_vehicles.json')
rv = json.load(f)
f.close()

print("Loading data_id.npy")
training_set = np.load(f'{dir_braid}data/data_id.npy')

print("Collecting instances.")

distribution = {}
distribution_rare = {}
validation_set = {}
used_set = {}
unused_set = {}
unchecked_set = {}
problematic_set = {}

cnt = 0
cnt_err = 0
cnt_rare = 0
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

        if groups not in used_groups:
            print("rare id:", id, groups)
            cnt_rare += 1

        if np.isin(id, training_set):
            validation_set[id] = groups
            used_set[id] = groups

        else:
            validation_set[id] = 'invalid'
            unused_set[id] = groups

            if 'errors' in props[id]:
                errors = props[id]['errors']
                if not 'multiple_vehicles' in errors:
                    if 'cannot_label' in errors:
                        #print(id, props[id]['errors'], groups)
                        problematic_set[id] = groups

            else:
                #print(id, props[id])
                pass
            cnt_err += 1
    else:
        validation_set[id] = ''
        unchecked_set[id] = ''

cnt = 0
for groups in distribution:
    print(f'{groups} {distribution[groups]}')
    cnt += distribution[groups]

print("Distribution sum:", cnt)
print("Rare groups:", cnt_rare)

quit()

with open(f'{dir_braid}validation_set.json', 'w') as file:
    json.dump(validation_set, file)

f = open(f'{dir_braid}unchecked_set.txt', "w")
for id in unchecked_set:
    f.write(f'{id} ?\n')
f.close()

print("Errors:", cnt_err)

print("Copying photos.")

dir_photos_unchecked = f'{dir_braid}unchecked_photos/'
dir_photos_used = f'{dir_braid}used_photos/'
dir_photos_unused = f'{dir_braid}unused_photos/'
dir_photos_problematic = f'{dir_braid}problematic_photos/'

if not os.path.exists(dir_photos_used):
    os.mkdir(dir_photos_used)

if not os.path.exists(dir_photos_unused):
    os.mkdir(dir_photos_unused)

if not os.path.exists(dir_photos_unchecked):
    os.mkdir(dir_photos_unchecked)

if not os.path.exists(dir_photos_problematic):
    os.mkdir(dir_photos_problematic)

for id in used_set:
    src = f'{dir_yolo_photos}{int(id / 1000)}/{id}.png'
    dst = f'{dir_photos_used}{id}-{used_set[id]}.png'
    shutil.copyfile(src, dst)

for id in unused_set:
    src = f'{dir_yolo_photos}{int(id / 1000)}/{id}.png'
    dst = f'{dir_photos_unused}{id}-{unused_set[id]}.png'
    shutil.copyfile(src, dst)

for id in unchecked_set:
    src = f'{dir_yolo_photos}{int(id / 1000)}/{id}.png'
    dst = f'{dir_photos_unchecked}{id}.png'
    shutil.copyfile(src, dst)

for id in problematic_set:
    src = f'{dir_photos}{int(id / 1000)}/{id}.png'
    dst = f'{dir_photos_problematic}{id}-{problematic_set[id]}.png'
    shutil.copyfile(src, dst)