import os
import json
import h5py
import numpy as np
import shutil

dir_braid = '/home/hicup/disk/braid/'
training_class_size = 50

miss_index = {}

def copy_png(id, label):
    global miss_index

    ai = label
    if id in miss_index:
        ai = miss_index[id]['AI']

    src = f'{dir_braid}cropped_photos/{label}/{id}.png'
    dst = f'./photos/{label}/{id}_{label}_{ai}.png'

    if not os.path.exists(f'./photos/'):
        os.mkdir(f'./photos/')
    if not os.path.exists(f'./photos/{label}/'):
        os.mkdir(f'./photos/{label}/')

    shutil.copyfile(src, dst)

    return dst

# Build the misclassified index
for root, dirs, files in os.walk('./miss/', topdown=False):
    for name in files:
        parts = name.split(".")[0].split("_")
        id = parts[0]
        label = parts[1]
        ai = parts[2]
        miss_index[id] = {'label':label, 'AI':ai}

# Build the valid photos index
with open(f'{dir_braid}valid_photos.json', 'r') as f:
    valid_photos = json.load(f)

valid_ids = []
for photo in valid_photos:
    valid_ids.append(photo['photo_id'])

with open(f'{dir_braid}group_index.json', 'r') as f:
    group_index = json.load(f)
valid_groups = list(group_index.keys())

with open('../metadata/preselected_ids.json', 'r') as f:
    preselected_ids = json.load(f)

distribution = {}

with h5py.File('../metadata/metadata.hdf5', 'r') as meta:
    for groups in meta.keys():
        if groups not in valid_groups:
            continue
        data = meta[groups]

        selected_ids = []
        remaining_ids = []

        for id in data:
            if id not in valid_ids:
                continue

            prop = json.loads(meta[f'{groups}/{id}'].asstr()[()])
            true_groups = prop['axle_groups'] if 'axle_groups' in prop else groups

            if id in miss_index:
                ai = miss_index[id]['AI']
            else:
                ai = true_groups

            seen_by = ''
            changed_by = ''
            if 'seen_by' in prop and prop['seen_by'] != None:
                seen_by = prop['seen_by'][1]
            if 'changed_by' in prop and prop['changed_by'] != None:
                changed_by = prop['changed_by'][1]

            if id in preselected_ids:
                selected_ids.append(id)
                png_path = copy_png(id, true_groups)
                print(f'{id},{groups},{true_groups},{ai},{seen_by},{changed_by},\'=HYPERLINK("{png_path}","{png_path}")\',critical')
            else:
                remaining_ids.append(id)
        
        remaining_ids = np.array(remaining_ids)
        np.random.shuffle(remaining_ids)

        for i in range(training_class_size - len(selected_ids)):
            id = remaining_ids[i]

            prop = json.loads(meta[f'{groups}/{id}'].asstr()[()])
            true_groups = prop['axle_groups'] if 'axle_groups' in prop else groups

            selected_ids.append(id)
            print(f'{id},{groups},{true_groups},{ai},{seen_by},{changed_by},\'=HYPERLINK("{png_path}","{png_path}")\',')
            png_path = copy_png(id, true_groups)

        distribution[groups] = selected_ids

with open('../metadata/testing_ids.json', 'w') as f:
    json.dump(distribution, f)