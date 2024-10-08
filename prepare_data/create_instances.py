import os
import numpy as np
import json
from PIL import Image
import tensorflow as tf

min_class_size = 100

def run(dir_braid):
    dir_photos = dir_braid + 'cropped_photos/'
    
    print("Checking the class distribution.")
    class_distribution = {}
    
    for _, dirs, _ in os.walk(dir_photos):
        for dir in dirs:
            for _, _, files in os.walk(f'{dir_photos}/{dir}'):
                file_cnt = len(files)
                class_distribution[dir] = file_cnt
    
    class_distribution = dict(sorted(class_distribution.items(), key=lambda item: item[1], reverse=True))

    with open(f'{dir_braid}/class_distribution.txt', 'w') as distr_file:
        for group in class_distribution:
            distr_file.write(f'{group},{class_distribution[group]}\n')
    distr_file.close()

    print('Used groups: ', end='')
    groups = []
    group_index = {}
    idx = 0
    for group in class_distribution:
        if class_distribution[group] >= min_class_size:
            groups.append(group)
            group_index[group] = idx
            idx += 1
            print(f'{group} ', end='')
    print()

    print('Saving group_index.json')
    with open(f'{dir_braid}group_index.json', "w") as outfile: 
        json.dump(group_index, outfile)

    print("Converting photos to instances.")
    ids = []
    pngs = []

    for group in groups:
        for root, dirs, files in os.walk(f'{dir_photos}{group}'):
            for file in files:
                ids.append(int(file.split('.')[0]))

                file_path = os.path.join(root, file)
                pngs.append([file_path, group])

    ids = np.array(ids, dtype=np.uint32)
    pngs = np.array(pngs)
    total_cnt = len(pngs)

    data_x = []
    data_y = []

    i = 0
    for [filename, group] in pngs:
        img = Image.open(filename)
        img = tf.keras.preprocessing.image.img_to_array(img)

        vector = np.zeros(len(groups))
        vector[group_index[group]] = 1

        data_x.append(img)
        data_y.append(vector)

        i += 1
        if i % 1000 == 0 or i == total_cnt:
            print(f'Converted {i}/{total_cnt}')

    print("Saving data_id.npy")
    np.save(f'{dir_braid}data_id.npy', np.array(ids))

    print('Saving data_x.npy')
    np.save(f'{dir_braid}data_x.npy', np.array(data_x))
    
    print('Saving data_y.npy')
    np.save(f'{dir_braid}data_y.npy', np.array(data_y))
