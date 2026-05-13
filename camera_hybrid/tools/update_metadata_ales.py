import csv
import json
import h5py

corrections = {}
with open('../metadata/corrections-ales.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        corrections[row[0]] = row[1]

cnt_all = 0
cnt_changed = 0
cnt_eliminated = 0
testing_ids = []

with h5py.File('../metadata/metadata-corrections-ales.hdf5', 'w') as new_meta:
    with h5py.File('../metadata/metadata-original.hdf5', 'r') as meta:
        for groups in meta.keys():
            data = meta[groups]
            new_data = new_meta.create_group(groups)
            for id in data:
                rec = json.loads(data[id].asstr()[()])

                if id in corrections:
                    manual = corrections[id]
                    cnt_all += 1

                    current_groups = groups if 'axle_groups' not in rec else rec['axle_groups']

                    if len(manual) == 0 or manual == current_groups:
                        testing_ids.append(id)
                        print(f'{id} kept {current_groups}.')
                    elif manual == 'x':
                        if not 'errors' in rec:
                            rec['errors'] = {}
                        rec['errors']['cannot_label'] = 2
                        cnt_eliminated += 1
                        print(f'{id} eliminated.')
                    else:
                        rec['axle_groups'] = manual
                        cnt_changed += 1
                        testing_ids.append(id)
                        print(f'{id} changed {current_groups} to {manual}.')
                        
                new_rec = new_data.create_dataset(id, data=json.dumps(rec))

print(f'All: {cnt_all}, Changed: {cnt_changed}, eliminated: {cnt_eliminated}.')

with open('../metadata/testing_ids.json', 'w') as f:
    json.dump(testing_ids, f)