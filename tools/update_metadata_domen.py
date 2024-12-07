import csv
import json
import h5py

corrections = {}
with open('../metadata/corrections.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        corrections[row[0]] = {'original': row[1], 'manual': row[2]}

cnt_changed = 0
cnt_eliminated = 0

with h5py.File('../metadata/metadata-corrections-domen.hdf5', 'w') as new_meta:
    with h5py.File('../metadata/metadata-original.hdf5', 'r') as meta:
        for groups in meta.keys():
            data = meta[groups]
            new_data = new_meta.create_group(groups)
            for id in data:
                rec = json.loads(data[id].asstr()[()])

                if id in corrections:
                    original = corrections[id]['original']
                    manual = corrections[id]['manual']
                    if manual == 'x' or manual == '?':
                        if not 'errors' in rec:
                            rec['errors'] = {}
                        rec['errors']['cannot_label'] = 2
                        cnt_eliminated += 1
                    elif manual != 'ok':
                        rec['axle_groups'] = manual
                        cnt_changed += 1
                        
                new_rec = new_data.create_dataset(id, data=json.dumps(rec))

print(f'Changed: {cnt_changed}, eliminated: {cnt_eliminated}.')
