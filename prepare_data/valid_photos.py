import h5py
import json

error_flags = ['yolo_error', 'photo_truncated', 'vehicle_joined', 'vehicle_split', 'cannot_label', 'inconsistent_data', 'off_lane', 'wrong_lane', 'multiple_vehicles', 'fixed']

def prop_has_errors(prop):
    if 'errors' not in prop:
        return False
    
    errors = prop['errors']

    for flag in error_flags:
        if flag in errors and errors[flag] != 0:
            return True

    return False

def run(dir_braid):
    with h5py.File('../metadata/metadata.hdf5', 'r') as file:
        cnt = 0
        cnt_seen = 0
        cnt_changed = 0
        cnt_changed_ok = 0
        cnt_ok = 0

        photos = []

        for groups in file.keys():
            data = file[groups]
            for id in data:
                prop = json.loads(file[f'{groups}/{id}'].asstr()[()])
                segment = prop['segment'] if 'segment' in prop else 'r'
                true_groups = prop['axle_groups'] if 'axle_groups' in prop else groups

                cnt += 1
                if cnt % 10000 == 0:
                    print(f'Processed {cnt} photos.')

                photo_ok = False

                if prop['seen_by'] == None:
                    continue

                cnt_seen += 1

                if prop['changed_by'] == None:
                    photo_ok = True
                else:
                    cnt_changed += 1
                    if not prop_has_errors(prop):
                        photo_ok = True
                        if true_groups != groups:
                            cnt_changed_ok += 1
                
                if photo_ok:
                    cnt_ok += 1
                    photos.append({'photo_id': id, 'segment': segment, 'class': true_groups})
        
    with open(f'{dir_braid}valid_photos.json', 'w') as file:
        json.dump(photos, file)
    
    print(f'All: {cnt}, seen: {cnt_seen}, ok: {cnt_ok}, changed: {cnt_changed}, changed and ok {cnt_changed_ok}.')