import os
import json
import h5py
import datetime
from progress.bar import Bar

dir_braid = '/mnt/workspace/braid/workdir_single/'
dir_vehicle_index = '/mnt/workspace/braid/workdir_camera/'
dir_pos_temp = '/mnt/workspace/braid/BrAId/camera_hybrid/metadata/'
fname_signals = 'nn_normalised_signals.hdf5'
fname_pulses = 'nn_normalised_pulses.json'

def load_vehicle_info(dir_braid, dir_vehicle_index):
    fpath_info = f'{dir_braid}vehicle_info.json'
    fpath_index = f'{dir_vehicle_index}vehicle_index.json'

    if os.path.exists(fpath_info):
        with open(fpath_info, 'r') as file:
            vehicle_info = json.load(file)
        file.close()
        return vehicle_info
    
    if not os.path.exists(fpath_index):
        return None

    with open(fpath_index, 'r') as file:
        vehicle_index = json.load(file)
    file.close()

    bar = Bar('Generating vehicle info', max=len(vehicle_index))
    vehicle_info = {}
    for vehicle in vehicle_index:
        vehicle_info[str(vehicle['ts_vehicle'])] = {
            'id': vehicle['id'],
            'photo': f'{dir_braid}camera/photos/{vehicle['file']}'
        }

        bar.next()
    bar.finish()

    with open(fpath_info, "w") as file:
        json.dump(vehicle_info, file)
    file.close()

    return vehicle_info

def load_pos_temp(dir_pos_temp):
    with open(f'{dir_pos_temp}pos_and_temp.json', 'r') as file:
        json_pos_temp = json.load(file)
        file.close()

    data = {}
    for rec in json_pos_temp:
        s1 = rec['ts'].split('T')
        sd = s1[0].split('-')
        year = sd[0]
        month = sd[1]
        day = sd[2]
        st = s1[1].split('.')[0].split(':')
        hour = st[0]
        minute = st[1]
        second = st[2]

        id_str = f'{year}-{month}-{day}-{hour}-{minute}-{second}'
        data[id_str] = {
            'location': rec['loc'],
            'gvw': rec['gvw'],
            'temperature': rec['T'],
            'w1': rec['w1']
        }

    return data

pos_temp = load_pos_temp(dir_pos_temp)
vehicle_info = load_vehicle_info(dir_braid, dir_vehicle_index)

with open(f'{dir_braid}data/{fname_pulses}', 'r') as file:
    pulses_file = json.load(file)
    file.close()

data = []

bar = Bar('Exporting instances', max=len(pulses_file))
for pulse in pulses_file:
    ts = str(pulse['ts'])
    ets = str(pulse['ets'])

    if ts not in vehicle_info:
        continue

    id = vehicle_info[ts]['id']

    time_components = pulse['ts_str'].split('-')
    year = time_components[0]
    month = time_components[1]
    day = time_components[2]
    hour = time_components[3]
    minute = time_components[4]
    second = time_components[5]

    key = f'{year}-{month}-{day}-{hour}-{minute}-{second}'

    if key not in pos_temp:
        continue

    data.append({
        'id': id,
        'date': {
            'year': year,
            'month': month,
            'day': day
        },
        'time': {
            'hour': hour,
            'minute': minute,
            'second': second
        },
        'speed': pulse['v'],
        'gvw': pos_temp[key]['gvw'],
        'location': pos_temp[key]['location'],
        'w1': pos_temp[key]['w1'],
        'temperature': pos_temp[key]['temperature'],
        'axles': {
            'detected': {
                'distances': pulse['vehicle']['detected']['axle_distance'],
                'groups': pulse['vehicle']['detected']['axle_groups']
            },
            'weighed': {
                'distances': pulse['vehicle']['weighed']['axle_distance'],
                'weights': pulse['vehicle']['weighed']['axle_weight'],
                'groups': pulse['vehicle']['weighed']['axle_groups']
            },
            'final': {
                'distances': pulse['vehicle']['final']['axle_distance'],
                'weights': pulse['vehicle']['final']['axle_weight'],
                'groups': pulse['vehicle']['final']['axle_groups']
            }
        }
    })

with open('sentvid_meta.json', 'w') as file:
    json.dump(data, file, indent=4)
