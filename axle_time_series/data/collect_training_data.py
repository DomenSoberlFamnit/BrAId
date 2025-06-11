import os
from swm import factory, filesys
from swm.vehicle import Vehicle
import datetime
import h5py
import json

admp_ch = 7
axle_ch = 0

def process_events(dir_siwim, dir_braid, vehicles, meta_index, events):
    cnt = 0
    for vehicle in vehicles:
        ts = vehicle.timestamp.timestamp()

        print(vehicle.__dir__())

        print(vehicle.timestamp)
        print(vehicle.event_timestamp)

        quit()

        cnt += 1
#        if cnt % 10000 == 0:
#            print(f'Processed {len(vehicles)} vehicles.')

        if ts in meta_index:
            id = meta_index[ts]

            if (ts, id) in events:
                print(f'Duplicate at {cnt}: {ts}, {id}.')

            events.append((ts, id))

def run(dir_siwim, dir_braid, vehicle_index):
    # In the .nswd structure, there is 'tmstmp' and 'event_timestamp'. In metadata, 'tmstmp' is used, but
    # in the .event files, the timestamp refers to event_timestamp. We have to make a translation to be
    # able to correctly associate the metadata ID (photo of the vehicle) with the time series.
    print('Translating event timestamps.')

    ts_translate = {}
    
    vehicles = Vehicle.from_txt_files(dir_siwim + "sites/AC_Sentvid_2012_2/rp03/cf/2014.nswd", glob=False)
    for vehicle in vehicles:
        ts_translate[vehicle.timestamp.timestamp()] = vehicle.event_timestamp.timestamp()

    vehicles = Vehicle.from_txt_files(dir_siwim + "sites/AC_Sentvid_2012_2/rp03/cf/2015.nswd", glob=False)
    for vehicle in vehicles:
        ts_translate[vehicle.timestamp.timestamp()] = vehicle.event_timestamp.timestamp()

    # From metadata.hdf5 we get the information which events have been checked for correct labels.
    # We take only those that were seen by someone, but not changed, which means there are no arrors.
    print('Collecting IDs of vehicles that have been checked manually.')

    meta_index = {}
    classes = {}

    with h5py.File('../metadata/metadata.hdf5', 'r') as file:
        cnt = 0
        cnt_seen = 0
        cnt_ok = 0

        for groups in file.keys():
            data = file[groups]
            for id in data:
                prop = json.loads(file[f'{groups}/{id}'].asstr()[()])
                segment = prop['segment'] if 'segment' in prop else 'r'
                true_groups = prop['axle_groups'] if 'axle_groups' in prop else groups

                vehicle = vehicle_index[int(id)]
                ts = vehicle['ts_vehicle']  # This is the timestamp from the nswd.
                event_ts = ts_translate[ts] # Translated to the event timestamp (about a second of a difference).

                photo_ok = False

                cnt += 1

                if prop['seen_by'] == None:
                    continue
                
                cnt_seen += 1

                if prop['changed_by'] == None:
                    photo_ok = True
                
                if photo_ok:
                    cnt_ok += 1

                    meta_index[event_ts] = (id, true_groups)

                    if true_groups not in classes:
                        classes[true_groups] = 0
                    classes[true_groups] += 1
        
        classes = dict(sorted(classes.items(), key=lambda item: item[1], reverse=True))

        print(f'Vehicles seen: {cnt_seen}, vehicles OK: {len(meta_index)}.')
        for key in classes:
            print(f'{key}: {classes[key]}')

    # We now have the IDs that we want. Go through all .event files and filter those out.
    # Store the ID, timestamp, series, axles.
    # ID can be used to find the photo.
    print('Collecting the events from the asstrabase.')

    events = []
    cnt = 0

    for root, dirs, files in os.walk(dir_siwim + "sites/AC_Sentvid_2012_2/rp01/cf/", topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            filenameshort = filename.replace(dir_siwim + "sites/AC_Sentvid_2012_2/rp01/cf/", "")
            try:
                event = factory.read_file(filename)
            except:
                print("Error:", filename)
                continue

            cnt += 1
            if cnt % 50000 == 0:
                print(f'Processed {cnt} events, found {len(events)} useful events so far.')

            ts = event.tmstmp.timestamp()
   
            if ts in meta_index:
                (id, groups) = meta_index[ts]

                if (ts, id) in events:
                    print(f'Duplicate at {cnt}: {ts}, {id}.')

                acqdata = event.acqdata
                admp = acqdata.a[admp_ch].data - acqdata.a[admp_ch].offset()
                axles = [int(x[0]) for x in acqdata.d[axle_ch].data]

                error = False

                try:
                    weigh_diag = event.module_trace.last_module('weigh').diags[0][1]
                except:
                    print(f'Error: No weight diags for ID={id}')
                    error = True

                if not error:
                    on = weigh_diag.d[0].data
                    if len(on) > 1:
                        print(f'More than one "on" interval for ID={id}')
                        error = True
                    else:
                        on = tuple(int(x) for x in on[0])

                if error:
                    continue

                trim_from = on[0]
                trim_to = on[1]

                event_data = {
                    'id': id,
                    'timestamp': ts,
                    'admp': admp,
                    'axles': axles,
                    'trim_from': trim_from,
                    'trim_to': trim_to,
                    'groups': groups
                }

                events.append(event_data)

    print(f'Found {len(events)} in the database.')

    print("Saving axle_training_data.json.")
    with open(dir_braid + "axle_training_data.json", "w") as f:
        json.dump(events, f)
