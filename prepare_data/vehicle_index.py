from swm import factory, filesys
from swm.vehicle import Vehicle
import datetime
import json
import os

def find_closest(index, value):
    l = 0
    r = len(index) - 1
    target = None

    while target == None:
        lval = index[l]['timestamp']
        rval = index[r]['timestamp']

        if value < lval or value > rval:
            break

        if value == lval:
            target = l
            continue
        
        if value == rval:
            target = r
            continue

        if r == l + 1:
            dl = value - lval
            dr = rval - value
            target = l if dl < dr else r
            continue

        m = int((l + r) / 2)
        mval = index[m]['timestamp']

        if value < mval:
            r = m
        else:
            l = m
    
    return target

def group2str(group):
    s = ""
    for n in group:
        s += str(n)
    return s

def process_vehicles(dir_siwim, dir_braid, vehicles, photo_index, data):
    skipped = 0

    id = len(data)
    for vehicle in vehicles:
        gvw = vehicle.gvw()

        ts = vehicle.timestamp.timestamp()
        vehicle_ts = datetime.datetime.fromtimestamp(ts)
        target = find_closest(photo_index, ts)
        if target == None:
            skipped += 1
            continue
        
        entry = photo_index[target]
        photo_ts = datetime.datetime.fromtimestamp(entry['timestamp'])

        dts = (photo_ts - vehicle_ts).total_seconds()
        if abs(dts) > 1.0:
            skipped += 1
            continue

        dst_dir = dir_braid + "photos/" + str(int(id / 1000)) + '/'
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
            print("Created " + dst_dir)
        dst_file = dst_dir + str(id) + '.png'

        content = factory.read_file(f'{dir_siwim}sites/AC_Sentvid_2012_2/live/' + entry['filename'])
        image = content.photos[content.best].image()
        image.save(dst_file)

        data.append({'id':id, 'ts_vehicle':ts, 'ts_photo':entry['timestamp'], 'axles':len(vehicle.axle), 'groups':group2str(vehicle.groups), 'gvw':vehicle.gvw(), 'file':str(int(id / 1000)) + '/' + str(id) + '.png'})
        id += 1

    print(f'Skipped {skipped} vehicles.')

def run(dir_siwim, dir_braid):
    if not os.path.exists(f'{dir_braid}photos/'):
        os.mkdir(f'{dir_braid}photos/')

    with open(f'{dir_braid}photo_index.json', "r") as fp:
        photo_index = json.load(fp)

    data = []

    vehicles = Vehicle.from_txt_files(dir_siwim + "sites/AC_Sentvid_2012_2/rp03/cf/2014.nswd", glob=False)
    process_vehicles(dir_siwim, dir_braid, vehicles, photo_index, data)
    print("Finished batch at", len(data))

    vehicles = Vehicle.from_txt_files(dir_siwim + "sites/AC_Sentvid_2012_2/rp03/cf/2015.nswd", glob=False)
    process_vehicles(dir_siwim, dir_braid, vehicles, photo_index, data)
    print("Finished batch at", len(data))

    print("Saving vehicle_index.json.")
    with open(dir_braid + "vehicle_index.json", "w") as fout:
        json.dump(data, fout)
