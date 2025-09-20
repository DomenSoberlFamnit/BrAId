import os
import numpy as np
import h5py
import json
from datetime import datetime
import shutil

dir_braid = '/home/hicup/disk/braid/'
dir_yolo_photos = f'{dir_braid}yolo_photos/'

year = 2014
month = 3
day = 10

print("Loading recognized_vehicles.json")
f = open(f'{dir_braid}recognized_vehicles.json')
rv = json.load(f)
f.close()

print("Collecting instances.")

distribution = {}

cnt = 0
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
    
    if dt.hour not in distribution:
        distribution[dt.hour] = 0
    
    distribution[dt.hour] += 1

print(distribution)


