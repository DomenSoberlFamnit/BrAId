import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image, ImageDraw, ImageFont

dir_braid = '/mnt/workspace/braid/workdir/'
signal_name = '11admp'

ids = [201539, 216478, 223271, 64608, 237727, 50573, 50892, 59133, 99015, 122037, 136040, 150012, 151488, 200207, 203385, 203684, 207279, 217619, 227095, 245567]

fname_signals = f'{dir_braid}data/unprocessed/nn_normalised_signals.hdf5'
fname_pulses = f'{dir_braid}data/unprocessed/nn_normalised_pulses.json'

#fname_signals = f'{dir_braid}data/unprocessed/nn_normalised_signals_extended.hdf5'
#fname_pulses = f'{dir_braid}data/unprocessed/nn_normalised_pulses_extended.json'

#fname_signals = f'{dir_braid}data/nn_normalised_signals.hdf5'
#fname_pulses = f'{dir_braid}data/nn_normalised_pulses.json'


with open(f'{dir_braid}camera/recognized_vehicles.json', 'r') as file:
    vehicle_info = json.load(file)
file.close()

photo_ids = {}
for record in vehicle_info:
    if record['photo_id'] in ids:
        photo_ids[record['vehicle_timestamp']] = record['photo_id']

signals = h5py.File(fname_signals, 'r')

with open(fname_pulses, 'r') as file:
    pulses_file = json.load(file)

for pulse in pulses_file:
    if pulse['ts'] not in photo_ids:
        continue

    photo_id = photo_ids[pulse['ts']]
    ts = pulse['ts_str']

    X = np.array(signals[ts][signal_name])
    minmag = max(0, round(-200*np.min(X)) + 1)

    width = max(max(pulse['vehicle']['final']['axle_pulses']) + 10, len(X))

    Y = np.zeros(width)
    for location in pulse['vehicle']['final']['axle_pulses']:
        Y[location] = 1
    
    img = Image.new(mode="RGB", size=(width, 200 + minmag), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    points = [[(x, 200), (x, 0)] for (x, pulse) in enumerate(Y) if pulse > 0]
    cnt_pulses = len(points)
    for point in points:
        draw.line(point, fill=(255, 0, 0), width=3)
    
    points = [(x, 200 - 200*y) for (x, y) in enumerate(X)]
    draw.line(points, fill=(0, 0, 255), width=2)

    draw.line([(0, 200), (width, 200)], fill=(0, 0, 0), width=1)

    fname = f'plot_{photo_id}_cut.png'
    print(f'Saving {fname}')
    img.save(fname)

signals.close()