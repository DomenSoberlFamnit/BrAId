import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image, ImageDraw, ImageFont

dir_braid = '/mnt/workspace/braid/workdir/'
signal_name = '11admp'

fname_signals = f'{dir_braid}data/nn_normalised_signals.hdf5'
fname_pulses = f'{dir_braid}data/nn_normalised_pulses.json'

def signal_trim_back(raw_signal, last_pulse_idx, threshold):
    idx = last_pulse_idx
    
    while idx < len(raw_signal) and raw_signal[idx] > 0:
        idx += 1
    
    while idx < len(raw_signal) and raw_signal[idx] < 0:
        idx += 1
    
    amount = 0
    while idx < len(raw_signal):
        amount += abs(raw_signal[idx])
        raw_signal[idx] = 0
        idx += 1

    return amount

with open(f'{dir_braid}camera/recognized_vehicles.json', 'r') as file:
    vehicle_info = json.load(file)
file.close()


signals = h5py.File(fname_signals, 'r')

with open(fname_pulses, 'r') as file:
    pulses_file = json.load(file)

for pulse in pulses_file:
    ts = pulse['ts_str']

    X = np.array(signals[ts][signal_name])
    minmag = max(0, round(-200*np.min(X)) + 1)

    raw = np.copy(X)
    amount = signal_trim_back(X, pulse['vehicle']['final']['axle_pulses'][-1], 0.02)
    #X = raw

    if amount == 0:
        continue

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

    if amount >= 50:
        fname = f'plot_{ts}_cut.png'
        print(f'Saving {fname} | amount = {amount}')
        img.save(fname)

signals.close()