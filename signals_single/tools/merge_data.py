import os
import json
import h5py
from progress.bar import Bar

dir_braid = '/mnt/workspace/braid/workdir/'

fname_signals = f'{dir_braid}data/nn_normalised_signals.hdf5'
fname_pulses = f'{dir_braid}data/nn_normalised_pulses.json'
fname_signals_extended = f'{dir_braid}data/nn_normalised_signals_extended.hdf5'
fname_pulses_extended = f'{dir_braid}data/nn_normalised_pulses_extended.json'

signals = h5py.File(fname_signals, 'r')
signals_extended = h5py.File(fname_signals_extended, 'r')
output = h5py.File(f'nn_normalised_signals.hdf5', 'w')

replaced = []

bar = Bar('Copying hdf5', max=len(signals))
for ts in signals:
    if ts in signals_extended:
        signals_extended.copy(ts, output)
        replaced.append(str(ts))
    else:
        signals.copy(ts, output)
    bar.next()
bar.finish()

signals.close()
signals_extended.close()
output.close()

print(f"Replaced {len(replaced)} instances:")
for ts in replaced:
    print(ts)

with open(fname_pulses, 'r') as f:
    pulses = json.load(f)
f.close()

with open(fname_pulses_extended, 'r') as f:
    pulses_extended = json.load(f)
f.close()

replaced = []
output = []

bar = Bar('Copying json', max=len(pulses))
for pulse in pulses:
    replacement = None
    for potential in pulses_extended:
        if pulse['ts'] == potential['ts']:
            replacement = potential
            break
    
    if replacement is not None:
        output.append(replacement)
        replaced.append(replacement['ts'])
    else:
        output.append(pulse)

    bar.next()
bar.finish()

with open('nn_normalised_pulses.json', 'w') as f:
    json.dump(output, f, indent=2)
f.close()

print(f"Replaced {len(replaced)} instances:")
for ts in replaced:
    print(ts)