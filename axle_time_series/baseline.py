import json
import h5py

def accuracy(dir_braid, normalized_signals):
    if normalized_signals:
        fname_signals = 'nn_normalised_signals.hdf5'
        fname_pulses = 'nn_normalised_pulses.json'
    else:
        fname_signals = 'nn_signals.hdf5'
        fname_pulses = 'nn_pulses.json'
    
    print(f'Analyzing the signals.')

    sum_correct = 0
    count = 0

    with h5py.File(f'{dir_braid}data/{fname_signals}', 'r') as signals:
        with open(f'{dir_braid}data/{fname_pulses}', 'r') as file:
            pulses = json.load(file)

            for pulse in pulses:
                detected = pulse['vehicle']['detected']['axle_distance']
                weighed = pulse['vehicle']['weighed']['axle_distance']
                final = pulse['vehicle']['final']['axle_distance']

                correct = detected == weighed and weighed == final
                if correct:
                    sum_correct += 1
                count += 1

    return sum_correct / count
                