import os
import json
import h5py
from progress.bar import Bar

dir_braid = '/mnt/workspace/braid/workdir/'
siwim_phase = 'detected'
#siwim_phase = 'weighed'
database = 'fixed'

pulse_tolerance = 4

def generate_vehicle_id():
    fpath_index = f'{dir_braid}camera/vehicle_index.json'

    with open(fpath_index, 'r') as file:
        vehicle_index = json.load(file)
    file.close()

    bar = Bar('Generating vehicle info', max=len(vehicle_index))
    vehicle_id = {}
    for vehicle in vehicle_index:
        photo_id = vehicle['id']
        ts = vehicle['ts_vehicle']
        vehicle_id[str(ts)] = photo_id
        bar.next()
    bar.finish()

    return vehicle_id

def get_siwim_pulses(phase, vehicle_id):
    signals_file = h5py.File(f'{dir_braid}data/nn_normalised_signals.hdf5', 'r')

    file = open(f'{dir_braid}data/nn_normalised_pulses.json', 'r')
    json_pulses = json.load(file)
    file.close()

    bar = Bar('Generating SiWIM pulses', max=len(json_pulses))
    siwim_pulses = {}
    for pulses in json_pulses:
        ts = pulses['ts']
        photo_id = vehicle_id[str(ts)]

        signal_length = len(signals_file[pulses['ts_str']]['11admp'])
        offset = int((1300 - signal_length) / 2)
        offset = offset if offset >= 0 else 0

        output_pulses = []
        for value in pulses['vehicle'][phase]['axle_pulses']:
            output_pulses.append(value + offset)
        siwim_pulses[str(photo_id)] = output_pulses

        bar.next()
    bar.finish()

    signals_file.close()

    return siwim_pulses
    

def distance_from_pulse(value, pulses):
    min_dist = None

    for pulse in pulses:
        dist = abs(value - pulse)
        if min_dist is None or dist < min_dist:
            min_dist = dist
    
    return min_dist

def evaluate_pulses(pulses_predicted, pulses_truth):
    cnt_pulses = len(pulses_truth)
    cnt_pulses_correct = 0
    error = 0
    missed = 0
    ghost = 0

    if len(pulses_predicted) == len(pulses_truth):
        pulses_predicted.sort()
        pulses_truth.sort()

        for i in range(len(pulses_truth)):
            dist = abs(pulses_predicted[i] - pulses_truth[i])
            
            if dist <= pulse_tolerance:
                cnt_pulses_correct += 1
            error += dist

    elif len(pulses_predicted) < len(pulses_truth):
        for predicted in pulses_predicted:
            dist = distance_from_pulse(predicted, pulses_truth)

            if dist <= pulse_tolerance:
                cnt_pulses_correct += 1
            error += dist

        missed += len(pulses_truth) - len(pulses_predicted)  
    
    elif len(pulses_predicted) > len(pulses_truth):
        for truth in pulses_truth:
            dist = distance_from_pulse(truth, pulses_predicted)

            if dist <= pulse_tolerance:
                cnt_pulses_correct += 1
            error += dist

        ghost += len(pulses_predicted) - len(pulses_truth)

    return cnt_pulses, cnt_pulses_correct, error, missed, ghost

def main():
    siwim_pulses_index = get_siwim_pulses(siwim_phase, generate_vehicle_id())

    file = open(f'{dir_braid}data/ground_truth.json', 'r')
    data_json = json.load(file)
    file.close()

    cnt_events = 0
    cnt_axles = 0
    cnt_events_correct_siwim = 0
    cnt_axles_correct_siwim = 0
    cnt_events_correct_ai = 0
    cnt_axles_correct_ai = 0
    sum_error_siwim = 0
    sum_error_ai = 0
    missed_siwim = 0
    ghost_siwim = 0
    missed_ai = 0
    ghost_ai = 0

    for photo_id in data_json:
        record = data_json[photo_id]

        if record['database'] != database:
            continue

        if record['keep'] == False:
            continue

        # siwim_pulses = record['siwim_pulses']
        siwim_pulses = siwim_pulses_index[str(photo_id)]
        ai_pulses = record['ai_pulses']
        
        if record['adjusted'] == False:
            expert_pulses = record['expert_pulses']
        else:
            expert_pulses = record['adjusted_pulses']

        expert_pulses = record['expert_pulses']

        cnt_axles += len(expert_pulses)
        cnt_events += 1

        cnt_pulses, cnt_pulses_correct, error, missed, ghost = evaluate_pulses(siwim_pulses, expert_pulses)

        if cnt_pulses == cnt_pulses_correct:
            cnt_events_correct_siwim += 1
        
        cnt_axles_correct_siwim += cnt_pulses_correct
        sum_error_siwim += error

        missed_siwim += missed
        ghost_siwim += ghost

        cnt_pulses, cnt_pulses_correct, error, missed, ghost = evaluate_pulses(ai_pulses, expert_pulses)

        if cnt_pulses == cnt_pulses_correct:
            cnt_events_correct_ai += 1
        
        cnt_axles_correct_ai += cnt_pulses_correct
        sum_error_ai += error

        missed_ai += missed
        ghost_ai += ghost

    print('SIWIM results:')
    print(f'Events: {cnt_events_correct_siwim} / {cnt_events} = {(100 * cnt_events_correct_siwim / cnt_events):.4f}')
    print(f'Axles: {cnt_axles_correct_siwim} / {cnt_axles} = {(100 * cnt_axles_correct_siwim / cnt_axles):.4f}')
    print(f'MAE: {sum_error_siwim / cnt_axles}')
    print(f'Missed: {missed_siwim} / {cnt_events} = {(missed_siwim / cnt_events):.4f}')
    print(f'Ghost: {ghost_siwim} / {cnt_events} = {(ghost_siwim / cnt_events):.4f}')

    print()
    print('AI results:')
    print(f'Events: {cnt_events_correct_ai} / {cnt_events} = {(100 * cnt_events_correct_ai / cnt_events):.4f}')
    print(f'Axles: {cnt_axles_correct_ai} / {cnt_axles} = {(100 * cnt_axles_correct_ai / cnt_axles):.4f}')
    print(f'MAE: {sum_error_ai / cnt_axles}')
    print(f'Missed: {missed_ai} / {cnt_events} = {(missed_ai / cnt_events):.4f}')
    print(f'Ghost: {ghost_ai} / {cnt_events} = {(ghost_ai / cnt_events):.4f}')

main()
