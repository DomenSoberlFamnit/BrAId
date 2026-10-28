import os
import numpy as np
from scipy.signal import find_peaks
from progress.bar import Bar

import dataset
import evaluation

dir_braid = '/mnt/workspace/braid/workdir_single/'
signal_length = 1280

#####################   Data preparation   ########################################

# Load or generate the vehicle index (a leftover from the camera project).
vehicle_info = dataset.load_vehicle_info(dir_braid, f'{dir_braid}camera/')
if vehicle_info is None:
    print('Cannot load vehicle info.')
    quit()

data_train = dataset.get_data(
     dir_braid=dir_braid,
     input_signal='11admp',
     output_type='pulses',
     normalized_signals=True,
     include_correct=True,
     include_fixed=False,
     signal_length=signal_length,
     shuffle=False,
     gen_daily_dist=False,
     exclude_days=['2014-4-3']
 )

data_test = dataset.get_data(
     dir_braid=dir_braid,
     input_signal='11admp',
     output_type='pulses',
     normalized_signals=True,
     include_correct=True,
     include_fixed=False,
     signal_length=signal_length,
     shuffle=False,
     gen_daily_dist=False,
     include_days=['2014-4-3']
 )

########################################   Evaluate Method   ########################################

def evaluate_predictions(data, peak_distance, peak_prominence, tolerance, progress=True):
    (meta, X, Y) = data

    sum_tp = 0
    sum_fn = 0
    sum_fp = 0
    sum_mae = 0
    axles = 0

    if progress:
        bar = Bar('Evaluating predictions', max=len(meta))

    for (m, signal, pulses) in zip(meta, X, Y):
        ts = m[0]
        id = vehicle_info[ts]['id']
        photo = vehicle_info[ts]['photo']

        pulse_indices = np.flatnonzero(pulses)
        peak_indices, peak_properties = find_peaks(signal, distance=peak_distance, prominence=peak_prominence)       

        tp, fn, fp, mae = evaluation.prediction_metrics(pulse_indices, peak_indices, tolerance=tolerance)
        axles += len(pulse_indices)

        sum_tp += tp
        sum_fn += fn
        sum_fp += fp
        sum_mae += mae

        if progress:
            bar.next()

    if progress:
        bar.finish()

    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score, sum_mae / len(meta), sum_tp, sum_fn, sum_fp, axles


# Cross validation on the training set.
fold_n = 10
results = {
    'precision': [],
    'recall': [],
    'f1_score': [],
    'mae': []
}

for fold_k in range(fold_n):
    _, data_validate = dataset.split_fold(data_train, fold_k, fold_n)
    precision, recall, f1_score, mae, _, _, _, _ = evaluate_predictions(data_validate, peak_distance=10, peak_prominence=0.02, tolerance=3)

    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1_score'].append(f1_score)
    results['mae'].append(mae)

print(f"Precision: {np.mean(results['precision']):.4f} ± {np.std(results['precision']):.4f}, Recall: {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}, F1-Score: {np.mean(results['f1_score']):.4f} ± {np.std(results['f1_score']):.4f}, MAE: {np.mean(results['mae']):.4f} ± {np.std(results['mae']):.4f}")

# Validation on testing set.
precision, recall, f1_score, mae, tp, fn, fp, axles = evaluate_predictions(data_test, peak_distance=10, peak_prominence=0.02, tolerance=1)

print("Test set:")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, MAE: {mae:.4f}")
print(f"Axles: {axles}, TP: {tp}, FN: {fn}, FP: {fp}")