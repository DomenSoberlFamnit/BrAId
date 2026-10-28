import os
import numpy as np
from progress.bar import Bar

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_sample(filename, signal, pulses, prediction):
    bars = []
    cnt = 0
    for pulse in pulses:
        if pulse > 0:
            bars.append(cnt)
        cnt += 1

    maxy = np.max(signal)

    fig, ax = plt.subplots()
    ax.plot(signal)
    ax.plot(prediction * maxy/10)
    ax.vlines(x=[x for x in bars], ymin=-maxy/10, ymax=0, color='r')
    ax.axhline(color='k', linewidth=0.5, linestyle='--')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(512))

    plt.savefig(filename)
    plt.close()

def distances_to_pulses(distances, width=1300, offset=10):
    pulses = np.zeros(width)

    for position in distances:
        pulses[offset + round(position)] = 1.0

    return pulses

def max_filter(signal, threshold, kernel_size):
    filtered = signal.copy()

    eps = int(kernel_size / 2)
    for i in range(len(signal)):
        start = max(0, i - eps)
        end = min(i + eps + 1, len(signal))

        kernel = filtered[start:end]
        idx_max = np.argmax(kernel)
        max_value = kernel[idx_max]

        filtered[start:end] = 0
        if max_value >= threshold:
            filtered[start + idx_max] = max_value
    
    return filtered

def sample_accuracy(pulses, prediction, class_threshold, kernel_size):
    tp, fn, fp = 0, 0, 0
    n = 0
    error = 0

    places_checked = []

    # Iterate through the pulses, find TP and FN.
    for i, p in enumerate(pulses):
        value = round(p, 2)

        # If found a pulse.
        if value > 0.5:  # == 1.0 should also work.
            # Check the neighborhood (kernel).
            epsilon = int(kernel_size / 2)
            max_prediction = np.max(prediction[(i - epsilon):(i + epsilon + 1)])
            places_checked = list(set(places_checked + list(range(i - epsilon, i + epsilon + 1))))

            if max_prediction >= class_threshold:
                tp += 1
            else:
                fn += 1
            
            n += 1
            error += abs(max_prediction - value)

    # Find FP
    for i, p in enumerate(prediction):
        if i in places_checked:
            continue

        value = round(p, 2)
        if value >= class_threshold:
            fp += 1
        
        error += abs(value)
        n += 1
    
    return tp, fn, fp, (error / n)

def sample_thresholds(pulses, prediction, kernel_size):
    thrs_t = []
    thrs_f = []
   
    for i, p in enumerate(prediction):
        value = round(p, 2)

        if value == 0:
            continue
            
        hit = False
        for j in range(kernel_size):
            idx = i + j - int(kernel_size/2)
            if pulses[idx] >= 0.5:  # == 1.0 should also work.
                hit = True
        
        if hit:
            thrs_t.append(value)
        else:
            thrs_f.append(value)
    
    return thrs_t, thrs_f

def prediction_metrics(pulses, prediction, tolerance):
    tp = 0
    fn = 0
    fp = 0
    mae = 0
    count = 0

    if len(prediction) == 0:
        fn = len(pulses)
        return tp, fn, fp, mae

    available = prediction.copy()

    for axle in pulses:
        if len(available) == 0:
            fn += 1
            continue

        # Find the closest prediction within the tolerance range.
        distances = abs(available - axle)
        closest = np.argmin(distances)

        if distances[closest] <= tolerance:
            tp += 1
            mae += distances[closest]
            count += 1
            available = np.delete(available, closest)
        else:
            fn += 1

    fp = len(available)  # Remaining predictions are false positives.
    mae = mae / count if count > 0 else 0

    return tp, fn, fp, mae

def evaluate_predictions(predictions, pulses, class_threshold=0.5, filter_kernel_size=9, tolerance=3, progress=True, save_confidence_data=True):
    sum_tp = 0
    sum_fn = 0
    sum_fp = 0
    sum_mae = 0
    axles = 0

    if progress:
        bar = Bar('Evaluating predictions', max=len(predictions))

    for (pulse, prediction) in zip(pulses, predictions):
        prediction = prediction.flatten()

        pulse_indices = np.flatnonzero(pulse)

        if filter_kernel_size is not None and filter_kernel_size > 1:
            prediction = max_filter(prediction, class_threshold, filter_kernel_size)
        
        detected_positions = np.where(prediction >= class_threshold)[0]

        tp, fn, fp, mae = prediction_metrics(pulse_indices, detected_positions, tolerance=tolerance)
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
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score, sum_mae / len(predictions), sum_tp, sum_fn, sum_fp, axles

def compute_pr_auc(pulses, predictions, filter_kernel_size=9, tolerance=3, model_name=None, output_dir='.'):
    assert model_name is not None, "Model name must be provided for saving PR curve plots and CSV."

    plot_path = os.path.join(output_dir, f'{model_name}_pr_curve.png')
    csv_path = os.path.join(output_dir, f'{model_name}_pr_curve.csv')

    pr_points = []

    bar = Bar('Evaluating class thresholds', max=100)
    for threshold in np.arange(0.0, 1.01, 0.01):
        precision, recall, f1_score, mae, tp, fn, fp, axles = evaluate_predictions(
            predictions=predictions,
            pulses=pulses,
            class_threshold=threshold,
            filter_kernel_size=filter_kernel_size,
            tolerance=tolerance,
            progress=False
        )

        pr_points.append((recall, precision, threshold, f1_score))

        bar.next()
    bar.finish()

    if not pr_points:
        return 0.0, None

    best_threshold = max(pr_points, key=lambda point: point[3])[2]
    pr_points.sort(key=lambda x: x[0])

    recalls = np.asarray([point[0] for point in pr_points], dtype=float)
    precisions = np.asarray([point[1] for point in pr_points], dtype=float)
    thresholds = np.asarray([point[2] for point in pr_points], dtype=float)

    order = np.argsort(recalls, kind='mergesort')
    recalls = recalls[order]
    precisions = precisions[order]
    thresholds = thresholds[order]

    recalls_for_auc = np.concatenate(([0.0], recalls, [1.0]))
    precisions_for_auc = np.concatenate(([1.0], precisions, [precisions[-1] if precisions.size > 0 else 1.0]))
    thresholds_for_csv = np.concatenate(([1.0], thresholds, [1.0]))

    pr_auc = float(np.sum((recalls_for_auc[1:] - recalls_for_auc[:-1]) * precisions_for_auc[1:]))

    csv_data = np.column_stack((thresholds_for_csv, recalls_for_auc, precisions_for_auc))
    np.savetxt(csv_path, csv_data, delimiter=',', header='threshold,recall,precision', comments='')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(recalls_for_auc, precisions_for_auc, where='post', color='tab:blue', linewidth=1.5)
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return pr_auc, best_threshold