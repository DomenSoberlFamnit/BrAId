import os
import numpy as np

def max_filter(signal, threshold, kernel_size):
    filtered = signal.copy()

    eps = kernel_size // 2
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