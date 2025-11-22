import os
import numpy as np

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

def sharpen_prediction(prediction):
    pulses = []

    max = 0
    idx = 0
    for i, p in enumerate(prediction):
        value = round(p, 2)

        if value == 0:
            if max != 0:
                pulses.append((idx, max))
            max = 0
        else:
            if value > max:
                max = value
                idx = i
            
    sharp = np.zeros(len(prediction))

    for (idx, max) in pulses:
        sharp[idx] = max
    
    return sharp

def sample_accuracy(pulses, prediction, class_threshold, kernel_size):
    tp, fn, fp = 0, 0, 0

    places_checked = []

    # Iterate through the pulses, find TP and FN.
    for i, p in enumerate(pulses):
        value = round(p, 2)

        # If found a pulse.
        if value > 0.5:  # == 1.0 should also work.
            # Check the neighborhood (kernel).
            hit = False
            for j in range(kernel_size):
                idx = i + j - int(kernel_size/2)
                if prediction[idx] >= class_threshold:
                    hit = True
                places_checked.append(idx)

            if hit:
                tp += 1
            else:
                fn += 1

    # Find FP
    for i, p in enumerate(prediction):
        if i in places_checked:
            continue

        value = round(p, 2)
        if value >= class_threshold:
            fp += 1
    
    return tp, fn, fp

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