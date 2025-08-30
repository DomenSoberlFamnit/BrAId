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

def count_axles(sample):
    cnt = 0
    for values in sample:
        pass

def evaluate(model, X, Y):
    print("Evaluating the model.")
    predictions = model.predict(X)

    #file = open("results.txt", "w")

    cnt = 0
    for (signal, pulses, prediction) in zip (X, Y, predictions):
        if cnt % 10 != 0:
            cnt += 1
            continue

        plot_sample(f'validation_{cnt}.png', signal, pulses, prediction)

        #file.write(f'{cnt}')
        #for p in prediction:
        #    file.write(f',{p}')
        #file.write('\n')

        cnt += 1
    
    #file.close()