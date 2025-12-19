import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_prediction(filename, signal, pulses, prediction, comment=''):
    bars = []
    for i, pulse in enumerate(pulses):
        if pulse > 0:
            bars.append(i)

    maxy = np.max(signal)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(signal, color='b')
    ax.plot(prediction * maxy/10, color='m')
    ax.vlines(x=[x for x in bars], ymin=-maxy/10, ymax=0, color='g')
    ax.axhline(color='k', linewidth=0.5, linestyle='--')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(512))

    ax.text(0, 1.1 * maxy, comment, fontsize=10, color="black")

    plt.savefig(filename, dpi=300)
    plt.close()

def plot_frame(filename, frame, pulse, miny, maxy):
    fig, ax = plt.subplots()
    plt.ylim(miny, maxy)
    ax.plot(frame)
    ax.vlines(x=int(len(frame) / 2), ymin=0, ymax=pulse, color='r', linewidth=3)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    plt.savefig(filename)
    plt.close()