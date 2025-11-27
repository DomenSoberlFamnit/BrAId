import os
import numpy as np

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

def plot_frame(filename, frame, pulse, miny, maxy):
    fig, ax = plt.subplots()
    plt.ylim(miny, maxy)
    ax.plot(frame)
    ax.vlines(x=int(len(frame) / 2), ymin=0, ymax=pulse, color='r', linewidth=3)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))

    plt.savefig(filename)
    plt.close()