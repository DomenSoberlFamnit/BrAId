import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

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

def plot_testing_sample(filename, signal, pulses, predictions, threshold, meta):
    img = Image.new(mode="RGB", size=(1300, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Shaded plot area
    draw.rectangle([
        (0, 500), (1300, 790)
        ], fill=(240, 252, 252), outline=(0, 0, 0), width=0)

    # Shaded tolerance
    draw.rectangle([
        (0, 700 - 200*threshold), (1300, 700)
        ], fill=(240, 224, 252), outline=(0, 0, 0), width=0)

    # Pulses (ground truth)
    points = [[(x, 790), (x, 500)] for (x, pulse) in enumerate(pulses) if pulse > 0]
    cnt_pulses = len(points)
    for point in points:
        draw.line(point, fill=(240, 192, 0), width=3)

    # Predictions
    points = [([(x, 700), (x, 700 - 200*p)], p) for (x, p) in enumerate(predictions) if p > 0]
    cnt_predictions = len(points)
    for (point, p) in points:
        if p >= threshold:
            draw.line(point, fill=(255, 0, 0), width=3)
        else:
            draw.line(point, fill=(0, 164, 0), width=3)

    # Signal
    points = [(x, 700 - 200*y) for (x, y) in enumerate(signal)]
    draw.line(points, fill=(0, 0, 255), width=2)

    # Horizontal lines
    draw.line([(0, 700), (1300, 700)], fill=(0, 0, 0), width=1)
    draw.line([(0, 700 - 200*threshold), (1300, 700 - 200*threshold)], fill=(0, 0, 0), width=1)
    draw.line([(0, 500), (1300, 500)], fill=(0, 0, 0), width=1)
    draw.line([(0, 790), (1300, 790)], fill=(0, 0, 0), width=1)

    # Embedded image
    embedded = Image.open(meta['image'])
    img.paste(embedded, (650, 10))

    # Information
    font = ImageFont.truetype("/usr/share/fonts/TTF/Consolas-Regular.ttf", 28)

    timestamp = str(datetime.fromtimestamp(float(meta['ts'])))
    draw.text((10, 10), timestamp, fill=(0, 0, 0), font=font)
    draw.text((10, 70), f'Axle groups: {meta['groups']}', fill=(0, 0, 0), font=font)
    
    if meta['siwim']:
        draw.text((10, 110), f'SiWIM: correct', fill=(0, 192, 0), font=font)
    else:
        draw.text((10, 110), f'SiWIM: incorrect', fill=(224, 0, 0), font=font)
    
    if meta['ai']:
        draw.text((10, 150), f'AI: correct', fill=(0, 192, 0), font=font)
    else:
        draw.text((10, 150), f'AI: incorrect', fill=(224, 0, 0), font=font)
    
    
    draw.text((10, 210), f'Found: {cnt_predictions}/{cnt_pulses}', fill=(0, 0, 0), font=font)
    draw.text((10, 250), f'Missed (FN): {meta['missed']}', fill=(0, 0, 0), font=font)
    draw.text((10, 290), f'Ghost (FP): {meta['ghost']}', fill=(0, 0, 0), font=font)

    draw.text((10, 370), f'Samples: 1300', fill=(0, 0, 0), font=font)
    draw.text((10, 410), f'Axle threshold: {threshold}', fill=(0, 0, 0), font=font)
    draw.text((10, 450), f'Axle tolerance: 1 sample', fill=(0, 0, 0), font=font)
    
    img.save(filename)