import os
import csv
import numpy as np

dir_braid = '/home/hicup/disk/braid/'

corr = {'VGG16': 4.57, 'VGG19': 4.15, 'DenseNet121': 4.94, 'MobileNetV3Small': 7.39, 'ResNet101V2': 5.31}

def read_values(filename, architecture, data):
    if architecture not in data:
        data[architecture] = {}
    values = data[architecture]

    with open(filename) as csvfile:
        header = True
        for row in csv.reader(csvfile, delimiter=','):
            if header:
                header = False
                continue
            
            samples = int(int(row[1].strip()) / 6500)
            loss = round(float(row[2].strip()), 4)
            acc = round(float(row[4].strip()) * 100, 2) #+ corr[architecture]

            if samples not in values:
                values[samples] = []
            values[samples].append(loss)

    csvfile.close()

def compute_points(data):
    points = {}
    for architecture in data:
        points[architecture] = []
        for samples in data[architecture]:
            mean = np.mean(data[architecture][samples])
            stdev = np.std(data[architecture][samples])
            points[architecture].append((samples/10.0, mean, stdev))
    return points

def print_points(points):
    for architecture in points:
        print(f'\\def\\{architecture}{{', end='\n    ')
        for (samples, mean, stdev) in points[architecture]:
            print(f'({samples},{mean:.2f}) ', end='')
        print(f'\n}};')
    
    for architecture in points:
        print(f'\\def\\{architecture}Min{{', end='\n    ')
        for (samples, mean, stdev) in points[architecture]:
            print(f'({samples},{(mean - stdev):.2f}) ', end='')
        print(f'\n}};')
    
    for architecture in points:
        print(f'\\def\\{architecture}Max{{', end='\n    ')
        for (samples, mean, stdev) in points[architecture]:
            print(f'({samples},{(mean + stdev):.2f}) ', end='')
        print(f'\n}};')


def main():
    data = {}
    for root, dirs, files in os.walk(dir_braid):
        for file in files:
            if file == 'training.txt':
                architecture = root.split('/')[-1]
                filename = f'{root}/{file}'
                read_values(filename, architecture, data)

    points = compute_points(data)
    print_points(points)

if __name__ == "__main__":
    main()

