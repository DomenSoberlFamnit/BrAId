import os
import csv
import numpy as np

dir_results = '/home/hicup/disk/res-noaug-50/results/'

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
            
            epoch = int(int(row[0].strip()))
            loss = round(float(row[2].strip()), 4)

            if epoch not in values:
                values[epoch] = []
            values[epoch].append(loss)

    csvfile.close()

def main():
    data = {}
    for root, dirs, files in os.walk(dir_results):
        for file in files:
            if file == 'training.txt':
                architecture = root.split('/')[-1]
                filename = f'{root}/{file}'
                read_values(filename, architecture, data)

    for architecture in data:
        print(f'\\def\\{architecture}{{\n    ', end='')
        values = data[architecture]
        for epoch in values:
            value = np.max(values[epoch])
            print(f'({epoch},{np.mean(value):.2f})', end=' ')
        print(f'\n}};')

    quit()

    by_epoch = {}

    for architecture in data:
        values = data[architecture]
        for epoch in values:
            value = np.min(values[epoch])
            if epoch not in by_epoch:
                by_epoch[epoch] = []
            by_epoch[epoch].append(value)

    print(f'\\def\\mean{{')
    for epoch in by_epoch:
        values = by_epoch[epoch]
        print(f'({epoch},{np.mean(values):.2f})', end=' ')
    print(f'\n}};')

    print(f'\\def\\min{{')
    for epoch in by_epoch:
        values = by_epoch[epoch]
        print(f'({epoch},{np.min(values):.2f})', end=' ')
    print(f'\n}};')

    print(f'\\def\\max{{')
    for epoch in by_epoch:
        values = by_epoch[epoch]
        print(f'({epoch},{np.max(values):.2f})', end=' ')
    print(f'\n}};')

if __name__ == "__main__":
    main()
