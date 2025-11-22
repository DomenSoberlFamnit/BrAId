import numpy as np
import dataset

dir_braid = '/home/hicup/disk/braid/'

# Load the dataset.
X, Y = dataset.load_training_samples(dir_braid)

# Dataset statistics
print('Computing dataset statistics.')

cnt = 0
distances = {}

for y in Y:
    i0 = None
    for i, p in enumerate(y):
        if round(p, 2) != 0:
             if i0 != None:
                d = i - i0
                i0 = i
                if d not in distances:
                    distances[d] = 0
                distances[d] += 1
            else:
                i0 = i
    
    cnt += 1
    if cnt % 1000 == 0:
        print(f'{cnt}/{len(Y)}')

dist = np.array(list(distances.keys()))
print('Minimal axle distance:', np.min(dist))

#counts, bin_edges = np.histogram(np.array(values), bins=20)

#print('Histogram:')
#for (count, edge) in zip(counts, bin_edges):
#    print(edge, count)
